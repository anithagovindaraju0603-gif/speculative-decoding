"""Verify that speculative decoding produces the same output distribution
as the baseline autoregressive decoder.

Method: for each of many prompts, sample N tokens both ways (baseline and 
spec decode), build empirical distributions of the first generated token,
compute KL divergence. Should be near zero.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model import load_pair
from baseline import autoregressive_decode
from spec_decode_vanilla import speculative_decode


N_SAMPLES = 300  # per prompt per method
TEMPERATURE = 0.7
K = 4

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "The quick brown fox",
    "Once upon a time",
]


def sample_first_token(fn, tokenizer, prompt, seed, **kwargs):
    """Run `fn`, return the first generated token id."""
    torch.manual_seed(seed)
    generated, stats = fn(**kwargs)
    return generated[0, stats["prompt_len"]].item()


def main():
    target, draft, tokenizer = load_pair()
    print(f"Loaded: {torch.cuda.memory_allocated()/1e9:.1f} GB\n")

    results = []
    for prompt_text in PROMPTS:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True,
        )
        print(f"Prompt: {prompt_text!r}")
        print(f"  Sampling {N_SAMPLES} tokens per method...")

        baseline_tokens = []
        spec_tokens = []

        for i in range(N_SAMPLES):
            # Baseline
            torch.manual_seed(i)
            gen, stats = autoregressive_decode(
                target, tokenizer, prompt,
                max_new_tokens=1, temperature=TEMPERATURE,
            )
            baseline_tokens.append(gen[0, stats["prompt_len"]].item())

            # Spec decode
            torch.manual_seed(i + 100000)  # different seed space
            gen, stats = speculative_decode(
                target, draft, tokenizer, prompt,
                max_new_tokens=1, K=K, temperature=TEMPERATURE,
            )
            spec_tokens.append(gen[0, stats["prompt_len"]].item())

        # Build empirical distributions
        vocab_size = target.config.vocab_size
        b_counts = torch.bincount(torch.tensor(baseline_tokens), minlength=vocab_size).float()
        s_counts = torch.bincount(torch.tensor(spec_tokens), minlength=vocab_size).float()
        b_dist = (b_counts + 1e-10) / (b_counts.sum() + 1e-10 * vocab_size)
        s_dist = (s_counts + 1e-10) / (s_counts.sum() + 1e-10 * vocab_size)

        # KL(spec || baseline) — how surprised is baseline by spec's output?
        kl = F.kl_div(b_dist.log(), s_dist, reduction="sum").item()

        # Also look at the top-K overlap
        b_top = set(torch.tensor(baseline_tokens).unique().tolist())
        s_top = set(torch.tensor(spec_tokens).unique().tolist())
        overlap = len(b_top & s_top) / len(b_top | s_top)

        # And the most common token from each
        b_mode = max(set(baseline_tokens), key=baseline_tokens.count)
        s_mode = max(set(spec_tokens), key=spec_tokens.count)

        result = {
            "prompt": prompt_text,
            "kl_divergence": kl,
            "unique_token_overlap": overlap,
            "baseline_mode": tokenizer.decode([b_mode]),
            "spec_mode": tokenizer.decode([s_mode]),
            "baseline_unique_count": len(b_top),
            "spec_unique_count": len(s_top),
        }
        results.append(result)

        print(f"  KL divergence:     {kl:.4f}   (near 0 = same distribution)")
        print(f"  Token overlap:     {overlap:.1%}")
        print(f"  Baseline top tok:  {tokenizer.decode([b_mode])!r}")
        print(f"  Spec top tok:      {tokenizer.decode([s_mode])!r}")
        print()

    print("=== Summary ===")
    max_kl = max(r["kl_divergence"] for r in results)
    print(f"Max KL divergence: {max_kl:.4f}")
    print(f"Pass threshold:    0.05")
    if max_kl < 0.05:
        print("✓ PASS — distributions are equivalent")
    else:
        print("✗ FAIL — distributions differ, investigate")

    out = ROOT / "results" / "distribution_equivalence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()