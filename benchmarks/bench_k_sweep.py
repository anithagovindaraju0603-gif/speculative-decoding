"""K sweep — vary speculation length, measure tokens/sec + acceptance rate.

Expected: speed peaks at some intermediate K (typically 4-8). Too low means
insufficient parallelism; too high means acceptance collapses and draft work
is wasted.
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model import load_pair
from spec_decode_vanilla import speculative_decode


K_VALUES = [1, 2, 3, 4, 6, 8, 10, 12]
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 256
SEED = 0

PROMPTS = [
    "Explain photosynthesis in one paragraph.",
    "Write a Python function that computes the nth Fibonacci number using memoization.",
    "Write a short story about a lighthouse keeper who receives a mysterious letter.",
    "What is the capital of France, and what is it best known for?",
    "Compose a poem about the changing seasons.",
    "If a train leaves Chicago at 3pm going 60mph and another leaves New York at 4pm going 80mph, when do they meet? Cities are 790 miles apart.",
    "Show, with code, how to reverse a singly linked list in Python.",
    "Give three arguments for and against a universal basic income.",
]


def main():
    target, draft, tokenizer = load_pair()
    print(f"Loaded: {torch.cuda.memory_allocated()/1e9:.1f} GB\n")

    results = {}
    for K in K_VALUES:
        print(f"=== K = {K} ===")
        per_prompt = []
        for i, user_msg in enumerate(PROMPTS):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False, add_generation_prompt=True,
            )
            torch.manual_seed(SEED + i)
            generated, stats = speculative_decode(
                target, draft, tokenizer, prompt,
                max_new_tokens=MAX_NEW_TOKENS, K=K, temperature=TEMPERATURE,
            )
            tps = stats["tokens_generated"] / stats["decode_time"]
            acc = stats["accepted"] / max(stats["accepted"] + stats["rejected"], 1)
            tpc = stats["tokens_generated"] / stats["big_model_calls"]
            per_prompt.append({
                "tokens_generated": stats["tokens_generated"],
                "acceptance_rate": acc,
                "tokens_per_verify": tpc,
                "tokens_per_sec": tps,
            })
            print(f"  [{i+1}] {stats['tokens_generated']:>3} toks | "
                  f"acc {acc:.1%} | tok/verify {tpc:.2f} | {tps:.1f} tok/s")

        # Aggregate
        tps_list = sorted(p["tokens_per_sec"] for p in per_prompt)
        acc_list = sorted(p["acceptance_rate"] for p in per_prompt)
        tpc_list = sorted(p["tokens_per_verify"] for p in per_prompt)
        median = lambda xs: xs[len(xs) // 2]

        summary = {
            "K": K,
            "median_tokens_per_sec": median(tps_list),
            "median_acceptance_rate": median(acc_list),
            "median_tokens_per_verify": median(tpc_list),
        }
        results[K] = {"summary": summary, "per_prompt": per_prompt}
        print(f"  --> median {median(tps_list):.2f} tok/s, "
              f"acc {median(acc_list):.1%}, tok/verify {median(tpc_list):.2f}\n")

    # Print sweep table
    print("=== K sweep summary ===")
    print(f"{'K':>3} | {'tok/s':>7} | {'acc':>7} | {'tok/verify':>10}")
    print("-" * 40)
    for K in K_VALUES:
        s = results[K]["summary"]
        print(f"{K:>3} | {s['median_tokens_per_sec']:>7.2f} | "
              f"{s['median_acceptance_rate']:>6.1%} | "
              f"{s['median_tokens_per_verify']:>10.2f}")

    best_K = max(K_VALUES, key=lambda k: results[k]["summary"]["median_tokens_per_sec"])
    print(f"\nBest K: {best_K} @ {results[best_K]['summary']['median_tokens_per_sec']:.2f} tok/s")
    print(f"Baseline (from Phase 1): 52.80 tok/s")
    print(f"Best speedup: {results[best_K]['summary']['median_tokens_per_sec'] / 52.80:.2f}x")

    out = ROOT / "results" / "k_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()