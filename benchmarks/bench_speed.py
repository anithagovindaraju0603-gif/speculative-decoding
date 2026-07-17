"""
Phase 1 speed benchmark.

Loads the target model, runs the baseline decode over a variety of
prompts, prints a summary, and saves per-prompt results to
results/baseline.json.

Run from the project root:
    python benchmarks/bench_speed.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make src/ importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from baseline import autoregressive_decode


# ---------- Config ----------

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DTYPE = torch.bfloat16
DEVICE = "cuda"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
SEED = 0
RESULTS_PATH = ROOT / "results" / "baseline.json"


# ---------- Prompts ----------
# Grouped by category so Phase 4 can slice results by task type.
# When we get to Phase 2, this dict moves into a shared benchmarks/prompts.py
# so bench_acceptance.py and bench_by_prompt_type.py can import it too.

PROMPTS = {
    "factual": [
        "What is the capital of France, and what is it best known for?",
        "Explain in one paragraph how photosynthesis works.",
    ],
    "creative": [
        "Write a short story about a lighthouse keeper who receives a mysterious letter.",
        "Compose a poem about the changing seasons.",
    ],
    "code": [
        "Write a Python function that computes the nth Fibonacci number using memoization.",
        "Show, with code, how to reverse a singly linked list in Python.",
    ],
    "reasoning": [
        "If a train leaves Chicago at 3pm going 60mph and another leaves New York at 4pm going 80mph, when do they meet? Cities are 790 miles apart.",
        "Give three arguments for and against a universal basic income.",
    ],
}


# ---------- Helpers ----------

def format_prompt(tokenizer, user_msg: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )


def run_one(model, tokenizer, user_msg: str) -> dict:
    prompt = format_prompt(tokenizer, user_msg)
    generated, stats = autoregressive_decode(
        model, tokenizer, prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
    output = tokenizer.decode(
        generated[0, stats["prompt_len"]:], skip_special_tokens=True
    )

    # Decode-time throughput excludes the first token (came from prefill).
    n = max(stats["tokens_generated"] - 1, 1)
    tps = n / stats["decode_time"] if stats["decode_time"] > 0 else 0.0
    ms_per_tok = 1000.0 * stats["decode_time"] / n if stats["decode_time"] > 0 else 0.0

    return {
        "prompt": user_msg,
        "output_text": output,
        "prompt_len": stats["prompt_len"],
        "tokens_generated": stats["tokens_generated"],
        "prefill_time": stats["prefill_time"],
        "decode_time": stats["decode_time"],
        "tokens_per_sec": tps,
        "ms_per_token": ms_per_tok,
    }


# ---------- Main ----------

def main():
    torch.manual_seed(SEED)

    print(f"Loading {MODEL_NAME} ({DTYPE})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map=DEVICE
    )
    model.eval()
    print(f"Loaded: {torch.cuda.memory_allocated()/1e9:.1f} GB on {model.device}")

    # Warmup — first CUDA kernels carry setup cost.
    print("Warmup...")
    autoregressive_decode(
        model, tokenizer, format_prompt(tokenizer, "Hi."), max_new_tokens=16
    )

    # Benchmark
    results = []
    for category, prompts in PROMPTS.items():
        print(f"\n[{category}]")
        for user_msg in prompts:
            r = run_one(model, tokenizer, user_msg)
            r["category"] = category
            results.append(r)
            print(f"  {r['tokens_generated']:>3} toks | "
                  f"prefill {r['prefill_time']*1000:6.1f} ms | "
                  f"decode {r['tokens_per_sec']:5.1f} tok/s")

    # Aggregate
    tps_all = sorted(r["tokens_per_sec"] for r in results)
    median = tps_all[len(tps_all) // 2]
    mean = sum(tps_all) / len(tps_all)

    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["tokens_per_sec"])

    print("\n=== Summary ===")
    print(f"Median tok/s: {median:.2f}")
    print(f"Mean   tok/s: {mean:.2f}")
    print(f"Min/Max:      {min(tps_all):.2f} / {max(tps_all):.2f}")
    print("Per category (median):")
    for cat, vals in by_cat.items():
        v = sorted(vals)
        print(f"  {cat:>10}: {v[len(v)//2]:.2f} tok/s")

    # Save
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "config": {
                "model": MODEL_NAME,
                "dtype": str(DTYPE),
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
                "seed": SEED,
            },
            "summary": {
                "median_tokens_per_sec": median,
                "mean_tokens_per_sec": mean,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nSaved {RESULTS_PATH}")


if __name__ == "__main__":
    main()