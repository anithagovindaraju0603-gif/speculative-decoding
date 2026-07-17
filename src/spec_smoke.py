import torch
from models import load_pair
from spec_decode_vanilla import speculative_decode

print("Loading models...")
target, draft, tokenizer = load_pair()
print(f"Loaded: {torch.cuda.memory_allocated()/1e9:.1f} GB on {target.device}")

PROMPTS = [
    "Explain photosynthesis in one paragraph.",
    "Write a Python function that computes the nth Fibonacci number using memoization.",
    "Write a short story about a lighthouse keeper.",
    "What is the capital of France, and what is it best known for?",
    "Compose a poem about the changing seasons.",
]

print("\nRunning speculative decode on 5 prompts...")
torch.manual_seed(0)

all_stats = []
for i, user_msg in enumerate(PROMPTS):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True,
    )
    generated, stats = speculative_decode(
        target, draft, tokenizer, prompt,
        max_new_tokens=256, K=4, temperature=0.7,
    )
    tps = stats["tokens_generated"] / stats["decode_time"]
    acc = stats["accepted"] / max(stats["accepted"] + stats["rejected"], 1)
    tokens_per_call = stats["tokens_generated"] / stats["big_model_calls"]
    print(f"  [{i+1}] {stats['tokens_generated']:>3} toks | "
          f"acc {acc:.1%} | "
          f"tok/verify {tokens_per_call:.2f} | "
          f"{tps:.1f} tok/s")
    all_stats.append((tps, acc, tokens_per_call))

median_tps = sorted(t for t, _, _ in all_stats)[len(all_stats) // 2]
median_acc = sorted(a for _, a, _ in all_stats)[len(all_stats) // 2]
print(f"\nMedian tok/s: {median_tps:.2f}")
print(f"Median acceptance: {median_acc:.1%}")
print(f"Baseline (from earlier): 52.80 tok/s")
print(f"Speedup: {median_tps / 52.80:.2f}x")