import torch
from model import load_pair
from spec_decode_vanilla import speculative_decode

print("Loading models...")
target, draft, tokenizer = load_pair()
print(f"Loaded: {torch.cuda.memory_allocated()/1e9:.1f} GB on {target.device}")

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain photosynthesis in one paragraph."}],
    tokenize=False, add_generation_prompt=True
)

print("Running speculative decode...")
torch.manual_seed(0)
generated, stats = speculative_decode(
    target, draft, tokenizer, prompt,
    max_new_tokens=64, K=4, temperature=0.7,
)

print("\n=== Output ===")
print(tokenizer.decode(generated[0, stats["prompt_len"]:], skip_special_tokens=True))

print("\n=== Stats ===")
for k, v in stats.items():
    print(f"  {k}: {v}")

if stats["accepted"] + stats["rejected"] > 0:
    acc_rate = stats["accepted"] / (stats["accepted"] + stats["rejected"])
    print(f"\n  acceptance rate: {acc_rate:.2%}")
if stats["decode_time"] > 0:
    tps = stats["tokens_generated"] / stats["decode_time"]
    print(f"  tokens/sec:      {tps:.2f}")