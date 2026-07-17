# Save as src/kl_noise_check.py

import torch
import torch.nn.functional as F
from model import load_pair
from baseline import autoregressive_decode

target, draft, tokenizer = load_pair()

prompt_text = "Once upon a time"
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt_text}],
    tokenize=False, add_generation_prompt=True,
)

def sample_batch(seed_offset, n=300):
    tokens = []
    for i in range(n):
        torch.manual_seed(seed_offset + i)
        gen, stats = autoregressive_decode(
            target, tokenizer, prompt, max_new_tokens=1, temperature=0.7,
        )
        tokens.append(gen[0, stats["prompt_len"]].item())
    return tokens

print("Sampling baseline batch A (seeds 0..299)...")
batch_a = sample_batch(0)
print("Sampling baseline batch B (seeds 1000..1299)...")
batch_b = sample_batch(1000)

vocab_size = target.config.vocab_size
def to_dist(tokens):
    counts = torch.bincount(torch.tensor(tokens), minlength=vocab_size).float()
    return (counts + 1e-10) / (counts.sum() + 1e-10 * vocab_size)

a_dist = to_dist(batch_a)
b_dist = to_dist(batch_b)
kl_ab = F.kl_div(a_dist.log(), b_dist, reduction="sum").item()
print(f"\nKL(baseline A || baseline B) = {kl_ab:.4f}")
print(f"Unique tokens A: {len(set(batch_a))}, B: {len(set(batch_b))}")
print(f"Overlap: {len(set(batch_a) & set(batch_b)) / len(set(batch_a) | set(batch_b)):.1%}")
print()
print("If this KL is close to 0.31, our spec-vs-baseline test is just seeing noise.")
print("If it's much smaller (say <0.05), then spec really diverges from baseline.")