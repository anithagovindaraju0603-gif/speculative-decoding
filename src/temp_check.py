import torch
from models import load_pair
from spec_decode_vanilla import speculative_decode, _logits_to_probs

target, draft, tokenizer = load_pair()

# 1. Sanity-check the helper directly on random logits
print("=== Helper sanity ===")
fake_logits = torch.randn(1, 128256, device=target.device)
p_hot = _logits_to_probs(fake_logits, temperature=0.0)
p_warm = _logits_to_probs(fake_logits, temperature=0.7)
p_flat = _logits_to_probs(fake_logits, temperature=2.0)

print(f"T=0.0: max prob = {p_hot.max().item():.4f} (should be 1.0)")
print(f"T=0.7: max prob = {p_warm.max().item():.4f} (should be small, e.g. 0.01-0.5)")
print(f"T=2.0: max prob = {p_flat.max().item():.4f} (should be even smaller)")
print(f"T=0.7: entropy  = {-(p_warm * (p_warm.clamp(min=1e-20)).log()).sum().item():.2f}")
print(f"       (near 0 = greedy-like, log(V)={torch.tensor(128256.0).log().item():.2f} = uniform)")

# 2. Run same prompt TWICE with T=0.7. Outputs should differ.
print("\n=== Stochasticity check ===")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Write one sentence about the ocean."}],
    tokenize=False, add_generation_prompt=True,
)

torch.manual_seed(42)
gen1, _ = speculative_decode(target, draft, tokenizer, prompt,
                              max_new_tokens=40, K=4, temperature=0.7)

torch.manual_seed(43)
gen2, _ = speculative_decode(target, draft, tokenizer, prompt,
                              max_new_tokens=40, K=4, temperature=0.7)

text1 = tokenizer.decode(gen1[0, gen1.shape[1]-40:], skip_special_tokens=True)
text2 = tokenizer.decode(gen2[0, gen2.shape[1]-40:], skip_special_tokens=True)
print(f"Run 1 (seed 42): {text1}")
print(f"Run 2 (seed 43): {text2}")
print(f"\nIdentical? {text1 == text2} (should be False — T=0.7 is stochastic)")

# 3. Run with T=0.0 twice — should be IDENTICAL (proves T=0 path works)
print("\n=== Greedy determinism check ===")
torch.manual_seed(1)
gen3, _ = speculative_decode(target, draft, tokenizer, prompt,
                              max_new_tokens=40, K=4, temperature=0.0)
torch.manual_seed(2)
gen4, _ = speculative_decode(target, draft, tokenizer, prompt,
                              max_new_tokens=40, K=4, temperature=0.0)

text3 = tokenizer.decode(gen3[0, gen3.shape[1]-40:], skip_special_tokens=True)
text4 = tokenizer.decode(gen4[0, gen4.shape[1]-40:], skip_special_tokens=True)
print(f"Run 3 (T=0, seed 1): {text3}")
print(f"Run 4 (T=0, seed 2): {text4}")
print(f"\nIdentical? {text3 == text4} (should be True — T=0 is deterministic)")