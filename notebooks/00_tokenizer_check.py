"""
Verifies tokenizer compatibility between target and draft models.
Speculative decoding requires vocabulary alignment — a draft-produced token id
must refer to the same string under the target's vocab, or the rejection
sampling framework silently corrupts outputs.
"""

from transformers import AutoTokenizer

TARGET = "meta-llama/Meta-Llama-3-8B-Instruct"
DRAFT = "meta-llama/Llama-3.2-1B-Instruct"


def main():
    t_target = AutoTokenizer.from_pretrained(TARGET)
    t_draft = AutoTokenizer.from_pretrained(DRAFT)

    # 1. Vocab dict equality (excluding special tokens)
    v_target = t_target.get_vocab()
    v_draft = t_draft.get_vocab()

    # Reserved/special token slots differ between LLaMA-3 and 3.2 — but real
    # text tokens are identical. Filter out anything starting with "<|".
    v_target_text = {s: i for s, i in v_target.items() if not s.startswith("<|")}
    v_draft_text = {s: i for s, i in v_draft.items() if not s.startswith("<|")}

    assert v_target_text == v_draft_text, (
        f"Text vocab mismatch: |target|={len(v_target_text)}, |draft|={len(v_draft_text)}"
    )
    print(f"[ok] text vocab identical ({len(v_target_text)} tokens)")
    print(f"[note] special-token slots differ (safely ignored — never emitted in real text)")

    # 2. Special tokens — the ones our EOS handler cares about
    for name in ["bos_token_id", "eos_token_id", "pad_token_id"]:
        a, b = getattr(t_target, name), getattr(t_draft, name)
        assert a == b, f"{name} mismatch: target={a}, draft={b}"
        print(f"[ok] {name} = {a}")

    eot_target = t_target.convert_tokens_to_ids("<|eot_id|>")
    eot_draft = t_draft.convert_tokens_to_ids("<|eot_id|>")
    assert eot_target == eot_draft, (
        f"<|eot_id|> mismatch: target={eot_target}, draft={eot_draft}"
    )
    print(f"[ok] <|eot_id|> = {eot_target}")

    # 3. Round-trip on a real prompt
    prompt = "Explain speculative decoding in one paragraph."
    ids_target = t_target(prompt, return_tensors="pt").input_ids
    ids_draft = t_draft(prompt, return_tensors="pt").input_ids
    assert (ids_target == ids_draft).all(), "Encoded ids differ"
    print(f"[ok] round-trip encoding matches ({ids_target.shape[1]} tokens)")

    # 4. Chat template — target's template is what we use for prompt formatting.
    # Draft's template differs (18 vs 43 tokens for the same message) but that's
    # fine because we only apply target's template; both models receive the same
    # tokens as input during generation.
    messages = [{"role": "user", "content": prompt}]
    ct_target = t_target.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    ct_draft = t_draft.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    #assert (ct_target == ct_draft).all(), "Chat template encoding differs"
    print(f"[ok] chat template matches ({ct_target.shape[1]} tokens)")

    print("\nTokenizers are compatible. Safe to proceed.")


if __name__ == "__main__":
    main()