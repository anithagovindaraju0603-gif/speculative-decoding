"""Vanilla speculative decoding — Leviathan et al. 2023."""

import time
import torch

from baseline import eos_token_ids
from rejection_sampling import rejection_sample


def _logits_to_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Convert logits to a temperature-scaled probability distribution (fp32).

    Handles T=0 as the mathematical limit (one-hot at argmax) rather than
    letting softmax(logits/0) blow up.
    """
    if temperature <= 0.0:
        argmax = logits.argmax(dim=-1, keepdim=True)
        probs = torch.zeros_like(logits, dtype=torch.float32)
        probs.scatter_(-1, argmax, 1.0)
        return probs
    return torch.softmax(logits.float() / temperature, dim=-1)


@torch.inference_mode()
def draft_k_tokens(draft_model, context, draft_kv_cache, K, temperature):
    """Run the draft model autoregressively for K steps."""
    device = draft_model.device
    vocab_size = draft_model.config.vocab_size

    draft_tokens = torch.empty(K, dtype=torch.long, device=device)
    draft_probs = torch.empty(K, vocab_size, dtype=torch.float32, device=device)

    input_ids = context
    for i in range(K):
        outputs = draft_model(
            input_ids=input_ids,
            past_key_values=draft_kv_cache,
            use_cache=True,
        )
        draft_kv_cache = outputs.past_key_values

        probs = _logits_to_probs(outputs.logits[:, -1, :], temperature)  # [1, V]
        token = torch.multinomial(probs, num_samples=1)                  # [1, 1]

        draft_tokens[i] = token.squeeze()
        draft_probs[i] = probs.squeeze(0)

        input_ids = token

    return draft_tokens, draft_probs, draft_kv_cache


@torch.inference_mode()
def verify_with_target(target_model, verify_input, target_kv_cache, K, temperature):
    """One forward pass of target; return [K+1, V] distributions + updated cache.

    Last K+1 output positions work for both cold cache (input=[prompt+drafts])
    and warm cache (input=[last_committed+drafts]) cases.
    """
    outputs = target_model(
        input_ids=verify_input,
        past_key_values=target_kv_cache,
        use_cache=True,
    )
    target_kv_cache = outputs.past_key_values
    logits = outputs.logits[0, -(K + 1):, :]                   # [K+1, V]
    target_probs = _logits_to_probs(logits, temperature)       # [K+1, V]

    return target_probs, target_kv_cache


@torch.inference_mode()
def speculative_decode(
    target_model,
    draft_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    K: int = 4,
    temperature: float = 0.7,
):
    """Vanilla speculative decoding with proper KV cache management."""
    device = target_model.device
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    prompt_len = input_ids.shape[1]
    generated = input_ids.clone()
    eos_ids = eos_token_ids(tokenizer)

    target_kv_cache = None
    draft_kv_cache = None

    stats = {
        "prompt_len": prompt_len,
        "prefill_time": 0.0,       # folded into decode_time; kept for schema parity
        "decode_time": 0.0,
        "tokens_generated": 0,
        "accepted": 0,
        "rejected": 0,
        "big_model_calls": 0,
        "draft_model_calls": 0,
    }

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    stop = False
    while stats["tokens_generated"] < max_new_tokens and not stop:
        # ---- Determine draft input: whatever the draft hasn't processed yet ----
        # On iter 1: full generated (cache is None).
        # Typical iter: last 1 token (cache holds generated.shape[1] - 1).
        # After an all-accept iter: last 2 tokens (draft is 1 behind the invariant).
        if draft_kv_cache is None:
            draft_input = generated
        else:
            draft_prefix_len = draft_kv_cache.get_seq_length()
            draft_input = generated[:, draft_prefix_len:]

        # ---- Draft phase ----
        draft_tokens, draft_probs, draft_kv_cache = draft_k_tokens(
            draft_model, draft_input, draft_kv_cache, K, temperature
        )
        stats["draft_model_calls"] += K

        # ---- Verify phase input ----
        # Iter 1 (cold): [prompt, d_1..d_K].  Later (warm): [last_committed, d_1..d_K].
        if target_kv_cache is None:
            verify_input = torch.cat(
                [generated, draft_tokens.unsqueeze(0)], dim=-1
            )
        else:
            verify_input = torch.cat(
                [generated[:, -1:], draft_tokens.unsqueeze(0)], dim=-1
            )

        target_probs, target_kv_cache = verify_with_target(
            target_model, verify_input, target_kv_cache, K, temperature
        )
        stats["big_model_calls"] += 1

        # ---- Rejection sampling ----
        n_accepted, replacement = rejection_sample(
            draft_tokens, draft_probs, target_probs
        )
        stats["accepted"] += n_accepted
        if n_accepted < K:
            stats["rejected"] += 1  # per-paper: 1 rejection event per verify call, rest of teh tokens are just unchecked(because by default rejected)

        # ---- Commit: accepted drafts + replacement (or bonus) ----
        new_tokens = torch.cat([draft_tokens[:n_accepted], replacement], dim=0)
        generated = torch.cat([generated, new_tokens.unsqueeze(0)], dim=-1)
        stats["tokens_generated"] += new_tokens.shape[0]

        # ---- Truncate caches to restore the invariant ----
        # Target: always exactly generated.shape[1] - 1 (the last committed token is replacement/bonus — never processed by target).
        target_kv_cache.crop(generated.shape[1] - 1)

        # Draft: aim for the same. But if all-accept, draft cache is already 1
        # short (d_K was sampled but not fed). Only crop when we need to shrink.
        target_draft_len = generated.shape[1] - 1
        if draft_kv_cache.get_seq_length() > target_draft_len:
            draft_kv_cache.crop(target_draft_len)

        # ---- EOS check on newly-appended tokens ----
        for tok in new_tokens.tolist():
            if tok in eos_ids:
                stop = True
                break

    torch.cuda.synchronize()
    stats["decode_time"] = time.perf_counter() - t0

    return generated, stats