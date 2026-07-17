"""Autoregressive decode with KV cache."""

import time
import torch


def sample_next(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Sample one token from a categorical distribution over the vocab.

    logits: [1, vocab_size]
    returns: [1, 1] token id
    """
    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    prob = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(prob, num_samples=1)


def eos_token_ids(tokenizer) -> set[int]:
    """
    LLaMA-3-Instruct ends turns with <|eot_id|> (128009), NOT the 'true'
    EOS <|end_of_text|> (128001). If you only check tokenizer.eos_token_id
    the model won't stop and you'll waste tokens on garbage.
    """
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot != tokenizer.unk_token_id:
        ids.add(eot)
    return ids


@torch.inference_mode()
def autoregressive_decode(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
):
    """
    Standard autoregressive decode with KV cache.

    Returns:
        generated: [1, prompt_len + num_generated] token ids
        stats: dict with prompt_len, prefill_time, decode_time, tokens_generated
    """
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
    generated = input_ids.clone()
    prompt_len = input_ids.shape[1]
    eos_ids = eos_token_ids(tokenizer)

    # ----- Prefill -----
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    output = model(
        input_ids=generated, past_key_values=None, use_cache=True
    )
    past_key_values = output.past_key_values         
    first_logits = output.logits[:, -1, :]             
    # Shape note: output.logits is [batch, seq_len, vocab_size].
    # After [:, -1, :] it's [batch, vocab_size] — one distribution per batch row.

    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - t0

    # First token comes "free" from prefill's logits — NOT counted in decode time.
    next_token = sample_next(first_logits, temperature)
    generated = torch.cat([generated, next_token], dim=-1)
    num_generated = 1

    if next_token.item() in eos_ids:
        return generated, {
            "prompt_len": prompt_len,
            "prefill_time": prefill_time,
            "decode_time": 0.0,
            "tokens_generated": num_generated,
        }

    # ----- Decode: one token at a time, only new token goes in each step -----
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    for _ in range(max_new_tokens - 1):
        outputs = model(
            input_ids=generated[:, -1:],
            past_key_values=past_key_values,
            use_cache=True,
        )

        past_key_values = outputs.past_key_values
        next_tokens = sample_next(outputs.logits[:, -1, :], temperature)
        generated = torch.cat([generated, next_tokens], dim=-1)
        num_generated += 1

        if next_tokens.item() in eos_ids:              
            break

    torch.cuda.synchronize()
    decode_time = time.perf_counter() - t1            

    return generated, {
        "prompt_len": prompt_len,
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "tokens_generated": num_generated,
    }