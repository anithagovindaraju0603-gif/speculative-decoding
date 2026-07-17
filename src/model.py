"""Model + tokenizer loading for speculative decoding experiments.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DRAFT_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DTYPE = torch.bfloat16
DEVICE = "cuda"


def load_tokenizer():
    """Both models share a tokenizer — load once from the target."""
    return AutoTokenizer.from_pretrained(TARGET_NAME)


def load_target():
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_NAME, torch_dtype=DTYPE, device_map=DEVICE
    )
    model.eval()
    return model


def load_draft():
    model = AutoModelForCausalLM.from_pretrained(
        DRAFT_NAME, torch_dtype=DTYPE, device_map=DEVICE
    )
    model.eval()
    return model


def load_pair():
    """Convenience: (target, draft, tokenizer) on the same device."""
    tokenizer = load_tokenizer()
    target = load_target()
    draft = load_draft()
    return target, draft, tokenizer