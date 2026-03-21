"""GPT-2 provider -- model + tokenizer."""
from __future__ import annotations


def load_gpt2() -> tuple:
    """Returns (model, tokenizer) on the appropriate device."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    return model, tokenizer
