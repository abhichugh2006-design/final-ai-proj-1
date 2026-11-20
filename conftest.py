"""A tiny, randomly initialised GPT-2 and a byte-level tokenizer, built offline.

The real tool uses pretrained `gpt2` from Hugging Face, but tests must run fast and without
internet (GitHub Actions included), so they use the same architecture at a toy size.
"""
import pytest
import torch


def _byte_vocab() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    vocab = {chr(c): i for i, c in enumerate(cs)}
    vocab["<|endoftext|>"] = len(vocab)
    return vocab


@pytest.fixture
def tiny():
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

    torch.manual_seed(0)
    tokenizer = GPT2Tokenizer(vocab=_byte_vocab(), merges=[])
    config = GPT2Config(vocab_size=len(tokenizer), n_positions=512, n_embd=32, n_layer=2, n_head=2,
                        bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    model = GPT2LMHeadModel(config)
    model.eval()
    return tokenizer, model
