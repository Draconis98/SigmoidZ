import torch

from config import ModelConfig
from model import CausalLM


def test_model_forward_conservative() -> None:
    cfg = ModelConfig(context_length=16, hidden_size=32, num_layers=2, num_heads=4, intermediate_size=64)
    model = CausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss is not None


def test_model_forward_research() -> None:
    cfg = ModelConfig(context_length=16, hidden_size=32, num_layers=2, num_heads=4, intermediate_size=64, block_variant="research")
    model = CausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss is not None
