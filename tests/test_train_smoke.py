import torch

from config import tiny_config
from data import RandomTokenBatcher, load_token_ids
from model import CausalLM


def test_one_training_step() -> None:
    cfg = tiny_config()
    cfg.model.context_length = 16
    cfg.train.batch_size = 2
    token_ids, vocab_size = load_token_ids(cfg.train, cfg.model)
    cfg.model.vocab_size = vocab_size
    model = CausalLM(cfg.model)
    batcher = RandomTokenBatcher(token_ids, cfg.model.context_length, cfg.train.batch_size, torch.device("cpu"))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, y = batcher.next()
    _, loss = model(x, y)
    assert loss is not None
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
