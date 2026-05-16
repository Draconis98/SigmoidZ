from __future__ import annotations

import argparse

import torch

from config import load_config
from data import RandomTokenBatcher, load_token_ids
from model import CausalLM
from utils import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--preset", type=str, default="tiny")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config, args.preset)
    device = resolve_device(cfg.train.device)
    token_ids, vocab_size = load_token_ids(cfg.train, cfg.model)
    if cfg.train.dataset_name is None:
        cfg.model.vocab_size = vocab_size
    model = CausalLM(cfg.model).to(device)
    batcher = RandomTokenBatcher(token_ids, cfg.model.context_length, cfg.train.batch_size, device)
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(args.steps):
            x, y = batcher.next()
            _, loss = model(x, y)
            assert loss is not None
            losses.append(loss.item())
    print(f"loss={sum(losses) / len(losses):.4f}")


if __name__ == "__main__":
    main()
