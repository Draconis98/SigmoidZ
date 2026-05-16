from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch

from config import ModelConfig, TrainConfig


SMOKE_TEXT = """
SigmoidZ treats each hidden coordinate as the mean of a Bernoulli latent variable.
The centered mean 2 sigmoid(x) - 1 has the same range as tanh and a direct MFVI interpretation.
This tiny corpus is only for correctness checks, not for model quality.
"""


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


class Batcher(Protocol):
    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        ...


def _load_tokenizer(train_cfg: TrainConfig):
    from transformers import AutoTokenizer

    assert train_cfg.tokenizer_name is not None, "dataset training requires tokenizer_name"
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_token_ids(train_cfg: TrainConfig, model_cfg: ModelConfig) -> tuple[torch.Tensor, int]:
    if train_cfg.text_file is not None:
        text = Path(train_cfg.text_file).read_text()
        tokenizer = ByteTokenizer()
        ids = tokenizer.encode(text)
        return torch.tensor(ids, dtype=torch.long), tokenizer.vocab_size

    if train_cfg.dataset_name is None:
        tokenizer = ByteTokenizer()
        ids = tokenizer.encode(SMOKE_TEXT * 256)
        return torch.tensor(ids, dtype=torch.long), tokenizer.vocab_size

    from datasets import load_dataset

    tokenizer = _load_tokenizer(train_cfg)
    dataset = load_dataset(train_cfg.dataset_name, train_cfg.dataset_config, split=train_cfg.dataset_split)
    text_column = "text"
    assert text_column in dataset.column_names, f"Expected a text column, got {dataset.column_names}"
    ids: list[int] = []
    for sample in dataset[text_column]:
        if not isinstance(sample, str) or not sample.strip():
            continue
        ids.extend(tokenizer.encode(sample, add_special_tokens=False))
        if train_cfg.max_train_tokens is not None and len(ids) >= train_cfg.max_train_tokens:
            ids = ids[: train_cfg.max_train_tokens]
            break
    assert len(ids) > model_cfg.context_length + 1, "dataset is too small for the configured context length"
    return torch.tensor(ids, dtype=torch.long), int(len(tokenizer))


class RandomTokenBatcher:
    def __init__(self, token_ids: torch.Tensor, context_length: int, batch_size: int, device: torch.device) -> None:
        assert token_ids.ndim == 1
        assert token_ids.numel() > context_length + 1
        self.token_ids = token_ids
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(0, self.token_ids.numel() - self.context_length - 1, (self.batch_size,))
        rows = [self.token_ids[s : s + self.context_length + 1] for s in starts]
        batch = torch.stack(rows).to(self.device)
        return batch[:, :-1], batch[:, 1:]


class StreamingTokenBatcher:
    def __init__(
        self,
        train_cfg: TrainConfig,
        model_cfg: ModelConfig,
        batch_size: int,
        device: torch.device,
        rank: int,
        world_size: int,
    ) -> None:
        from datasets import load_dataset

        assert train_cfg.dataset_name is not None
        self.tokenizer = _load_tokenizer(train_cfg)
        dataset = load_dataset(
            train_cfg.dataset_name,
            train_cfg.dataset_config,
            split=train_cfg.dataset_split,
            streaming=True,
        )
        if world_size > 1:
            dataset = dataset.shard(num_shards=world_size, index=rank)
        self.samples = iter(dataset)
        self.context_length = model_cfg.context_length
        self.batch_size = batch_size
        self.device = device
        self.max_train_tokens = train_cfg.max_train_tokens
        self.tokens_seen = 0
        self.buffer: list[int] = []
        self.text_column = "text"

    @property
    def vocab_size(self) -> int:
        return int(len(self.tokenizer))

    def _fill(self, target_tokens: int) -> None:
        while len(self.buffer) < target_tokens:
            sample = next(self.samples)
            assert self.text_column in sample, f"Expected a text column, got {sample.keys()}"
            text = sample[self.text_column]
            if not isinstance(text, str) or not text.strip():
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if self.max_train_tokens is not None:
                remaining = self.max_train_tokens - self.tokens_seen
                assert remaining > 0, "max_train_tokens exhausted before max_steps completed"
                ids = ids[:remaining]
            self.buffer.extend(ids)
            self.tokens_seen += len(ids)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        needed = self.batch_size * (self.context_length + 1)
        self._fill(needed)
        rows = []
        for _ in range(self.batch_size):
            row = self.buffer[: self.context_length + 1]
            del self.buffer[: self.context_length]
            rows.append(torch.tensor(row, dtype=torch.long))
        batch = torch.stack(rows).to(self.device)
        return batch[:, :-1], batch[:, 1:]


def build_batcher(
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[Batcher, int]:
    if train_cfg.dataset_name is not None and train_cfg.streaming:
        batcher = StreamingTokenBatcher(train_cfg, model_cfg, train_cfg.batch_size, device, rank, world_size)
        return batcher, batcher.vocab_size
    token_ids, vocab_size = load_token_ids(train_cfg, model_cfg)
    return RandomTokenBatcher(token_ids, model_cfg.context_length, train_cfg.batch_size, device), vocab_size
