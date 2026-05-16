from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from config import ModelConfig
from modules import TransformerBlock, make_norm


class CausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.hidden_size % cfg.num_heads == 0
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=cfg.hidden_size,
                    num_heads=cfg.num_heads,
                    hidden_dim=cfg.intermediate_size,
                    dropout=cfg.dropout,
                    max_seq_len=cfg.context_length,
                    norm_type=cfg.norm_type,
                    block_variant=cfg.block_variant,
                    alpha_attn=cfg.alpha_attn,
                    alpha_other=cfg.alpha_other,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.norm = make_norm(cfg.norm_type, cfg.hidden_size, cfg.alpha_other)
        self.output = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert input_ids.ndim == 2
        assert input_ids.size(1) <= self.cfg.context_length
        x = self.tok_embeddings(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.output(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return logits, loss

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def estimate_params(cfg: ModelConfig) -> int:
    model = CausalLM(cfg)
    return model.parameter_count()
