from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y * self.weight


class DyT(nn.Module):
    def __init__(self, dim: int, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.alpha * x) * self.weight + self.bias


class SigmoidZNorm(nn.Module):
    """Centered Bernoulli mean parameterization: 2 * sigmoid(2 alpha x + beta) - 1."""

    def __init__(self, dim: int, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.logit_bias = nn.Parameter(torch.zeros(dim))
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(2.0 * self.alpha * x + self.logit_bias)
        centered = 2.0 * z - 1.0
        return centered * self.weight + self.bias


def make_norm(norm_type: str, dim: int, alpha: float) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    if norm_type == "dyt":
        return DyT(dim, alpha)
    if norm_type == "sigmoidz":
        return SigmoidZNorm(dim, alpha)
    assert False, f"Unknown norm_type: {norm_type}"


def precompute_rope_frequencies(head_dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    freqs = torch.outer(positions, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    bsz, seq_len, n_heads, head_dim = x.shape
    x_pair = x.float().reshape(bsz, seq_len, n_heads, head_dim // 2, 2)
    x_complex = torch.view_as_complex(x_pair)
    freqs = freqs_cis[:seq_len].to(x.device).view(1, seq_len, 1, head_dim // 2)
    rotated = torch.view_as_real(x_complex * freqs).flatten(3)
    return rotated.type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, max_seq_len: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
        self.register_buffer("freqs_cis", precompute_rope_frequencies(self.head_dim, max_seq_len), persistent=False)

    def message(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, self.freqs_cis).transpose(1, 2)
        k = apply_rope(k, self.freqs_cis).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        return y.transpose(1, 2).contiguous().view(bsz, seq_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.message(x))


class SigmoidZAttentionUpdate(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, max_seq_len: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk = nn.Linear(dim, 2 * dim, bias=False)
        self.neighbor_logits = nn.Linear(dim, dim, bias=False)
        self.unary = nn.Linear(dim, dim, bias=False)
        self.pairwise_active = nn.Linear(dim, dim, bias=False)
        self.pairwise_inactive = nn.Linear(dim, dim, bias=False)
        self.neighbor_logit_bias = nn.Parameter(torch.zeros(dim))
        self.logit_bias = nn.Parameter(torch.zeros(dim))
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
        self.register_buffer("freqs_cis", precompute_rope_frequencies(self.head_dim, max_seq_len), persistent=False)

    def _attention_messages(
        self, q: torch.Tensor, k: torch.Tensor, active_state: torch.Tensor, inactive_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, dim = active_state.shape
        active_state = active_state.view(bsz, seq_len, self.num_heads, self.head_dim)
        inactive_state = inactive_state.view(bsz, seq_len, self.num_heads, self.head_dim)
        v = torch.cat([active_state, inactive_state], dim=-1).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        active_message, inactive_message = y.split(self.head_dim, dim=-1)
        active_message = active_message.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        inactive_message = inactive_message.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        return active_message, inactive_message

    def messages(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape
        q, k = self.qk(x).chunk(2, dim=-1)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim)
        q = apply_rope(q, self.freqs_cis).transpose(1, 2)
        k = apply_rope(k, self.freqs_cis).transpose(1, 2)
        neighbor_state = torch.sigmoid(self.neighbor_logits(x) + self.neighbor_logit_bias)
        return self._attention_messages(q, k, neighbor_state, 1.0 - neighbor_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active_message, inactive_message = self.messages(x)
        logits = (
            self.unary(x)
            + self.pairwise_active(active_message)
            + self.pairwise_inactive(inactive_message)
            + self.logit_bias
        )
        z = 2.0 * torch.sigmoid(logits) - 1.0
        return self.out(z)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        max_seq_len: int,
        attn_norm_type: str,
        ffn_norm_type: str,
        block_variant: str,
        alpha_attn: float,
        alpha_other: float,
    ) -> None:
        super().__init__()
        assert block_variant in {"conservative", "research"}
        self.block_variant = block_variant
        self.attn_norm = make_norm(attn_norm_type, dim, alpha_attn)
        self.ffn_norm = make_norm(ffn_norm_type, dim, alpha_other)
        if block_variant == "research":
            self.attn = SigmoidZAttentionUpdate(dim, num_heads, dropout, max_seq_len)
        else:
            self.attn = CausalSelfAttention(dim, num_heads, dropout, max_seq_len)
        self.ffn = FeedForward(dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
