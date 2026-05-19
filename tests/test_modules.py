import torch

from modules import CausalSelfAttention, DyT, SigmoidZAttentionUpdate, SigmoidZNorm


def test_sigmoidz_matches_dyt_when_bias_zero() -> None:
    x = torch.randn(4, 8, 16)
    dyt = DyT(16, alpha=0.7)
    sig = SigmoidZNorm(16, alpha=0.7)
    with torch.no_grad():
        sig.weight.copy_(dyt.weight)
        sig.bias.copy_(dyt.bias)
        sig.alpha.copy_(dyt.alpha)
        sig.logit_bias.zero_()
    torch.testing.assert_close(sig(x), dyt(x), atol=1e-6, rtol=1e-6)


def test_sigmoidz_backward() -> None:
    x = torch.randn(2, 4, 8, requires_grad=True)
    norm = SigmoidZNorm(8)
    y = norm(x).sum()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_attention_message_matches_forward_shape() -> None:
    x = torch.randn(2, 6, 16)
    attn = CausalSelfAttention(dim=16, num_heads=4, dropout=0.0, max_seq_len=8)
    assert attn.message(x).shape == x.shape
    assert attn(x).shape == x.shape


def test_sigmoidz_attention_update_backward() -> None:
    x = torch.randn(2, 6, 16, requires_grad=True)
    update = SigmoidZAttentionUpdate(dim=16, num_heads=4, dropout=0.0, max_seq_len=8)
    y = update(x).sum()
    y.backward()
    assert x.grad is not None
    assert update.logit_bias.grad is not None
    assert torch.isfinite(x.grad).all()
