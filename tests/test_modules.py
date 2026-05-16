import torch

from modules import DyT, SigmoidZNorm


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
