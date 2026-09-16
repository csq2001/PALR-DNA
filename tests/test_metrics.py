import torch
from pytorch_msssim import ms_ssim as standard_ms_ssim

from utils.metrics import ms_ssim, ms_ssim_loss, ms_ssim_tensor


def test_ms_ssim_matches_standard_library() -> None:
    generator = torch.Generator().manual_seed(42)
    x = torch.rand((1, 3, 256, 256), generator=generator)
    y = torch.rand((1, 3, 256, 256), generator=generator)

    expected = standard_ms_ssim(
        x,
        y,
        data_range=1.0,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
    )

    assert torch.equal(ms_ssim_tensor(x, y), expected)
    assert ms_ssim(x, y) == expected.item()


def test_ms_ssim_identity_and_loss_gradient() -> None:
    x = torch.rand((1, 3, 256, 256), requires_grad=True)
    loss = ms_ssim_loss(x, x)

    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
