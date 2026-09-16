import math

import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim as standard_ms_ssim


def mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, y)


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    value = mse(x, y).item()
    if value <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / value)


def _gaussian_window(size: int, sigma: float, channels: int, device) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.expand(channels, 1, size, size).contiguous()


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    channels = x.shape[1]
    window = _gaussian_window(window_size, sigma, channels, x.device)
    padding = window_size // 2
    mu_x = F.conv2d(x, window, padding=padding, groups=channels)
    mu_y = F.conv2d(y, window, padding=padding, groups=channels)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, window, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy
    c1 = 0.01**2
    c2 = 0.03**2
    value = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return value.mean()


def ms_ssim(x: torch.Tensor, y: torch.Tensor) -> float:
    return ms_ssim_tensor(x, y).item()


def ms_ssim_tensor(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return standard_ms_ssim(
        x,
        y,
        data_range=1.0,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
    )


def ms_ssim_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 1.0 - ms_ssim_tensor(x, y)
