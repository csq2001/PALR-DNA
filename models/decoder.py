import torch.nn as nn

from .encoder import ResidualBlock


class ImageDecoder(nn.Module):
    def __init__(self, out_channels: int = 1, latent_channels: int = 16, base_channels: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(latent_channels),
            nn.ConvTranspose2d(latent_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(base_channels * 2),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(base_channels),
            nn.ConvTranspose2d(base_channels, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, y_hat):
        return self.net(y_hat)
