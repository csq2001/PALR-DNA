import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, latent_channels: int = 16, base_channels: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(base_channels),
            nn.Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(base_channels * 2),
            nn.Conv2d(base_channels * 2, latent_channels, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(latent_channels),
        )

    def forward(self, x):
        return self.net(x)
