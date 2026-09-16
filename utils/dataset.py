from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


class PortraitImageDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        patch_size: Optional[int] = 256,
        training: bool = True,
        channels: int = 1,
    ) -> None:
        self.root = Path(root)
        self.paths = sorted([p for p in self.root.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
        if not self.paths:
            raise FileNotFoundError(f"No images found in {self.root}")
        self.patch_size = patch_size
        self.training = training
        self.mode = "L" if channels == 1 else "RGB"

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = Image.open(path).convert(self.mode)
        tensor = self._to_tensor(image)
        tensor = self._crop(tensor)
        return tensor, path.name

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        channels = len(image.getbands())
        data = data.view(image.height, image.width, channels).permute(2, 0, 1).float() / 255.0
        return data

    def _crop(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.patch_size is None:
            return tensor
        _, height, width = tensor.shape
        size = min(self.patch_size, height, width)
        if self.training:
            top = torch.randint(0, height - size + 1, ()).item()
            left = torch.randint(0, width - size + 1, ()).item()
        else:
            top = max((height - size) // 2, 0)
            left = max((width - size) // 2, 0)
        return tensor[:, top : top + size, left : left + size]
