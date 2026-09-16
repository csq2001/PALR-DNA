from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from core.dlrt_codec import encode_chain, full_chain_byte_labels, int8_vector_to_byte_labels, tokenize_dna
from core.dna_edit_code import codebook_metadata, make_byte_codebook
from utils.dna_channel import DNAChannelConfig, simulate_dna_channel


class LatentDNAByteDataset(Dataset):
    def __init__(
        self,
        path: Path,
        *,
        samples: int,
        max_len: int,
        position_nt: int,
        latent_channels: int,
        sub_rate: tuple[float, float],
        ins_rate: tuple[float, float],
        del_rate: tuple[float, float],
        seed: int,
        byte_code_nt: int = 7,
        byte_codebook_mode: str = "designed",
        byte_codebook_seed: int = 42,
    ) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        latents = payload["latents"]
        if latents.dtype != torch.int8:
            raise ValueError(f"{path} latents must be int8, got {latents.dtype}")
        if latents.ndim != 4 or latents.shape[1] != latent_channels:
            raise ValueError(
                f"{path} expected [N,{latent_channels},H,W], got {list(latents.shape)}"
            )
        self.path = path
        self.latents = latents.contiguous()
        self.samples = samples
        self.max_len = max_len
        self.position_nt = position_nt
        self.latent_channels = latent_channels
        self.sub_rate = sub_rate
        self.ins_rate = ins_rate
        self.del_rate = del_rate
        self.seed = seed
        self.byte_code_nt = byte_code_nt
        self.byte_codebook_mode = byte_codebook_mode
        self.byte_codebook_seed = byte_codebook_seed
        self.byte_codebook = make_byte_codebook(
            length=byte_code_nt,
            mode=byte_codebook_mode,
            seed=byte_codebook_seed,
        )
        self.codebook_metadata = codebook_metadata(
            length=byte_code_nt,
            mode=byte_codebook_mode,
            seed=byte_codebook_seed,
        )
        self.epoch = 0
        self.num_images, _, self.height, self.width = self.latents.shape
        if self.height * self.width > 2**16:
            raise ValueError(
                f"latent grid {self.height}x{self.width} exceeds 2-byte index capacity"
            )

    def __len__(self):
        return self.samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _sample_rate(self, rng: random.Random, rate_range: tuple[float, float]) -> float:
        low, high = rate_range
        return low if low == high else rng.uniform(low, high)

    def __getitem__(self, index):
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        rng = random.Random(
            self.seed
            + self.epoch * 1_000_003
            + index * 1009
            + worker_id * 9176
        )
        image_index = rng.randrange(self.num_images)
        row = rng.randrange(self.height)
        col = rng.randrange(self.width)
        position_id = row * self.width + col
        payload_labels = int8_vector_to_byte_labels(self.latents[image_index, :, row, col]).tolist()
        labels = torch.tensor(full_chain_byte_labels(position_id, payload_labels), dtype=torch.long)
        clean_dna = encode_chain(
            position_id,
            payload_labels,
            self.position_nt,
            self.byte_codebook,
        )
        channel = DNAChannelConfig(
            substitution_rate=self._sample_rate(rng, self.sub_rate),
            insertion_rate=self._sample_rate(rng, self.ins_rate),
            deletion_rate=self._sample_rate(rng, self.del_rate),
        )
        noisy = simulate_dna_channel(clean_dna, channel, rng=rng).sequence
        input_ids, valid_mask = tokenize_dna(noisy, self.max_len)
        return input_ids, valid_mask, labels


def make_loader(path: Path, args, samples: int, seed: int, shuffle: bool) -> DataLoader:
    dataset = LatentDNAByteDataset(
        path,
        samples=samples,
        max_len=args.max_len,
        position_nt=args.position_nt,
        latent_channels=args.latent_channels,
        sub_rate=args.sub_rate,
        ins_rate=args.ins_rate,
        del_rate=args.del_rate,
        seed=seed,
        byte_code_nt=args.byte_code_nt,
        byte_codebook_mode=args.byte_codebook_mode,
        byte_codebook_seed=args.byte_codebook_seed,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

