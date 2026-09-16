from __future__ import annotations

import zlib
from typing import Iterable

import torch

from core.dna_edit_code import decode_dna_byte_candidates
from core.dna_edit_code import make_byte_codebook
from core.dlrt_codec import (
    CRC_BYTES,
    INDEX_BYTES,
    PAYLOAD_BYTES,
    encode_chain,
    full_chain_byte_labels,
    int8_vector_to_byte_labels,
    position_id_from_bytes,
)
from utils.dna_channel import mutate_dna


def bytes_to_signed(values: list[int]) -> list[int]:
    return [value - 256 if value >= 128 else value for value in values]


def recover_observed_crc(
    noisy_dna: str,
    *,
    beam: int = 32,
    byte_codebook: tuple[str, ...] | None = None,
) -> tuple[int | None, int | None, int]:
    best: tuple[int, int, int, int] | None = None
    codebook = make_byte_codebook() if byte_codebook is None else byte_codebook
    byte_code_nt = len(codebook[0])
    for suffix_len in range(CRC_BYTES * (byte_code_nt - 1), CRC_BYTES * (byte_code_nt + 1) + 1):
        if suffix_len > len(noisy_dna):
            continue
        suffix = noisy_dna[-suffix_len:]
        for raw, edits in decode_dna_byte_candidates(suffix, 4, beam=beam, codebook=codebook):
            value = int.from_bytes(raw, "big")
            score = edits + abs(suffix_len - CRC_BYTES * byte_code_nt)
            if best is None or (score, edits, suffix_len) < (best[0], best[1], best[2]):
                best = (score, edits, suffix_len, value)
    if best is None:
        return None, None, 0
    return best[3], best[1], best[2]


def topk_candidates(log_probs: torch.Tensor, top_k: int) -> list[list[dict]]:
    values, indices = torch.topk(log_probs, top_k, dim=-1)
    return [
        [
            {"byte": int(index), "logp": float(value)}
            for index, value in zip(channel_indices.tolist(), channel_values.tolist())
        ]
        for channel_indices, channel_values in zip(indices, values)
    ]


def beam_search_full_chain_crc(
    candidates: list[list[dict]],
    *,
    total_positions: int,
    beam_size: int,
) -> tuple[dict | None, list[dict], int]:
    expected_len = INDEX_BYTES + PAYLOAD_BYTES + CRC_BYTES
    if len(candidates) != expected_len:
        raise ValueError(f"expected {expected_len} byte positions, got {len(candidates)}")

    beam = [([], 0.0)]
    for byte_candidates in candidates:
        expanded = []
        for prefix, score in beam:
            for candidate in byte_candidates:
                expanded.append((prefix + [candidate["byte"]], score + candidate["logp"]))
        expanded.sort(key=lambda item: item[1], reverse=True)
        beam = expanded[:beam_size]

    checked = []
    hit = None
    for rank, (chain_bytes, score) in enumerate(beam, start=1):
        index_bytes = chain_bytes[:INDEX_BYTES]
        payload = chain_bytes[INDEX_BYTES : INDEX_BYTES + PAYLOAD_BYTES]
        observed_crc_bytes = chain_bytes[INDEX_BYTES + PAYLOAD_BYTES :]
        position_id = position_id_from_bytes(index_bytes)
        in_range = position_id < total_positions
        expected_crc = zlib.crc32(position_id.to_bytes(INDEX_BYTES, "big") + bytes(payload)) & 0xFFFFFFFF
        observed_crc = int.from_bytes(bytes(observed_crc_bytes), "big")
        passed = in_range and expected_crc == observed_crc
        entry = {
            "rank": rank,
            "score": score,
            "position_id": position_id,
            "crc": f"{expected_crc:08x}",
            "observed_crc": f"{observed_crc:08x}",
            "crc_pass": passed,
            "bytes": payload,
            "chain_bytes": chain_bytes,
            "signed": bytes_to_signed(payload),
        }
        checked.append(entry)
        if passed and hit is None:
            hit = entry
            break
    return hit, checked, len(beam)


def encode_latent_to_dna_records(
    latent: torch.Tensor,
    *,
    image_name: str,
    position_nt: int = 6,
    byte_codebook: tuple[str, ...] | None = None,
) -> Iterable[dict]:
    codebook = make_byte_codebook() if byte_codebook is None else byte_codebook
    channels, height, width = latent.shape
    for flat_index in range(height * width):
        row = flat_index // width
        col = flat_index % width
        byte_values = int8_vector_to_byte_labels(latent[:, row, col]).tolist()
        yield {
            "event": "chain",
            "image": image_name,
            "row": row,
            "col": col,
            "position_id": flat_index,
            "bytes": byte_values,
            "chain_bytes": full_chain_byte_labels(flat_index, byte_values),
            "signed": latent[:, row, col].tolist(),
            "dna": encode_chain(flat_index, byte_values, position_nt, codebook),
        }


def signed_tensor_from_bytes(byte_values: list[int]) -> torch.Tensor:
    return torch.tensor(bytes_to_signed(byte_values), dtype=torch.int8)

