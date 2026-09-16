from __future__ import annotations

import zlib

import torch

from core.dna_edit_code import DNA_BYTE_CODEWORDS, encode_dna_bytes
from utils.dna_channel import DNA_BASES


PAD_ID = 4
BASE_TO_ID = {base: index for index, base in enumerate(DNA_BASES)}
BYTE_CODEWORDS = DNA_BYTE_CODEWORDS
INDEX_BYTES = 2
CRC_BYTES = 4
PAYLOAD_BYTES = 16
FULL_CHAIN_BYTES = INDEX_BYTES + PAYLOAD_BYTES + CRC_BYTES


def position_id_to_bytes(position_id: int) -> list[int]:
    if position_id < 0 or position_id >= 2 ** (8 * INDEX_BYTES):
        raise ValueError(f"position_id {position_id} does not fit in {INDEX_BYTES} bytes")
    return list(position_id.to_bytes(INDEX_BYTES, "big"))


def position_id_from_bytes(byte_values: list[int]) -> int:
    if len(byte_values) != INDEX_BYTES:
        raise ValueError(f"expected {INDEX_BYTES} index bytes, got {len(byte_values)}")
    return int.from_bytes(bytes(byte_values), "big")


def crc_bytes_for_chain(position_id: int, payload_bytes: list[int]) -> list[int]:
    crc = zlib.crc32(position_id.to_bytes(INDEX_BYTES, "big") + bytes(payload_bytes)) & 0xFFFFFFFF
    return list(crc.to_bytes(CRC_BYTES, "big"))


def full_chain_byte_labels(position_id: int, payload_bytes: list[int]) -> list[int]:
    return position_id_to_bytes(position_id) + payload_bytes + crc_bytes_for_chain(position_id, payload_bytes)


def encode_chain(
    position_id: int,
    byte_values: list[int],
    position_nt: int = 14,
    byte_codebook: tuple[str, ...] = DNA_BYTE_CODEWORDS,
) -> str:
    del position_nt
    payload = bytes(byte_values)
    crc = zlib.crc32(position_id.to_bytes(INDEX_BYTES, "big") + payload) & 0xFFFFFFFF
    crc_dna = encode_dna_bytes(crc.to_bytes(CRC_BYTES, "big"), byte_codebook)
    index_dna = encode_dna_bytes(position_id.to_bytes(INDEX_BYTES, "big"), byte_codebook)
    return index_dna + "".join(byte_codebook[value] for value in byte_values) + crc_dna


def tokenize_dna(sequence: str, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.full((max_len,), PAD_ID, dtype=torch.long)
    mask = torch.zeros((max_len,), dtype=torch.bool)
    clipped = sequence[:max_len]
    for index, base in enumerate(clipped):
        ids[index] = BASE_TO_ID.get(base, PAD_ID)
        mask[index] = True
    return ids, mask


def int8_vector_to_byte_labels(vector: torch.Tensor) -> torch.Tensor:
    values = vector.to(torch.int16)
    values = torch.where(values < 0, values + 256, values)
    return values.to(torch.long)

