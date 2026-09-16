from __future__ import annotations

import random
import time
import zlib
from pathlib import Path

import torch
import torch.nn.functional as F

from core.dlrt_model import DNAByteNet
from core.dna_edit_code import make_byte_codebook
from core.dna_edit_code import decode_dna_byte_candidates
from .dna_codec import (
    beam_search_full_chain_crc,
    encode_chain,
    full_chain_byte_labels,
    int8_vector_to_byte_labels,
    mutate_dna,
    recover_observed_crc,
    signed_tensor_from_bytes,
    topk_candidates,
)
from core.dlrt_codec import (
    FULL_CHAIN_BYTES,
    INDEX_BYTES,
    PAYLOAD_BYTES,
    position_id_from_bytes,
    position_id_to_bytes,
    tokenize_dna,
)


def ranked_index_candidates(log_probs: torch.Tensor, index_top_k: int, total_positions: int) -> list[int]:
    values, indices = torch.topk(log_probs[:INDEX_BYTES], index_top_k, dim=-1)
    candidates = []
    for first_value, first_index in zip(values[0].tolist(), indices[0].tolist()):
        for second_value, second_index in zip(values[1].tolist(), indices[1].tolist()):
            position_id = position_id_from_bytes([int(first_index), int(second_index)])
            if position_id >= total_positions:
                continue
            candidates.append((position_id, float(first_value) + float(second_value)))
    candidates.sort(key=lambda item: item[1], reverse=True)
    seen = set()
    ranked = []
    for position_id, _ in candidates:
        if position_id in seen:
            continue
        seen.add(position_id)
        ranked.append(position_id)
    return ranked


def _crc_valid_full_chain(chain_bytes: list[int], total_positions: int) -> tuple[bool, int, list[int]]:
    index_bytes = chain_bytes[:INDEX_BYTES]
    payload = chain_bytes[INDEX_BYTES : INDEX_BYTES + PAYLOAD_BYTES]
    observed_crc_bytes = chain_bytes[INDEX_BYTES + PAYLOAD_BYTES :]
    position_id = position_id_from_bytes(index_bytes)
    if position_id >= total_positions:
        return False, position_id, payload
    expected_crc = zlib.crc32(position_id.to_bytes(INDEX_BYTES, "big") + bytes(payload)) & 0xFFFFFFFF
    observed_crc = int.from_bytes(bytes(observed_crc_bytes), "big")
    return expected_crc == observed_crc, position_id, payload


@torch.no_grad()
def decode_latent_from_dna_hard_codebook(
    *,
    model_args: dict,
    latent: torch.Tensor,
    sub_rate: float,
    ins_rate: float,
    del_rate: float,
    seed: int,
    chain_limit: int,
    hard_codebook_beam: int,
) -> dict:
    channels, latent_height, latent_width = latent.shape
    if channels != 16:
        raise ValueError(f"expected 16 latent channels, got {channels}")

    total_chains = latent_height * latent_width
    decode_chains = total_chains if chain_limit <= 0 else min(total_chains, chain_limit)
    rng = random.Random(seed)
    position_nt = int(model_args.get("position_nt", 6))
    byte_codebook = make_byte_codebook(
        length=int(model_args.get("byte_code_nt", 7)),
        mode=str(model_args.get("byte_codebook_mode", "designed")),
        seed=int(model_args.get("byte_codebook_seed", 42)),
    )

    recovered_latent = torch.zeros_like(latent)
    full_chain_truth = torch.full((decode_chains, FULL_CHAIN_BYTES), -1, dtype=torch.long)
    crc_erasure_full_chain = torch.full((decode_chains, FULL_CHAIN_BYTES), -1, dtype=torch.long)
    substitutions = insertions = deletions = total_errors = 0
    observed_crc_readable = 0
    observed_crc_matches_truth = 0
    hard_candidate_chains = 0
    hard_crc_pass = 0
    hard_top1_chain_correct = 0
    hard_top1_byte_correct = 0
    recovered_filled = torch.zeros((latent_height, latent_width), dtype=torch.bool)

    for flat_index in range(decode_chains):
        row = flat_index // latent_width
        col = flat_index % latent_width
        position_id = row * latent_width + col
        truth = int8_vector_to_byte_labels(latent[:, row, col]).tolist()
        chain_truth = full_chain_byte_labels(position_id, truth)
        full_chain_truth[position_id] = torch.tensor(chain_truth, dtype=torch.long)
        clean_dna = encode_chain(position_id, truth, position_nt, byte_codebook)
        expected_crc = zlib.crc32(position_id.to_bytes(INDEX_BYTES, "big") + bytes(truth)) & 0xFFFFFFFF
        noisy_dna, errors = mutate_dna(
            clean_dna,
            sub_rate=sub_rate,
            ins_rate=ins_rate,
            del_rate=del_rate,
            seed=rng.randrange(2**31),
        )
        observed_crc, _, _ = recover_observed_crc(noisy_dna, byte_codebook=byte_codebook)
        observed_crc_readable += int(observed_crc is not None)
        observed_crc_matches_truth += int(observed_crc == expected_crc)
        substitutions += errors["substitutions"]
        insertions += errors["insertions"]
        deletions += errors["deletions"]
        total_errors += errors["total"]

        candidates = decode_dna_byte_candidates(
            noisy_dna,
            FULL_CHAIN_BYTES,
            beam=hard_codebook_beam,
            codebook=byte_codebook,
        )
        hard_candidate_chains += len(candidates)
        if candidates:
            top1_bytes = list(candidates[0][0])
            hard_top1_chain_correct += int(top1_bytes == chain_truth)
            hard_top1_byte_correct += sum(int(a == b) for a, b in zip(top1_bytes, chain_truth))

        hit_chain = None
        hit_position_id = -1
        hit_payload: list[int] = []
        for raw, _edits in candidates:
            chain_bytes = list(raw)
            crc_ok, candidate_position_id, payload = _crc_valid_full_chain(chain_bytes, total_chains)
            if crc_ok:
                hit_chain = chain_bytes
                hit_position_id = candidate_position_id
                hit_payload = payload
                break

        if hit_chain is None or hit_position_id >= total_chains:
            continue
        hit_row = hit_position_id // latent_width
        hit_col = hit_position_id % latent_width
        if bool(recovered_filled[hit_row, hit_col]):
            continue
        hard_crc_pass += 1
        recovered_latent[:, hit_row, hit_col] = signed_tensor_from_bytes(hit_payload)
        recovered_filled[hit_row, hit_col] = True
        if hit_position_id < decode_chains:
            crc_erasure_full_chain[hit_position_id] = torch.tensor(hit_chain, dtype=torch.long)

    return {
        "recovered_latent": recovered_latent,
        "crc_top1_fallback_latent": recovered_latent.clone(),
        "full_chain_truth": full_chain_truth,
        "crc_erasure_full_chain": crc_erasure_full_chain,
        "crc_mllv_full_chain": crc_erasure_full_chain.clone(),
        "summary": {
            "decoder": "hard-codebook",
            "total_chains": total_chains,
            "decoded_chains": decode_chains,
            "not_decoded": total_chains - decode_chains,
            "beam_crc_pass": hard_crc_pass,
            "beam_erasures": decode_chains - hard_crc_pass,
            "mllv_filled": 0,
            "mllv_unfilled": decode_chains - hard_crc_pass,
            "mllv_collisions": 0,
            "index_top_k": 0,
            "index_top1_accuracy": 0.0,
            "index_topk_accuracy": 0.0,
            "top1_crc_pass": 0,
            "observed_crc_readable": observed_crc_readable,
            "observed_crc_matches_truth": observed_crc_matches_truth,
            "channel_top1_accuracy": hard_top1_byte_correct / max(decode_chains * FULL_CHAIN_BYTES, 1),
            "hard_top1_chain_accuracy": hard_top1_chain_correct / max(decode_chains, 1),
            "hard_candidate_chains": hard_candidate_chains,
            "average_errors_per_chain": total_errors / max(decode_chains, 1),
            "substitutions": substitutions,
            "insertions": insertions,
            "deletions": deletions,
            "base_errors": total_errors,
        },
    }


@torch.no_grad()
def decode_latent_from_dna(
    *,
    model,
    model_args: dict,
    device: torch.device,
    latent: torch.Tensor,
    sub_rate: float,
    ins_rate: float,
    del_rate: float,
    top_k: int,
    beam_size: int,
    infer_batch_size: int,
    seed: int,
    chain_limit: int,
    index_top_k: int | None = None,
    collect_mllv_diagnostics: bool = False,
) -> dict:
    channels, latent_height, latent_width = latent.shape
    if channels != 16:
        raise ValueError(f"expected 16 latent channels, got {channels}")

    total_chains = latent_height * latent_width
    effective_index_top_k = top_k if index_top_k is None else index_top_k
    decode_chains = total_chains if chain_limit <= 0 else min(total_chains, chain_limit)
    rng = random.Random(seed)
    records = []
    substitutions = insertions = deletions = total_errors = 0
    observed_crc_readable = 0
    observed_crc_matches_truth = 0
    position_nt = int(model_args.get("position_nt", 6))
    max_len = int(model_args.get("max_len", 192))
    byte_codebook = make_byte_codebook(
        length=int(model_args.get("byte_code_nt", 7)),
        mode=str(model_args.get("byte_codebook_mode", "designed")),
        seed=int(model_args.get("byte_codebook_seed", 42)),
    )

    for flat_index in range(decode_chains):
        row = flat_index // latent_width
        col = flat_index % latent_width
        position_id = row * latent_width + col
        truth = int8_vector_to_byte_labels(latent[:, row, col]).tolist()
        clean_dna = encode_chain(position_id, truth, position_nt, byte_codebook)
        expected_crc = zlib.crc32(position_id.to_bytes(INDEX_BYTES, "big") + bytes(truth)) & 0xFFFFFFFF
        noisy_dna, errors = mutate_dna(
            clean_dna,
            sub_rate=sub_rate,
            ins_rate=ins_rate,
            del_rate=del_rate,
            seed=rng.randrange(2**31),
        )
        observed_crc, observed_crc_edits, observed_crc_nt = recover_observed_crc(
            noisy_dna,
            byte_codebook=byte_codebook,
        )
        observed_crc_readable += int(observed_crc is not None)
        observed_crc_matches_truth += int(observed_crc == expected_crc)
        substitutions += errors["substitutions"]
        insertions += errors["insertions"]
        deletions += errors["deletions"]
        total_errors += errors["total"]
        input_ids, input_mask = tokenize_dna(noisy_dna, max_len)
        records.append(
            {
                "row": row,
                "col": col,
                "position_id": position_id,
                "truth": truth,
                "chain_truth": full_chain_byte_labels(position_id, truth),
                "expected_crc": expected_crc,
                "observed_crc": observed_crc,
                "observed_crc_edits": observed_crc_edits,
                "observed_crc_nt": observed_crc_nt,
                "input_ids": input_ids,
                "input_mask": input_mask,
            }
        )

    recovered_latent = torch.zeros_like(latent)
    crc_top1_fallback_latent = torch.zeros_like(latent)
    full_chain_truth = torch.full((decode_chains, FULL_CHAIN_BYTES), -1, dtype=torch.long)
    crc_erasure_full_chain = torch.full((decode_chains, FULL_CHAIN_BYTES), -1, dtype=torch.long)
    crc_mllv_full_chain = torch.full((decode_chains, FULL_CHAIN_BYTES), -1, dtype=torch.long)
    for record in records:
        full_chain_truth[record["position_id"]] = torch.tensor(record["chain_truth"], dtype=torch.long)
    beam_crc_pass = 0
    top1_crc_pass = 0
    channel_top1_correct = 0
    index_top1_correct = 0
    index_topk_contains_truth = 0
    recovered_filled = torch.zeros((latent_height, latent_width), dtype=torch.bool)
    top1_filled = torch.zeros((latent_height, latent_width), dtype=torch.bool)
    pending_mllv = []
    mllv_diagnostics = []

    infer_batch_size = max(1, infer_batch_size)
    for start in range(0, len(records), infer_batch_size):
        batch_records = records[start : start + infer_batch_size]
        input_ids = torch.stack([record["input_ids"] for record in batch_records]).to(device)
        input_mask = torch.stack([record["input_mask"] for record in batch_records]).to(device)
        logits_batch = model(input_ids, input_mask).detach().cpu()

        for record, logits in zip(batch_records, logits_batch):
            row = record["row"]
            col = record["col"]
            position_id = record["position_id"]
            truth = record["truth"]
            observed_crc = record["observed_crc"]
            log_probs = F.log_softmax(logits, dim=-1)
            top1_chain = logits.argmax(dim=-1).tolist()
            top1 = top1_chain[INDEX_BYTES : INDEX_BYTES + PAYLOAD_BYTES]
            top1_position_id = position_id_from_bytes(top1_chain[:INDEX_BYTES])
            index_candidates = ranked_index_candidates(log_probs, effective_index_top_k, total_chains)
            index_top1_correct += int(top1_position_id == position_id)
            index_topk_contains_truth += int(position_id in index_candidates)
            top1_crc_ok = (
                top1_position_id < total_chains
                and zlib.crc32(top1_position_id.to_bytes(INDEX_BYTES, "big") + bytes(top1)) & 0xFFFFFFFF
                == int.from_bytes(bytes(top1_chain[INDEX_BYTES + PAYLOAD_BYTES :]), "big")
            )
            top1_crc_pass += int(top1_crc_ok)
            channel_top1_correct += sum(int(a == b) for a, b in zip(top1, truth))

            if top1_crc_ok:
                hit = {"rank": 0, "bytes": top1, "position_id": top1_position_id, "chain_bytes": top1_chain}
                checked = [hit]
            else:
                candidates = topk_candidates(log_probs, top_k)
                hit, checked, _ = beam_search_full_chain_crc(
                    candidates,
                    total_positions=total_chains,
                    beam_size=beam_size,
                )

            if hit is not None:
                hit_position_id = int(hit.get("position_id", top1_position_id))
                if hit_position_id >= total_chains:
                    continue
                hit_row = hit_position_id // latent_width
                hit_col = hit_position_id % latent_width
                if bool(recovered_filled[hit_row, hit_col]):
                    pending_mllv.append(
                        {
                            "log_probs": log_probs,
                            "top1": top1,
                            "top1_chain": top1_chain,
                            "source_position_id": position_id,
                            "checked": checked if collect_mllv_diagnostics else None,
                        }
                    )
                    continue
                beam_crc_pass += 1
                recovered_latent[:, hit_row, hit_col] = signed_tensor_from_bytes(hit["bytes"])
                crc_top1_fallback_latent[:, hit_row, hit_col] = recovered_latent[:, hit_row, hit_col]
                if hit_position_id < decode_chains:
                    hit_chain = torch.tensor(hit["chain_bytes"], dtype=torch.long)
                    crc_erasure_full_chain[hit_position_id] = hit_chain
                    crc_mllv_full_chain[hit_position_id] = hit_chain
                recovered_filled[hit_row, hit_col] = True
                top1_filled[hit_row, hit_col] = True
            else:
                pending_mllv.append(
                    {
                        "log_probs": log_probs,
                        "top1": top1,
                        "top1_chain": top1_chain,
                        "source_position_id": position_id,
                        "checked": checked if collect_mllv_diagnostics else None,
                    }
                )

    mllv_filled = 0
    mllv_collisions = 0
    for pending in pending_mllv:
        log_probs = pending["log_probs"]
        top1 = pending["top1"]
        top1_chain = pending["top1_chain"]
        placed = False
        assigned_position_id = None
        for position_id in ranked_index_candidates(log_probs, effective_index_top_k, total_chains):
            row = position_id // latent_width
            col = position_id % latent_width
            if bool(recovered_filled[row, col]) or bool(top1_filled[row, col]):
                continue
            crc_top1_fallback_latent[:, row, col] = signed_tensor_from_bytes(top1)
            if position_id < decode_chains:
                completed_chain = list(top1_chain)
                completed_chain[:INDEX_BYTES] = position_id_to_bytes(position_id)
                crc_mllv_full_chain[position_id] = torch.tensor(completed_chain, dtype=torch.long)
            top1_filled[row, col] = True
            mllv_filled += 1
            placed = True
            assigned_position_id = position_id
            break
        mllv_collisions += int(not placed)
        if collect_mllv_diagnostics:
            probabilities = log_probs.exp()
            values, indices = torch.topk(probabilities, min(top_k, probabilities.shape[-1]), dim=-1)
            probability_rows = [
                [
                    {"byte": int(byte), "probability": float(probability)}
                    for byte, probability in zip(row_indices.tolist(), row_values.tolist())
                ]
                for row_indices, row_values in zip(indices, values)
            ]
            mllv_diagnostics.append(
                {
                    "source_position_id": int(pending["source_position_id"]),
                    "assigned_position_id": assigned_position_id,
                    "probability_rows": probability_rows,
                    "tcbs_checked": pending["checked"],
                }
            )

    return {
        "recovered_latent": recovered_latent,
        "crc_top1_fallback_latent": crc_top1_fallback_latent,
        "full_chain_truth": full_chain_truth,
        "crc_erasure_full_chain": crc_erasure_full_chain,
        "crc_mllv_full_chain": crc_mllv_full_chain,
        "mllv_diagnostics": mllv_diagnostics,
        "summary": {
            "total_chains": total_chains,
            "decoded_chains": decode_chains,
            "not_decoded": total_chains - decode_chains,
            "beam_crc_pass": beam_crc_pass,
            "beam_erasures": decode_chains - beam_crc_pass,
            "mllv_filled": mllv_filled,
            "mllv_unfilled": len(pending_mllv) - mllv_filled,
            "mllv_collisions": mllv_collisions,
            "index_top_k": effective_index_top_k,
            "index_top1_accuracy": index_top1_correct / max(decode_chains, 1),
            "index_topk_accuracy": index_topk_contains_truth / max(decode_chains, 1),
            "top1_crc_pass": top1_crc_pass,
            "observed_crc_readable": observed_crc_readable,
            "observed_crc_matches_truth": observed_crc_matches_truth,
            "channel_top1_accuracy": channel_top1_correct / max(decode_chains * PAYLOAD_BYTES, 1),
            "average_errors_per_chain": total_errors / max(decode_chains, 1),
            "substitutions": substitutions,
            "insertions": insertions,
            "deletions": deletions,
            "base_errors": total_errors,
        },
    }


@torch.no_grad()
def correct_corrupted_records_to_latents(
    *,
    records: list[dict],
    model,
    model_args: dict,
    device: torch.device,
    latent_shape: tuple[int, int, int],
    top_k: int,
    beam_size: int,
    infer_batch_size: int,
    index_top_k: int | None = None,
) -> dict:
    started = time.time()
    channels, height, width = latent_shape
    effective_index_top_k = top_k if index_top_k is None else index_top_k
    max_len = int(model_args.get("max_len", 192))
    byte_codebook = make_byte_codebook(
        length=int(model_args.get("byte_code_nt", 7)),
        mode=str(model_args.get("byte_codebook_mode", "designed")),
        seed=int(model_args.get("byte_codebook_seed", 42)),
    )
    recovered_latent = torch.zeros((channels, height, width), dtype=torch.int8)
    top1_completion_latent = torch.zeros((channels, height, width), dtype=torch.int8)
    truth_latent = torch.zeros((channels, height, width), dtype=torch.int8)

    prepared = []
    for record in records:
        input_ids, input_mask = tokenize_dna(record["noisy_dna"], max_len)
        prepared.append({**record, "input_ids": input_ids, "input_mask": input_mask})

    beam_crc_pass = 0
    top1_chain_correct = 0
    top1_byte_correct = 0
    observed_crc_readable = 0
    observed_crc_matches_truth = 0
    decoded_chains = len(prepared)
    recovered_filled = torch.zeros((height, width), dtype=torch.bool)
    top1_filled = torch.zeros((height, width), dtype=torch.bool)
    pending_mllv = []
    recovered_collisions = 0

    for start in range(0, decoded_chains, max(1, infer_batch_size)):
        batch = prepared[start : start + max(1, infer_batch_size)]
        input_ids = torch.stack([record["input_ids"] for record in batch]).to(device)
        input_mask = torch.stack([record["input_mask"] for record in batch]).to(device)
        logits_batch = model(input_ids, input_mask).detach().cpu()
        for record, logits in zip(batch, logits_batch):
            row = int(record["row"])
            col = int(record["col"])
            position_id = int(record["position_id"])
            truth = [int(value) for value in record.get("bytes", [])]
            if truth:
                truth_latent[:, row, col] = signed_tensor_from_bytes(truth)

            top1_chain = logits.argmax(dim=-1).tolist()
            top1 = top1_chain[INDEX_BYTES : INDEX_BYTES + PAYLOAD_BYTES]
            if truth:
                top1_chain_correct += int(top1 == truth)
                top1_byte_correct += sum(int(a == b) for a, b in zip(top1, truth))

            observed_crc, _, _ = recover_observed_crc(
                record["noisy_dna"],
                byte_codebook=byte_codebook,
            )
            expected_crc = None
            if truth:
                expected_crc = zlib.crc32(position_id.to_bytes(INDEX_BYTES, "big") + bytes(truth)) & 0xFFFFFFFF

            observed_crc_readable += int(observed_crc is not None)
            observed_crc_matches_truth += int(expected_crc is not None and observed_crc == expected_crc)
            log_probs = F.log_softmax(logits, dim=-1)
            candidates = topk_candidates(log_probs, top_k)
            hit, _, _ = beam_search_full_chain_crc(
                candidates,
                total_positions=height * width,
                beam_size=beam_size,
            )
            if hit is not None:
                hit_position_id = int(hit["position_id"])
                hit_row = hit_position_id // width
                hit_col = hit_position_id % width
                if bool(recovered_filled[hit_row, hit_col]):
                    recovered_collisions += 1
                    pending_mllv.append((log_probs, top1))
                    continue
                beam_crc_pass += 1
                recovered_latent[:, hit_row, hit_col] = signed_tensor_from_bytes(hit["bytes"])
                top1_completion_latent[:, hit_row, hit_col] = recovered_latent[:, hit_row, hit_col]
                recovered_filled[hit_row, hit_col] = True
                top1_filled[hit_row, hit_col] = True
            else:
                pending_mllv.append((log_probs, top1))

    mllv_filled = 0
    mllv_unfilled = 0
    mllv_collisions = 0
    for log_probs, top1 in pending_mllv:
        placed = False
        for candidate_position_id in ranked_index_candidates(log_probs, effective_index_top_k, height * width):
            row = candidate_position_id // width
            col = candidate_position_id % width
            if bool(recovered_filled[row, col]) or bool(top1_filled[row, col]):
                continue
            top1_completion_latent[:, row, col] = signed_tensor_from_bytes(top1)
            top1_filled[row, col] = True
            mllv_filled += 1
            placed = True
            break
        if not placed:
            mllv_unfilled += 1
            mllv_collisions += 1

    metrics = {
        "decoded_chains": decoded_chains,
        "beam_crc_pass": beam_crc_pass,
        "crc_validated_chain_rate": beam_crc_pass / max(decoded_chains, 1),
        "observed_crc_readable": observed_crc_readable,
        "observed_crc_matches_truth": observed_crc_matches_truth,
        "top1_chain_accuracy": top1_chain_correct / max(decoded_chains, 1),
        "top1_byte_accuracy": top1_byte_correct / max(decoded_chains * PAYLOAD_BYTES, 1),
        "top_k": top_k,
        "index_top_k": effective_index_top_k,
        "beam_size": beam_size,
        "mllv_filled": mllv_filled,
        "mllv_unfilled": mllv_unfilled,
        "mllv_collisions": mllv_collisions,
        "recovered_collisions": recovered_collisions,
        "elapsed_seconds": time.time() - started,
    }
    return {
        "recovered_latent": recovered_latent,
        "top1_completion_latent": top1_completion_latent,
        "truth_latent": truth_latent,
        "metrics": metrics,
    }


def load_dlrt_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    model = DNAByteNet(
        latent_channels=int(args.get("latent_channels", 16)),
        target_bytes=int(args.get("target_bytes", args.get("latent_channels", 16))),
        max_len=int(args.get("max_len", 192)),
        d_model=int(args.get("d_model", 192)),
        layers=int(args.get("layers", 4)),
        heads=int(args.get("heads", 6)),
        dropout=float(args.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, args

