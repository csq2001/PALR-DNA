from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import random
import math


def _load_default_codebook() -> tuple[str, ...]:
    path = Path(__file__).resolve().parents[1] / "outputs" / "codebook" / "7nt_byte_codebook.json"
    if not path.exists():
        raise FileNotFoundError(f"required 7-nt byte codebook not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    codebook = tuple(payload["codebook"])
    if len(codebook) != 256 or any(len(word) != 7 for word in codebook):
        raise ValueError(f"invalid DNA byte codebook: {path}")
    return codebook


DNA_BYTE_CODEWORDS = _load_default_codebook()
BASES = "ACGT"


def _base4_word(value: int, length: int) -> str:
    if value < 0 or value >= 4**length:
        raise ValueError(f"value {value} does not fit in {length} nt")
    digits = []
    for shift in range(length - 1, -1, -1):
        digits.append(BASES[(value >> (2 * shift)) & 3])
    return "".join(digits)


def _has_long_homopolymer(word: str, max_run: int = 3) -> bool:
    run = 1
    for left, right in zip(word, word[1:]):
        run = run + 1 if left == right else 1
        if run > max_run:
            return True
    return False


def _max_homopolymer_run(word: str) -> int:
    longest = 1
    run = 1
    for left, right in zip(word, word[1:]):
        run = run + 1 if left == right else 1
        longest = max(longest, run)
    return longest


def _gc_count(word: str) -> int:
    return word.count("G") + word.count("C")


def _gc_bounds(length: int, low: float = 0.4, high: float = 0.6) -> tuple[int, int]:
    return math.ceil(length * low), math.floor(length * high)


def _passes_synthesis_constraints(word: str) -> bool:
    low, high = _gc_bounds(len(word))
    return low <= _gc_count(word) <= high and not _has_long_homopolymer(word)


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + int(left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _all_words(length: int) -> list[str]:
    return [_base4_word(value, length) for value in range(4**length)]


def _candidate_words(length: int, rng: random.Random) -> list[str]:
    if length <= 7:
        return [word for word in _all_words(length) if _passes_synthesis_constraints(word)]
    target = 8192
    candidates: set[str] = set()
    while len(candidates) < target:
        word = "".join(rng.choice(BASES) for _ in range(length))
        if _passes_synthesis_constraints(word):
            candidates.add(word)
    return sorted(candidates)


def _hamming_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right))


def _distance_aware_codebook(length: int, rng: random.Random) -> tuple[str, ...]:
    candidates = _candidate_words(length, rng)
    if len(candidates) < 256:
        raise ValueError(
            f"only {len(candidates)} constrained {length}-nt candidates are available"
        )
    rng.shuffle(candidates)
    selected = [
        min(
            candidates,
            key=lambda word: (
                abs(_gc_count(word) - length / 2),
                _max_homopolymer_run(word),
                word,
            ),
        )
    ]
    remaining = [word for word in candidates if word != selected[0]]
    min_distances = [_hamming_distance(word, selected[0]) for word in remaining]

    while len(selected) < 256:
        sample_size = min(len(remaining), 1024)
        sampled_indices = rng.sample(range(len(remaining)), sample_size)
        best_index = max(
            sampled_indices,
            key=lambda index: (
                min_distances[index],
                -abs(_gc_count(remaining[index]) - length / 2),
                -_max_homopolymer_run(remaining[index]),
                remaining[index],
            ),
        )
        word = remaining.pop(best_index)
        min_distances.pop(best_index)
        selected.append(word)
        for index, candidate in enumerate(remaining):
            min_distances[index] = min(min_distances[index], _hamming_distance(candidate, word))
    return tuple(selected)


@lru_cache(maxsize=64)
def make_byte_codebook(
    *,
    length: int = 7,
    mode: str = "designed",
    seed: int = 42,
) -> tuple[str, ...]:
    """Build a deterministic 256-entry byte-to-DNA codebook.

    The canonical 7-nt codebook is used for mode="designed", length=7.
    Other designed lengths use deterministic parity/check bases so
    length ablations can be run without changing the byte labels.
    """
    if length < 4:
        raise ValueError("byte code nt length must be at least 4")
    if 4**length < 256:
        raise ValueError(f"{length}-nt codebook has fewer than 256 possible words")

    normalized_mode = mode.lower().replace("_", "-")
    if normalized_mode == "default":
        normalized_mode = "designed"
    if normalized_mode in {"legacy", "method", "boundary-safe"}:
        if length != 7:
            raise ValueError("legacy/method boundary-safe codebook is only defined for 7 nt")
        return tuple(DNA_BYTE_CODEWORDS)
    if normalized_mode == "designed" and length == 7:
        return tuple(DNA_BYTE_CODEWORDS)
    if normalized_mode in {"designed", "constrained-designed"}:
        return _distance_aware_codebook(length, rng=random.Random(seed + length * 1009))

    rng = random.Random(seed + length * 1009)
    words = _candidate_words(length, rng)

    if normalized_mode == "base4":
        return tuple(words[:256])

    if normalized_mode == "random":
        return tuple(rng.sample(words, 256))

    raise ValueError(
        f"unknown codebook mode {mode!r}; expected designed, constrained-designed, "
        "legacy, random, or base4"
    )


def codebook_metadata(
    *,
    length: int = 7,
    mode: str = "designed",
    seed: int = 42,
) -> dict:
    codebook = make_byte_codebook(length=length, mode=mode, seed=seed)
    min_distance = min(
        _levenshtein_distance(left, right)
        for index, left in enumerate(codebook)
        for right in codebook[index + 1 :]
    )
    min_hamming = min(
        _hamming_distance(left, right)
        for index, left in enumerate(codebook)
        for right in codebook[index + 1 :]
    )
    gc_values = [
        (word.count("G") + word.count("C")) / len(word)
        for word in codebook
    ]
    return {
        "byte_code_nt": length,
        "byte_codebook_mode": mode,
        "byte_codebook_seed": seed,
        "codebook_size": len(codebook),
        "min_levenshtein_distance": min_distance,
        "min_hamming_distance": min_hamming,
        "mean_gc": sum(gc_values) / len(gc_values),
        "min_gc": min(gc_values),
        "max_gc": max(gc_values),
        "max_homopolymer": max(_max_homopolymer_run(word) for word in codebook),
    }


@dataclass
class _DecodeState:
    position: int
    edits: int
    raw: bytes


@lru_cache(maxsize=64)
def _build_edit_lookup(codebook: tuple[str, ...] = DNA_BYTE_CODEWORDS) -> dict[str, tuple[int, int] | None]:
    lookup: dict[str, tuple[int, int] | None] = {}

    def add(sequence: str, byte: int, edits: int) -> None:
        value = (byte, edits)
        previous = lookup.get(sequence)
        if previous is None and sequence in lookup:
            return
        if previous is not None and previous[0] != byte:
            lookup[sequence] = None
        else:
            lookup[sequence] = value

    for byte, word in enumerate(codebook):
        add(word, byte, 0)
        for index in range(len(word)):
            add(word[:index] + word[index + 1 :], byte, 1)
            for base in BASES:
                if base != word[index]:
                    add(word[:index] + base + word[index + 1 :], byte, 1)
        for index in range(len(word) + 1):
            for base in BASES:
                add(word[:index] + base + word[index:], byte, 1)
    return lookup


EDIT_LOOKUP = _build_edit_lookup()


def encode_dna_bytes(raw: bytes, codebook: tuple[str, ...] = DNA_BYTE_CODEWORDS) -> str:
    return "".join(codebook[value] for value in raw)


def _advance_states(
    sequence: str,
    states: list[_DecodeState],
    expected_bytes: int,
    beam: int,
    edit_lookup: dict[str, tuple[int, int] | None] = EDIT_LOOKUP,
    codeword_length: int = 7,
) -> list[_DecodeState]:
    current = states
    for _ in range(expected_bytes):
        candidates: dict[int, _DecodeState] = {}
        for state in current:
            for length in (codeword_length - 1, codeword_length, codeword_length + 1):
                end = state.position + length
                if end > len(sequence):
                    continue
                decoded = edit_lookup.get(sequence[state.position:end])
                if decoded is None:
                    continue
                value, edits = decoded
                candidate = _DecodeState(
                    position=end,
                    edits=state.edits + edits,
                    raw=state.raw + bytes((value,)),
                )
                previous = candidates.get(end)
                if previous is None or candidate.edits < previous.edits:
                    candidates[end] = candidate
        if not candidates:
            return []
        current = sorted(
            candidates.values(),
            key=lambda item: (item.edits, abs(item.position - len(item.raw) * codeword_length)),
        )[:beam]
    return current


def decode_dna_byte_candidates(
    sequence: str,
    expected_bytes: int,
    *,
    beam: int = 96,
    codebook: tuple[str, ...] = DNA_BYTE_CODEWORDS,
) -> list[tuple[bytes, int]]:
    edit_lookup = EDIT_LOOKUP if codebook == DNA_BYTE_CODEWORDS else _build_edit_lookup(codebook)
    states = _advance_states(
        sequence,
        [_DecodeState(position=0, edits=0, raw=b"")],
        expected_bytes,
        beam,
        edit_lookup,
        len(codebook[0]),
    )
    complete = [state for state in states if state.position == len(sequence)]
    complete.sort(key=lambda state: state.edits)
    return [(state.raw, state.edits) for state in complete]


