from __future__ import annotations

import random
from dataclasses import dataclass


DNA_BASES = "ACGT"


@dataclass(frozen=True)
class DNAChannelConfig:
    substitution_rate: float = 0.0
    insertion_rate: float = 0.0
    deletion_rate: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("substitution_rate", self.substitution_rate),
            ("insertion_rate", self.insertion_rate),
            ("deletion_rate", self.deletion_rate),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class DNAChannelResult:
    sequence: str
    substitutions: int
    insertions: int
    deletions: int

    @property
    def total_errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions


def simulate_dna_channel(
    sequence: str,
    config: DNAChannelConfig,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> DNAChannelResult:
    if any(base not in DNA_BASES for base in sequence):
        raise ValueError("sequence must contain only A/C/G/T")
    rng = rng or random.Random(seed)
    output: list[str] = []
    substitutions = insertions = deletions = 0
    for base in sequence:
        if rng.random() < config.insertion_rate:
            output.append(rng.choice(DNA_BASES))
            insertions += 1
        if rng.random() < config.deletion_rate:
            deletions += 1
            continue
        if rng.random() < config.substitution_rate:
            choices = [candidate for candidate in DNA_BASES if candidate != base]
            output.append(rng.choice(choices))
            substitutions += 1
        else:
            output.append(base)
    return DNAChannelResult(
        sequence="".join(output),
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
    )


def mutate_dna(
    dna: str,
    *,
    sub_rate: float,
    ins_rate: float,
    del_rate: float,
    seed: int,
) -> tuple[str, dict]:
    result = simulate_dna_channel(
        dna,
        DNAChannelConfig(
            substitution_rate=sub_rate,
            insertion_rate=ins_rate,
            deletion_rate=del_rate,
        ),
        rng=random.Random(seed),
    )
    return result.sequence, {
        "substitutions": result.substitutions,
        "insertions": result.insertions,
        "deletions": result.deletions,
        "total": result.total_errors,
    }

