from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def add_project_root() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def add_default_arg(argv: list[str], name: str, value: str) -> None:
    if name not in argv:
        argv.extend([name, value])


def add_default_flag(argv: list[str], name: str) -> None:
    if name not in argv:
        argv.append(name)

