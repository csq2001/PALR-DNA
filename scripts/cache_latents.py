from __future__ import annotations

import sys

from _bootstrap import add_default_arg, add_project_root


def main() -> None:
    add_project_root()
    add_default_arg(sys.argv, "--checkpoint", "outputs/checkpoints/AE.pth")
    add_default_arg(sys.argv, "--out-dir", "outputs/latent_cache")
    from core.latent_cache import main as cache_main

    cache_main()


if __name__ == "__main__":
    main()
