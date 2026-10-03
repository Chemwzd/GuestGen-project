"""Convenience dispatcher for ``python -m preprocessing``."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="GuestGen preprocessing commands")
    parser.add_argument(
        "command", choices=("augment", "electron-density", "build-lmdb")
    )
    args, remaining = parser.parse_known_args(argv)
    if args.command == "augment":
        from .augmentation import main as command_main
    elif args.command == "electron-density":
        from .electron_density import main as command_main
    else:
        from .lmdb_builder import main as command_main
    return command_main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
