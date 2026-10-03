#!/usr/bin/env python3
"""Command-line entry point for ED-guided fragment growth.

Usage:
    python Generate.py --config <config.ini>

The INI config format is documented in the project README. ``run_case.py``
prepares project-owned inputs, writes a compatible config and can invoke this
entry point directly.
"""

from ed2mol.cli import main

if __name__ == "__main__":
    main()
