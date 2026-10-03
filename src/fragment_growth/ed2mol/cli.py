"""Command-line interface of the fragment-growth engine."""

from __future__ import annotations

import argparse
import os
import shutil

from .ed import load_ligand_ed
from .grow import run_generation
from .grow_binding import run_generation_binding
from .utils.Utils import GetConfigs, SetSeed


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ED-guided fragment growth")
    parser.add_argument("--config", type=str, required=True, help="Path to config (INI-style)")
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Do not delete the temporary working directory under output_dir/tmp",
    )
    return parser


def run(config_path: str, keep_tmp: bool = False) -> None:
    configs = GetConfigs(config_path)

    SetSeed(int(configs.get("sample", "seed")))

    output_dir = os.path.abspath(
        os.path.expanduser(configs.get("sample", "output_dir"))
    )
    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    liged_dir = (
        configs.get("sample", "ligED_dir")
        if configs.has_option("sample", "ligED_dir")
        else output_dir
    )
    liged = load_ligand_ed(liged_dir) if liged_dir else None

    use_binding_energy = configs.has_option(
        "sample", "use_binding_energy"
    ) and configs.getboolean("sample", "use_binding_energy")
    if use_binding_energy:
        print(f"Using binding energy: {use_binding_energy}")
        run_generation_binding(liged, configs)
    else:
        print("No binding energy was set")
        run_generation(liged, configs)

    if not keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    run(args.config, keep_tmp=args.keep_tmp)


if __name__ == "__main__":
    main()
