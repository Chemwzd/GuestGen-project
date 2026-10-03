"""Create an augmented host/guest training set.

Every source MOL is copied into an isolated work directory before any
transform is applied; source files are never modified. The transform kernel is
:mod:`preprocessing.legacy_augment` (see the module docstring for the two
intentional compatibility behaviors: per-pair re-seeding and the rotation-angle
expression).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import legacy_augment
from .common import (
    PreprocessingError,
    copy_readonly_input,
    ensure_workdir_is_safe,
    reject_output_inside_sources,
    sha256_file,
    write_json,
)
from .constants import (
    DEFAULT_AUGMENTATION_SEED,
    DEFAULT_NUM_AUG,
    DEFAULT_ROTATION_STEP_SIZE,
    DEFAULT_TRANSLATION_STEP_SIZE,
)
from .mol import choose_recenter_backend, recenter_pair_native, recenter_pair_sugar


@dataclass(frozen=True)
class MoleculePair:
    base_id: str
    host: Path
    guest: Path


def _strip_role(stem: str, role: str) -> str:
    suffixes = ("_1", "_host") if role == "host" else ("_2", "_guest")
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def discover_pairs(
    host: Optional[Path],
    guest: Optional[Path],
    input_dir: Optional[Path],
    explicit_name: Optional[str] = None,
) -> List[MoleculePair]:
    """Discover either one explicit pair or deterministic directory pairs."""

    if input_dir is not None:
        if host is not None or guest is not None:
            raise PreprocessingError("Use --input-dir OR --host/--guest, not both.")
        if not input_dir.is_dir():
            raise PreprocessingError(
                f"Pair input directory does not exist: {input_dir}"
            )
        pairs = []
        for host_path in sorted(input_dir.glob("*_1.mol")):
            base_id = host_path.stem[:-2]
            guest_path = input_dir / f"{base_id}_2.mol"
            if guest_path.is_file():
                pairs.append(MoleculePair(base_id, host_path, guest_path))
        if not pairs:
            raise PreprocessingError(
                f"No colocated <base>_1.mol / <base>_2.mol pairs found in {input_dir}"
            )
        return pairs

    if host is None or guest is None:
        raise PreprocessingError("Provide both --host and --guest, or use --input-dir.")
    if host.is_file() and guest.is_file():
        if explicit_name:
            base_id = explicit_name
        else:
            base_id = _strip_role(host.stem, "host")
        return [MoleculePair(base_id, host, guest)]
    if host.is_dir() and guest.is_dir():
        host_map = {_strip_role(path.stem, "host"): path for path in host.glob("*.mol")}
        guest_map = {
            _strip_role(path.stem, "guest"): path for path in guest.glob("*.mol")
        }
        shared = sorted(set(host_map).intersection(guest_map))
        if not shared:
            raise PreprocessingError(
                f"No matching MOL names found in host={host} and guest={guest}"
            )
        return [MoleculePair(key, host_map[key], guest_map[key]) for key in shared]
    raise PreprocessingError(
        "--host and --guest must both be files or both be directories."
    )


def _safe_base_id(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise PreprocessingError(f"Unsafe pair name: {value!r}")
    return value


def run_augmentation(args: argparse.Namespace) -> Dict[str, Any]:
    pairs = discover_pairs(args.host, args.guest, args.input_dir, args.name)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise PreprocessingError("No host/guest pairs selected after --limit.")

    source_roots = []
    if args.input_dir is not None:
        source_roots.append(args.input_dir)
    else:
        source_roots.extend(
            [
                args.host if args.host.is_dir() else args.host.parent,
                args.guest if args.guest.is_dir() else args.guest.parent,
            ]
        )
    reject_output_inside_sources(args.output, source_roots)
    ensure_workdir_is_safe(args.workdir, source_roots)
    args.output.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)
    molecule_output = args.output / "molecules"
    molecule_output.mkdir(parents=True, exist_ok=True)

    backend = choose_recenter_backend(args.recenter_backend)
    if args.recenter_backend == "auto" and backend == "native":
        print(
            "[WARN] `sugar` is unavailable; using the V2000 native centering "
            "fallback. Pass --recenter-backend sugar to require strict sugar "
            "output.",
            file=sys.stderr,
        )

    successes: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for pair in pairs:
        base_id = _safe_base_id(pair.base_id)
        pair_work = args.workdir / base_id
        copied_dir = pair_work / "inputs"
        generated_dir = pair_work / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        copied_host = copied_dir / f"{base_id}_1.mol"
        copied_guest = copied_dir / f"{base_id}_2.mol"
        try:
            input_provenance = {
                "host": copy_readonly_input(pair.host, copied_host),
                "guest": copy_readonly_input(pair.guest, copied_guest),
            }
            legacy_augment.generate_aug_files_double(
                str(copied_host),
                str(copied_guest),
                outdir=str(generated_dir),
                out_prefix=base_id,
                num_aug=args.num_aug,
                rotation_step_size=args.rotation_step_size,
                translation_step_size=args.translation_step_size,
                seed=args.seed,
            )

            outputs = []
            for aug_idx in range(args.num_aug):
                aug_host = generated_dir / f"{base_id}_1_{aug_idx}.mol"
                aug_guest = generated_dir / f"{base_id}_2_{aug_idx}.mol"
                if backend == "sugar":
                    recenter_pair_sugar(aug_host, aug_guest)
                elif backend == "native":
                    recenter_pair_native(aug_host, aug_guest)
                elif backend != "none":
                    raise PreprocessingError(f"Unknown recenter backend: {backend}")
                final_host = molecule_output / aug_host.name
                final_guest = molecule_output / aug_guest.name
                shutil.copy2(str(aug_host), str(final_host))
                shutil.copy2(str(aug_guest), str(final_guest))
                outputs.append(
                    {
                        "aug_idx": aug_idx,
                        "host": str(final_host.resolve()),
                        "host_sha256": sha256_file(final_host),
                        "guest": str(final_guest.resolve()),
                        "guest_sha256": sha256_file(final_guest),
                    }
                )
            successes.append(
                {
                    "base_id": base_id,
                    "inputs": input_provenance,
                    "outputs": outputs,
                }
            )
            print(f"[DONE] augmented {base_id}: {args.num_aug} pairs")
        except Exception as exc:  # continue with the remaining pairs
            failures.append(
                {
                    "base_id": base_id,
                    "host": str(pair.host.resolve()),
                    "guest": str(pair.guest.resolve()),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[ERROR] {base_id}: {exc}", file=sys.stderr)

    manifest = {
        "schema_version": 1,
        "stage": "structural_augmentation",
        "source_policy": "readonly-copy-before-processing",
        "kernel": "preprocessing.legacy_augment",
        "parameters": {
            "num_aug": args.num_aug,
            "rotation_step_size": args.rotation_step_size,
            "translation_step_size": args.translation_step_size,
            "seed": args.seed,
            "recenter_backend": backend,
        },
        "selected_pair_count": len(pairs),
        "success_count": len(successes),
        "failure_count": len(failures),
        "successes": successes,
    }
    write_json(args.output / "augmentation_manifest.json", manifest)
    write_json(args.output / "augmentation_failures.json", failures)
    if not successes:
        raise PreprocessingError(
            f"All {len(failures)} selected pairs failed; see augmentation_failures.json"
        )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a host/guest augmentation set. Every source MOL is copied "
            "to a work directory before any transformation."
        )
    )
    source = parser.add_argument_group("input selection")
    source.add_argument(
        "--input-dir", type=Path, help="Directory with <base>_1.mol/<base>_2.mol pairs"
    )
    source.add_argument("--host", type=Path, help="Host MOL file or host MOL directory")
    source.add_argument(
        "--guest", type=Path, help="Guest MOL file or guest MOL directory"
    )
    source.add_argument("--name", help="Output base ID for one explicit file pair")
    source.add_argument(
        "--limit",
        type=int,
        help="Maximum number of base pairs (sorted deterministically)",
    )

    parser.add_argument(
        "--output", type=Path, required=True, help="Output directory"
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        help="Isolated work directory (default: OUTPUT/_work)",
    )
    parser.add_argument("--num-aug", type=int, default=DEFAULT_NUM_AUG)
    parser.add_argument(
        "--rotation-step-size", type=float, default=DEFAULT_ROTATION_STEP_SIZE
    )
    parser.add_argument(
        "--translation-step-size", type=float, default=DEFAULT_TRANSLATION_STEP_SIZE
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_AUGMENTATION_SEED)
    parser.add_argument(
        "--recenter-backend",
        choices=("auto", "sugar", "native", "none"),
        default="auto",
        help="`sugar` is the released dataset backend; native is a V2000 fallback",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.num_aug <= 0:
        parser.error("--num-aug must be positive")
    args.output = args.output.expanduser().resolve(strict=False)
    args.workdir = (
        args.workdir.expanduser().resolve(strict=False)
        if args.workdir
        else args.output / "_work"
    )
    try:
        run_augmentation(args)
    except PreprocessingError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
