"""Build an ED-VAE training LMDB from processed electron-density pickles.

Input layout: a directory tree of ``<base_id>_<aug_idx>/property_aligned.pkl``
files, each containing the ``host_aligned_ed`` and ``guest_aligned_ed`` 48^3
float arrays. Every valid sample is written under an ASCII integer key as a
pickled ``{"base_id", "aug_idx", "host_ed", "guest_ed"}`` record, which is the
schema the ED-VAE training code reads.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import (
    PreprocessingError,
    reject_output_inside_sources,
    sha256_file,
    write_json,
)
from .constants import DEFAULT_LMDB_HIGH_VALUE_POLICY, DEFAULT_NUM_POINTS


def parse_sample_identity(path: Path) -> Tuple[str, int]:
    """Parse ``<base_id>_<aug_idx>/property_aligned.pkl`` robustly."""

    directory_name = path.parent.name
    try:
        base_id, aug_text = directory_name.rsplit("_", 1)
        aug_idx = int(aug_text)
    except (ValueError, IndexError) as exc:
        raise PreprocessingError(
            f"Parent directory must end in _<aug_idx>: {path.parent}"
        ) from exc
    if not base_id:
        raise PreprocessingError(f"Empty base_id parsed from {path.parent}")
    return base_id, aug_idx


def _convert_array(
    array: Any,
    label: str,
    expected_shape: Tuple[int, int, int],
    high_value_policy: str,
) -> np.ndarray:
    values = np.asarray(array)
    if values.shape != expected_shape:
        raise PreprocessingError(
            f"{label} shape is {values.shape}; expected {expected_shape}"
        )
    if not np.isfinite(values).all():
        raise PreprocessingError(f"{label} contains NaN or infinity")
    # Rejection happens before the float32 conversion. Interpolation should not
    # yield negative electron density.
    if (values < 0).any():
        raise PreprocessingError(f"{label} contains negative electron density")
    converted = values.astype(np.float32, copy=True)
    if high_value_policy == "zero":
        # Training-set policy: values > 1 become 0, not 1.
        converted[converted > 1.0] = 0.0
    elif high_value_policy == "clip":
        np.clip(converted, 0.0, 1.0, out=converted)
    elif high_value_policy == "keep":
        pass
    else:
        raise PreprocessingError(f"Unknown high-value policy: {high_value_policy}")
    return converted


def load_processed_sample(
    path: Path,
    num_points: int = DEFAULT_NUM_POINTS,
    high_value_policy: str = DEFAULT_LMDB_HIGH_VALUE_POLICY,
) -> Dict[str, Any]:
    """Load one pickle read-only and return the exact LMDB payload fields."""

    base_id, aug_idx = parse_sample_identity(path)
    try:
        with path.open("rb") as handle:
            raw = pickle.load(handle)
    except Exception as exc:
        raise PreprocessingError(f"Cannot read pickle {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PreprocessingError(f"Processed pickle is not a dict: {path}")
    missing = [key for key in ("host_aligned_ed", "guest_aligned_ed") if key not in raw]
    if missing:
        raise PreprocessingError(f"Missing keys in {path}: {', '.join(missing)}")
    expected_shape = (num_points, num_points, num_points)
    return {
        "base_id": base_id,
        "aug_idx": aug_idx,
        "host_ed": _convert_array(
            raw["host_aligned_ed"], "host_aligned_ed", expected_shape, high_value_policy
        ),
        "guest_ed": _convert_array(
            raw["guest_aligned_ed"],
            "guest_aligned_ed",
            expected_shape,
            high_value_policy,
        ),
    }


def _write_failures_csv(path: Path, failures: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["source", "error_type", "error"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for failure in failures:
            writer.writerow({key: failure.get(key, "") for key in fieldnames})


def build_lmdb(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.input.is_dir():
        raise PreprocessingError(
            f"Processed PKL input directory does not exist: {args.input}"
        )
    reject_output_inside_sources(args.output, [args.input])
    if args.output.exists() and any(args.output.iterdir()):
        raise PreprocessingError(
            f"LMDB output already exists and is non-empty: {args.output}. "
            "Choose a new output directory; this command never deletes an existing dataset."
        )
    candidates = sorted(
        path for path in args.input.rglob(args.pattern) if path.is_file()
    )
    if not candidates:
        raise PreprocessingError(
            f"No files matching {args.pattern!r} found below input {args.input}"
        )

    try:
        import lmdb
    except ImportError as exc:
        raise PreprocessingError(
            "The `lmdb` package is required to build the dataset. "
            "Activate the ED-VAE environment (guestgen)."
        ) from exc

    args.output.mkdir(parents=True, exist_ok=True)
    environment = lmdb.open(
        str(args.output),
        map_size=args.map_size,
        subdir=True,
        create=True,
        readonly=False,
        lock=True,
    )
    transaction = environment.begin(write=True)
    valid_entries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    valid_count = 0
    try:
        for candidate in candidates:
            if valid_count >= args.limit:
                break
            try:
                payload = load_processed_sample(
                    candidate,
                    num_points=args.num_points,
                    high_value_policy=args.high_value_policy,
                )
                key = str(valid_count).encode("ascii")
                transaction.put(key, pickle.dumps(payload))
                valid_entries.append(
                    {
                        "key": key.decode("ascii"),
                        "source": str(candidate.resolve()),
                        "source_sha256": sha256_file(candidate),
                        "base_id": payload["base_id"],
                        "aug_idx": payload["aug_idx"],
                        "host_min": float(payload["host_ed"].min()),
                        "host_max": float(payload["host_ed"].max()),
                        "guest_min": float(payload["guest_ed"].min()),
                        "guest_max": float(payload["guest_ed"].max()),
                    }
                )
                valid_count += 1
                if valid_count % args.commit_frequency == 0:
                    transaction.commit()
                    transaction = environment.begin(write=True)
            except Exception as exc:
                failures.append(
                    {
                        "source": str(candidate.resolve()),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"[SKIP] {candidate}: {exc}", file=sys.stderr)
        transaction.commit()
    except BaseException:
        transaction.abort()
        raise
    finally:
        environment.sync()
        environment.close()

    manifest = {
        "schema_version": 1,
        "stage": "lmdb_build",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "pattern": args.pattern,
        "requested_valid_limit": args.limit,
        "candidate_count": len(candidates),
        "valid_count": valid_count,
        "failure_count": len(failures),
        "num_points": args.num_points,
        "array_shape": [args.num_points, args.num_points, args.num_points],
        "array_dtype": "float32",
        "high_value_policy": args.high_value_policy,
        "negative_value_policy": "reject",
        "nonfinite_policy": "reject",
        "map_size": args.map_size,
        "entries": valid_entries,
        "format_note": "ASCII integer keys; values are pickled dicts; no __len__ key.",
    }
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "failures.json", failures)
    _write_failures_csv(args.output / "failures.csv", failures)
    if valid_count == 0:
        raise PreprocessingError(
            f"No valid samples were written; see {args.output / 'failures.json'}"
        )
    print(f"[DONE] LMDB: {valid_count} valid, {len(failures)} failed")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select deterministic samples from a processed_48 tree and write an "
            "ED-VAE LMDB plus manifest and failure list."
        )
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Processed_48 directory"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New LMDB directory"
    )
    parser.add_argument(
        "--limit", type=int, default=32, help="Number of VALID entries to write"
    )
    parser.add_argument("--pattern", default="property_aligned.pkl")
    parser.add_argument("--num-points", type=int, default=DEFAULT_NUM_POINTS)
    parser.add_argument(
        "--high-value-policy",
        choices=("zero", "clip", "keep"),
        default=DEFAULT_LMDB_HIGH_VALUE_POLICY,
        help="Training-set default is zero: every value >1 becomes 0",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=1 << 40,
        help="LMDB virtual map size (default 1 TiB)",
    )
    parser.add_argument("--commit-frequency", type=int, default=100)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        args.limit <= 0
        or args.num_points <= 0
        or args.map_size <= 0
        or args.commit_frequency <= 0
    ):
        parser.error(
            "--limit, --num-points, --map-size and --commit-frequency must be positive"
        )
    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve(strict=False)
    try:
        build_lmdb(args)
    except PreprocessingError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
