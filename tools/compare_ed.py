#!/usr/bin/env python3
"""Compare electron-density arrays with shape/statistical and voxel metrics."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def load_array(path: Path, key: str | None = None) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float64)
    if path.suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            record = pickle.load(handle)
        if key:
            return np.asarray(record[key], dtype=np.float64)
        for candidate in ("aligned_ed", "host_ed", "host_aligned_ed", "host_electron_density"):
            if candidate in record:
                return np.asarray(record[candidate], dtype=np.float64)
        raise KeyError(f"No recognized ED key in {path}; keys={sorted(record)}")
    raise ValueError(f"Unsupported file: {path}")


def describe(array: np.ndarray, threshold: float) -> dict[str, object]:
    return {
        "shape": list(array.shape),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "sum": float(array.sum()),
        "nonzero": int(np.count_nonzero(array)),
        "above_threshold": int(np.count_nonzero(array > threshold)),
    }


def compare(reference: np.ndarray, candidate: np.ndarray, threshold: float) -> dict[str, object]:
    if reference.shape != candidate.shape:
        return {
            "pass_shape": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    delta = candidate - reference
    ref_flat = reference.ravel()
    cand_flat = candidate.ravel()
    ref_centered = ref_flat - ref_flat.mean()
    cand_centered = cand_flat - cand_flat.mean()
    denom = np.linalg.norm(ref_centered) * np.linalg.norm(cand_centered)
    pearson = float(np.dot(ref_centered, cand_centered) / denom) if denom else float("nan")
    cosine_denom = np.linalg.norm(ref_flat) * np.linalg.norm(cand_flat)
    cosine = float(np.dot(ref_flat, cand_flat) / cosine_denom) if cosine_denom else float("nan")
    ref_mask = reference > threshold
    cand_mask = candidate > threshold
    union = int(np.count_nonzero(ref_mask | cand_mask))
    intersection = int(np.count_nonzero(ref_mask & cand_mask))
    return {
        "pass_shape": True,
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "max_abs_error": float(np.max(np.abs(delta))),
        "pearson": pearson,
        "cosine": cosine,
        "mask_iou": float(intersection / union) if union else 1.0,
        "mass_ratio_candidate_over_reference": float(candidate.sum() / reference.sum())
        if reference.sum()
        else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--reference-key")
    parser.add_argument("--candidate-key")
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reference = load_array(args.reference, args.reference_key)
    candidate = load_array(args.candidate, args.candidate_key)
    report = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "threshold": args.threshold,
        "reference_stats": describe(reference, args.threshold),
        "candidate_stats": describe(candidate, args.threshold),
        "comparison": compare(reference, candidate, args.threshold),
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
