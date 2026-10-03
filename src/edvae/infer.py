#!/usr/bin/env python3
"""ED-VAE inference: generate guest electron-density grids from host grids.

For every target directory under ``--input-dir`` the host grid is read from
``<target>/<input-name>`` (default: ``property_aligned_scale_0_1.pkl``), the
VAE generates the guest grid, and ``<target>.npy`` (float32, 48^3) is written
to ``--output-dir`` together with an ``inference_manifest.json`` that records
the checkpoint hash, architecture, seed and output statistics.

The VAE samples its latent code, so reliable regression requires ``--seed``
(optionally with ``--deterministic``). Without a seed every run samples a new
guest density.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from ed2mol.vae import VAE


ARCHITECTURE = {
    "max_channels": 128,
    "encoder_kernel_sizes": [5, 5, 4, 4, 1],
    "encoder_strides": [2, 2, 2, 2, 1],
    "encoder_paddings": [2, 2, 1, 1, 0],
    "decoder_kernel_sizes": [1, 4, 4, 5, 5],
    "decoder_strides": [1, 2, 2, 2, 2],
    "decoder_paddings": [0, 1, 1, 2, 2],
    "decoder_output_paddings": [0, 0, 0, 1, 1],
    "latent_dim": 128,
    "dropout_ratio": 0.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_all(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def load_model(checkpoint: Path, device: torch.device) -> VAE:
    model = VAE(**ARCHITECTURE)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {
        (key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key): value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def load_host_ed(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        record = pickle.load(handle)
    for key in ("aligned_ed", "host_aligned_ed", "host_electron_density", "host_ed"):
        if key in record:
            array = np.asarray(record[key], dtype=np.float32)
            break
    else:
        raise KeyError(f"No host ED key found in {path}; keys={sorted(record)}")
    if array.shape != (48, 48, 48):
        raise ValueError(f"Expected a 48^3 host ED grid, got {array.shape} from {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite host ED values in {path}")
    return array


def discover_targets(input_dir: Path, names: list[str] | None) -> list[str]:
    if names:
        return [name.lower() for name in names]
    return sorted(path.name.lower() for path in input_dir.iterdir() if path.is_dir())


def stats(array: np.ndarray) -> dict[str, float | int | list[int]]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "sum": float(array.sum()),
        "voxels_gt_0.01": int(np.count_nonzero(array > 0.01)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--targets", nargs="*", help="Case names; default discovers all directories")
    parser.add_argument(
        "--input-name",
        default="property_aligned_scale_0_1.pkl",
        help="Pickle filename inside each target directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible sampling.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Disable cuDNN benchmarking and request deterministic algorithms; requires --seed.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.deterministic and args.seed is None:
        parser.error("--deterministic requires --seed")
    if args.seed is not None:
        seed_all(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, device)

    manifest: dict[str, object] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "architecture": ARCHITECTURE,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "device": str(device),
        "torch": torch.__version__,
        "cases": {},
    }
    failures: dict[str, str] = {}
    for target in discover_targets(args.input_dir, args.targets):
        source = args.input_dir / target / args.input_name
        try:
            host_ed = load_host_ed(source)
            tensor = torch.from_numpy(host_ed).unsqueeze(0).to(device)
            with torch.no_grad():
                generated, mu, logvar = model(tensor)
            guest_ed = generated.detach().cpu().numpy().squeeze().astype(np.float32)
            destination = args.output_dir / f"{target}.npy"
            np.save(destination, guest_ed)
            manifest["cases"][target] = {
                "input": str(source.resolve()),
                "input_stats": stats(host_ed),
                "output": str(destination.resolve()),
                "output_stats": stats(guest_ed),
                "mu_mean": float(mu.mean().cpu()),
                "logvar_mean": float(logvar.mean().cpu()),
            }
            print(f"[OK] {target}: {destination}")
        except Exception as exc:  # continue across an explicitly requested batch
            failures[target] = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR] {target}: {failures[target]}")

    manifest["failures"] = failures
    manifest_path = args.output_dir / "inference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if failures:
        raise SystemExit(f"Inference failed for {len(failures)} case(s); see {manifest_path}")


if __name__ == "__main__":
    main()
