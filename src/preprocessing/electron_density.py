"""xTB -> Multiwfn -> 48^3 electron-density preprocessing.

Two layouts are supported:

``host``
    Independent host MOL files such as ``bezcoa.mol``. Each work copy is
    recentered on its heavy-atom centroid before xTB runs.

``training``
    Augmented pairs named ``<base>_1_<aug>.mol`` / ``<base>_2_<aug>.mol`` whose
    coordinates are already pair-centered, so no independent recentering is
    performed. A ``property_aligned.pkl`` with both host and guest grids is
    written for the LMDB builder.

Pipeline per molecule: Open Babel MOL -> XYZ (training layout), ``xtb --molden
--iterations 999``, Multiwfn export of the electron density on a 0.5 Bohr grid
(function 5 -> rho), and interpolation of the scattered points onto the 48^3
grid (adjacent points 0.35 Angstrom, every axis spanning [-8.225, +8.225]).
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import (
    PreprocessingError,
    copy_readonly_input,
    ensure_workdir_is_safe,
    reject_output_inside_sources,
    resolve_executable,
    run_logged,
    sha256_file,
    write_json,
)
from .constants import (
    DEFAULT_MULTIWFN_AUG3D,
    DEFAULT_MULTIWFN_GRID_SPACING,
    DEFAULT_NUM_POINTS,
    DEFAULT_TARGET_STEP_SIZE,
    DEFAULT_XTB_ITERATIONS,
)
from .mol import choose_recenter_backend, recenter_single_native, recenter_single_sugar


@dataclass(frozen=True)
class TrainingPair:
    base_id: str
    aug_idx: int
    host: Path
    guest: Path


def target_coordinates(
    num_points: int = DEFAULT_NUM_POINTS,
    step_size: float = DEFAULT_TARGET_STEP_SIZE,
) -> np.ndarray:
    """Return the exact target order used by the legacy AlignPointCloud step.

    For the default 48-point grid this is a 48^3 C-order array, adjacent points
    are 0.35 apart, and each axis spans [-8.225, +8.225].
    """

    if num_points <= 0:
        raise PreprocessingError("num_points must be positive")
    if step_size <= 0:
        raise PreprocessingError("step_size must be positive")
    mid = (num_points - 1) / 2.0
    axis = (np.arange(num_points, dtype=np.float64) - mid) * step_size
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T


def read_raw_ed(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    try:
        data = np.loadtxt(str(path), dtype=np.float64)
    except Exception as exc:
        raise PreprocessingError(
            f"Cannot parse Multiwfn point cloud {path}: {exc}"
        ) from exc
    if data.ndim != 2 or data.shape[1] != 4:
        raise PreprocessingError(
            f"Expected n x 4 [x y z ED] point cloud in {path}, got {data.shape}"
        )
    if not np.isfinite(data).all():
        raise PreprocessingError(f"Non-finite coordinate/property in {path}")
    return data[:, :3], data[:, 3]


def interpolate_density(
    raw_ed_path: Path,
    num_points: int = DEFAULT_NUM_POINTS,
    step_size: float = DEFAULT_TARGET_STEP_SIZE,
    backend: str = "pyvista",
) -> np.ndarray:
    points, properties = read_raw_ed(raw_ed_path)
    targets = target_coordinates(num_points, step_size)
    if backend == "pyvista":
        try:
            import pyvista as pv
        except ImportError as exc:
            raise PreprocessingError(
                "PyVista is required for the default interpolation. Activate the "
                "ED-VAE environment (guestgen). A SciPy-based alternative is "
                "available for diagnostics via --interpolation-backend scipy."
            ) from exc
        reference_cloud = pv.PolyData(points)
        reference_cloud["ED"] = properties
        target_cloud = pv.PolyData(targets)
        # Called without kernel/radius arguments on purpose: this keeps the
        # interpolation behavior of the released reference grids.
        interpolated = target_cloud.interpolate(reference_cloud)
        values = np.asarray(interpolated["ED"], dtype=np.float32)
    elif backend == "scipy":
        try:
            from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        except ImportError as exc:
            raise PreprocessingError(
                "SciPy fallback requested but scipy is not installed."
            ) from exc
        linear = LinearNDInterpolator(points, properties, fill_value=np.nan)
        values = np.asarray(linear(targets), dtype=np.float64)
        missing = ~np.isfinite(values)
        if missing.any():
            nearest = NearestNDInterpolator(points, properties)
            values[missing] = nearest(targets[missing])
        values = values.astype(np.float32)
    else:
        raise PreprocessingError(f"Unknown interpolation backend: {backend}")
    expected = num_points**3
    if values.size != expected:
        raise PreprocessingError(
            f"Interpolation returned {values.size} values; expected {expected}"
        )
    return values.reshape((num_points, num_points, num_points), order="C")


def clip_01(values: np.ndarray) -> np.ndarray:
    """Conventional [0,1] clipping, enabled only by explicit CLI request."""

    return np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)


def patch_multiwfn_settings(
    source: Path, destination: Path, extension: float
) -> Dict[str, Any]:
    """Copy settings.ini and change only the work-copy ``aug3D`` value."""

    if not source.is_file():
        raise PreprocessingError(
            f"Multiwfn settings.ini is missing: {source}. Pass --multiwfn-settings."
        )
    provenance = copy_readonly_input(source, destination)
    content = destination.read_text(encoding="utf-8", errors="replace")
    # str(float(...)) keeps the textual value `6.0` rather than rewriting it.
    replacement = f"aug3D= {float(extension)}"
    patched, count = re.subn(
        r"(?im)^\s*aug3D\s*=\s*[^\r\n]*(?:\r?\n)?",
        replacement + "\n",
        content,
    )
    if count != 1:
        raise PreprocessingError(
            f"Expected exactly one aug3D setting in {source}, found {count}. "
            "Refusing to guess a Multiwfn protocol."
        )
    destination.write_text(patched, encoding="utf-8")
    provenance["effective_work_copy_sha256"] = sha256_file(destination)
    provenance["effective_aug3D"] = extension
    return provenance


def multiwfn_stdin(grid_spacing: float) -> str:
    """Multiwfn menu sequence that exports rho on a custom grid.

    blank -> function 5 (output grid data) -> rho -> set spacing -> value ->
    export -> exit. ``aug3D`` is taken from the isolated settings.ini copy.
    """

    return f"\n5\n1\n4\n{grid_spacing:g}\n3\n\n"


def discover_hosts(input_path: Path, names: Sequence[str]) -> List[Tuple[str, Path]]:
    if input_path.is_file():
        candidates = [(input_path.stem, input_path)]
    elif input_path.is_dir():
        candidates = [(path.stem, path) for path in sorted(input_path.glob("*.mol"))]
    else:
        raise PreprocessingError(f"Host input does not exist: {input_path}")
    requested = set(names)
    if requested:
        candidates = [(name, path) for name, path in candidates if name in requested]
        found = {name for name, _ in candidates}
        missing = sorted(requested - found)
        if missing:
            raise PreprocessingError(
                f"Requested host MOL files not found in {input_path}: {', '.join(missing)}"
            )
    if not candidates:
        raise PreprocessingError(f"No host MOL files selected from {input_path}")
    return candidates


def discover_training_pairs(input_dir: Path) -> List[TrainingPair]:
    if not input_dir.is_dir():
        raise PreprocessingError(f"Training MOL directory does not exist: {input_dir}")
    grouped: Dict[Tuple[str, int], Dict[str, Path]] = {}
    malformed: List[str] = []
    for path in sorted(input_dir.glob("*.mol")):
        parts = path.stem.rsplit("_", 2)
        if len(parts) != 3 or parts[1] not in {"1", "2"}:
            malformed.append(path.name)
            continue
        try:
            aug_idx = int(parts[2])
        except ValueError:
            malformed.append(path.name)
            continue
        grouped.setdefault((parts[0], aug_idx), {})[parts[1]] = path
    pairs = [
        TrainingPair(base_id, aug_idx, role_paths["1"], role_paths["2"])
        for (base_id, aug_idx), role_paths in sorted(grouped.items())
        if "1" in role_paths and "2" in role_paths
    ]
    if not pairs:
        hint = f"; ignored examples: {malformed[:3]}" if malformed else ""
        raise PreprocessingError(
            "No <base>_1_<aug>.mol / <base>_2_<aug>.mol pairs found in "
            f"{input_dir}{hint}"
        )
    return pairs


def _resolve_settings(args: argparse.Namespace, multiwfn: Optional[Path]) -> Path:
    if args.multiwfn_settings:
        return args.multiwfn_settings.expanduser().resolve()
    env_setting = os.environ.get("GUESTGEN_MULTIWFN_SETTINGS") or os.environ.get(
        "MULTIWFN_SETTINGS"
    )
    if env_setting:
        return Path(env_setting).expanduser().resolve()
    if multiwfn is not None:
        return multiwfn.parent / "settings.ini"
    raise PreprocessingError(
        "Cannot infer Multiwfn settings.ini. Pass --multiwfn-settings or set "
        "GUESTGEN_MULTIWFN_SETTINGS."
    )


def _convert_with_obabel(
    mol_path: Path, xyz_path: Path, obabel: Path, calculation_dir: Path
) -> Path:
    run_logged(
        [str(obabel), str(mol_path), "-O", str(xyz_path)],
        cwd=calculation_dir,
        stdout_path=calculation_dir / "obabel.stdout.log",
        stderr_path=calculation_dir / "obabel.stderr.log",
    )
    if not xyz_path.is_file():
        raise PreprocessingError(f"Open Babel did not create expected XYZ: {xyz_path}")
    return xyz_path


def _one_molecule(
    source_mol: Path,
    molecule_id: str,
    workdir: Path,
    raw_output_dir: Path,
    args: argparse.Namespace,
    tools: Dict[str, Optional[Path]],
    recenter_host: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    calculation_dir = workdir / "calculation"
    calculation_dir.mkdir(parents=True, exist_ok=True)
    copied_mol = calculation_dir / source_mol.name
    input_provenance = copy_readonly_input(source_mol, copied_mol)

    recenter_backend = "none"
    if recenter_host:
        recenter_backend = choose_recenter_backend(args.recenter_backend)
        if args.recenter_backend == "auto" and recenter_backend == "native":
            print(
                "[WARN] `sugar` is unavailable; using V2000 native host "
                "centering. Pass --recenter-backend sugar to require strict "
                "sugar conversion.",
                file=sys.stderr,
            )
        if recenter_backend == "sugar":
            recenter_single_sugar(copied_mol)
        elif recenter_backend == "native":
            recenter_single_native(copied_mol)
        elif recenter_backend != "none":
            raise PreprocessingError(f"Unknown recenter backend: {recenter_backend}")
    input_provenance["effective_work_copy_sha256"] = sha256_file(copied_mol)

    manifest: Dict[str, Any] = {
        "molecule_id": molecule_id,
        "input": input_provenance,
        "recenter_backend": recenter_backend,
        "parameters": {
            "multiwfn_grid_spacing": args.grid_spacing,
            "multiwfn_aug3D_extension": args.extension,
            "num_points": args.num_points,
            "target_step_size": args.step_size,
            "target_axis_min": -((args.num_points - 1) / 2.0) * args.step_size,
            "target_axis_max": ((args.num_points - 1) / 2.0) * args.step_size,
            "clip_01": bool(args.clip_01),
            "interpolation_backend": args.interpolation_backend,
        },
    }
    if args.dry_run:
        manifest["status"] = "planned"
        return np.empty((0,), dtype=np.float32), manifest

    xtb = tools["xtb"]
    multiwfn = tools["multiwfn"]
    if xtb is None or multiwfn is None:
        raise PreprocessingError(
            "Internal error: external executables were not resolved"
        )
    settings_source = _resolve_settings(args, multiwfn)
    settings_provenance = patch_multiwfn_settings(
        settings_source, calculation_dir / "settings.ini", args.extension
    )

    xtb_input = copied_mol
    input_format = args.xtb_input_format
    if input_format == "auto":
        # Training MOLs are converted to XYZ with Open Babel; host ED
        # generation passes the MOL file directly to xTB.
        input_format = "xyz" if args.layout == "training" else "mol"
    if input_format == "xyz":
        obabel = tools["obabel"]
        if obabel is None:
            raise PreprocessingError(
                "XYZ xTB input requested but Open Babel is unavailable. Pass "
                "--obabel/OBABEL_BIN, or use --xtb-input-format mol."
            )
        xtb_input = _convert_with_obabel(
            copied_mol,
            calculation_dir / f"{source_mol.stem}.xyz",
            obabel,
            calculation_dir,
        )

    xtb_command = [
        str(xtb),
        xtb_input.name,
        "--molden",
        "--iterations",
        str(args.xtb_iterations),
    ]
    run_logged(
        xtb_command,
        cwd=calculation_dir,
        stdout_path=calculation_dir / "xtb.stdout.log",
        stderr_path=calculation_dir / "xtb.stderr.log",
    )
    molden = calculation_dir / "molden.input"
    if not molden.is_file():
        raise PreprocessingError(
            f"xTB did not produce molden.input in {calculation_dir}"
        )

    multiwfn_command = [str(multiwfn), molden.name]
    multiwfn_returncode = run_logged(
        multiwfn_command,
        cwd=calculation_dir,
        stdout_path=calculation_dir / "molden2ed.log",
        stderr_path=calculation_dir / "molden2ed.stderr.log",
        stdin_text=multiwfn_stdin(args.grid_spacing),
        # The menu sequence ends at EOF after the export. Multiwfn 3.8 writes a
        # complete output.txt and then exits with Fortran status 24 while
        # waiting for another menu choice; that status is accepted because the
        # required output file is checked below.
        allowed_returncodes=(0, 24),
    )
    raw_ed = calculation_dir / "output.txt"
    if not raw_ed.is_file():
        raise PreprocessingError(
            f"Multiwfn did not produce output.txt in {calculation_dir}; "
            "inspect molden2ed.log"
        )
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    final_raw = raw_output_dir / "output.txt"
    shutil.copy2(str(raw_ed), str(final_raw))

    density = interpolate_density(
        final_raw,
        num_points=args.num_points,
        step_size=args.step_size,
        backend=args.interpolation_backend,
    )
    if args.clip_01:
        density = clip_01(density)
    manifest.update(
        {
            "status": "complete",
            "raw_ed": str(final_raw.resolve()),
            "raw_ed_sha256": sha256_file(final_raw),
            "array": {
                "shape": list(density.shape),
                "dtype": str(density.dtype),
                "minimum": float(density.min()),
                "maximum": float(density.max()),
                "nonzero": int(np.count_nonzero(density)),
            },
            "tools": {
                "xtb": str(xtb),
                "xtb_sha256": sha256_file(xtb),
                "multiwfn": str(multiwfn),
                "multiwfn_sha256": sha256_file(multiwfn),
                "obabel": str(tools["obabel"]) if tools["obabel"] else None,
                "obabel_sha256": (
                    sha256_file(tools["obabel"]) if tools["obabel"] else None
                ),
                "multiwfn_settings": settings_provenance,
                "multiwfn_stdin": multiwfn_stdin(args.grid_spacing).splitlines(),
            },
            "commands": {"xtb": xtb_command, "multiwfn": multiwfn_command},
            "multiwfn_returncode": multiwfn_returncode,
        }
    )
    return density, manifest


def _resolve_tools(args: argparse.Namespace) -> Dict[str, Optional[Path]]:
    if args.dry_run:
        # Dry-run tests selection, source copying, centering and manifest
        # generation only, so it stays usable without the external programs.
        return {"xtb": None, "multiwfn": None, "obabel": None}
    xtb = resolve_executable(args.xtb, ("GUESTGEN_XTB", "XTB_BIN"), "xTB")
    multiwfn = resolve_executable(
        args.multiwfn, ("GUESTGEN_MULTIWFN", "MULTIWFN_BIN"), "Multiwfn"
    )
    obabel: Optional[Path] = None
    if args.obabel or os.environ.get("GUESTGEN_OBABEL") or os.environ.get("OBABEL_BIN"):
        obabel = resolve_executable(
            args.obabel or "obabel", ("GUESTGEN_OBABEL", "OBABEL_BIN"), "Open Babel"
        )
    elif args.xtb_input_format == "xyz" or (
        args.xtb_input_format == "auto" and args.layout == "training"
    ):
        obabel = resolve_executable(
            args.obabel or "obabel", ("GUESTGEN_OBABEL", "OBABEL_BIN"), "Open Babel"
        )
    return {"xtb": xtb, "multiwfn": multiwfn, "obabel": obabel}


def _run_host(
    args: argparse.Namespace, tools: Dict[str, Optional[Path]]
) -> Dict[str, Any]:
    hosts = discover_hosts(args.input, args.name)
    if args.limit is not None:
        hosts = hosts[: args.limit]
    successes = []
    failures = []
    for molecule_id, source_mol in hosts:
        try:
            density, item_manifest = _one_molecule(
                source_mol,
                molecule_id,
                args.workdir / molecule_id,
                args.output / molecule_id,
                args,
                tools,
                recenter_host=True,
            )
            if not args.dry_run:
                result_file = (
                    args.output / molecule_id / "property_aligned_scale_0_1.pkl"
                )
                with result_file.open("wb") as handle:
                    pickle.dump({"aligned_ed": density}, handle)
                item_manifest["aligned_pkl"] = str(result_file.resolve())
                item_manifest["aligned_pkl_sha256"] = sha256_file(result_file)
            write_json(args.output / molecule_id / "manifest.json", item_manifest)
            successes.append(item_manifest)
            print(f"[{'PLAN' if args.dry_run else 'DONE'}] host {molecule_id}")
        except Exception as exc:
            failures.append(
                {
                    "molecule_id": molecule_id,
                    "source": str(source_mol.resolve()),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[ERROR] host {molecule_id}: {exc}", file=sys.stderr)
    return {"selected": len(hosts), "successes": successes, "failures": failures}


def _run_training(
    args: argparse.Namespace, tools: Dict[str, Optional[Path]]
) -> Dict[str, Any]:
    pairs = discover_training_pairs(args.input)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    processed_root = args.output / f"processed_{args.num_points}"
    raw_root = args.output / "raw_ed"
    successes = []
    failures = []
    for pair in pairs:
        pair_id = f"{pair.base_id}_{pair.aug_idx}"
        try:
            host_density, host_manifest = _one_molecule(
                pair.host,
                f"{pair.base_id}_1_{pair.aug_idx}",
                args.workdir / pair_id / "host",
                raw_root / f"{pair.base_id}_1_{pair.aug_idx}",
                args,
                tools,
                recenter_host=False,
            )
            guest_density, guest_manifest = _one_molecule(
                pair.guest,
                f"{pair.base_id}_2_{pair.aug_idx}",
                args.workdir / pair_id / "guest",
                raw_root / f"{pair.base_id}_2_{pair.aug_idx}",
                args,
                tools,
                recenter_host=False,
            )
            pair_manifest = {
                "base_id": pair.base_id,
                "aug_idx": pair.aug_idx,
                "host": host_manifest,
                "guest": guest_manifest,
            }
            pair_output = processed_root / pair_id
            pair_output.mkdir(parents=True, exist_ok=True)
            if not args.dry_run:
                result_file = pair_output / "property_aligned.pkl"
                with result_file.open("wb") as handle:
                    pickle.dump(
                        {
                            "host_aligned_ed": host_density,
                            "guest_aligned_ed": guest_density,
                        },
                        handle,
                    )
                pair_manifest["aligned_pkl"] = str(result_file.resolve())
                pair_manifest["aligned_pkl_sha256"] = sha256_file(result_file)
            write_json(pair_output / "manifest.json", pair_manifest)
            successes.append(pair_manifest)
            print(f"[{'PLAN' if args.dry_run else 'DONE'}] training pair {pair_id}")
        except Exception as exc:
            failures.append(
                {
                    "base_id": pair.base_id,
                    "aug_idx": pair.aug_idx,
                    "host": str(pair.host.resolve()),
                    "guest": str(pair.guest.resolve()),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[ERROR] training pair {pair_id}: {exc}", file=sys.stderr)
    return {"selected": len(pairs), "successes": successes, "failures": failures}


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    source_root = args.input if args.input.is_dir() else args.input.parent
    reject_output_inside_sources(args.output, [source_root])
    ensure_workdir_is_safe(args.workdir, [source_root])
    args.output.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)
    tools = _resolve_tools(args)
    result = (
        _run_host(args, tools) if args.layout == "host" else _run_training(args, tools)
    )
    summary = {
        "schema_version": 1,
        "stage": "electron_density",
        "layout": args.layout,
        "source_policy": "readonly-copy-before-processing",
        "dry_run": bool(args.dry_run),
        "parameters": {
            "multiwfn_grid_spacing": args.grid_spacing,
            "multiwfn_aug3D_extension": args.extension,
            "num_points": args.num_points,
            "target_step_size": args.step_size,
            "clip_01": bool(args.clip_01),
            "interpolation_backend": args.interpolation_backend,
            "xtb_iterations": args.xtb_iterations,
            "xtb_input_format": args.xtb_input_format,
        },
        "selected_count": result["selected"],
        "success_count": len(result["successes"]),
        "failure_count": len(result["failures"]),
        "successes": result["successes"],
    }
    write_json(args.output / "ed_manifest.json", summary)
    write_json(args.output / "ed_failures.json", result["failures"])
    if not result["successes"]:
        raise PreprocessingError(
            f"All {result['selected']} ED items failed; see {args.output / 'ed_failures.json'}"
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run source-readonly xTB/Multiwfn ED generation and the 48^3 "
            "PyVista interpolation."
        )
    )
    parser.add_argument("--layout", choices=("host", "training"), required=True)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Host MOL file/dir or augmented training MOL dir",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output directory"
    )
    parser.add_argument(
        "--workdir", type=Path, help="Isolated workdir (default OUTPUT/_work)"
    )
    parser.add_argument(
        "--name",
        action="append",
        default=[],
        help="Host ID to select; repeat for bezcoa, levpuz and qupyik",
    )
    parser.add_argument("--limit", type=int, help="Maximum hosts or augmented pairs")
    parser.add_argument("--xtb", help="xTB executable; env GUESTGEN_XTB/XTB_BIN")
    parser.add_argument(
        "--multiwfn", help="Multiwfn executable; env GUESTGEN_MULTIWFN/MULTIWFN_BIN"
    )
    parser.add_argument(
        "--multiwfn-settings",
        type=Path,
        help="settings.ini to copy; aug3D is patched only in the work copy",
    )
    parser.add_argument(
        "--obabel", help="Open Babel executable; env GUESTGEN_OBABEL/OBABEL_BIN"
    )
    parser.add_argument(
        "--xtb-input-format",
        choices=("auto", "mol", "xyz"),
        default="auto",
        help="auto uses XYZ for training and MOL for host layout",
    )
    parser.add_argument("--xtb-iterations", type=int, default=DEFAULT_XTB_ITERATIONS)
    parser.add_argument("--grid-spacing", type=float, default=DEFAULT_MULTIWFN_GRID_SPACING)
    parser.add_argument(
        "--extension",
        type=float,
        default=DEFAULT_MULTIWFN_AUG3D,
        help="Multiwfn settings.ini aug3D (Bohr)",
    )
    parser.add_argument("--num-points", type=int, default=DEFAULT_NUM_POINTS)
    parser.add_argument("--step-size", type=float, default=DEFAULT_TARGET_STEP_SIZE)
    parser.add_argument(
        "--interpolation-backend",
        choices=("pyvista", "scipy"),
        default="pyvista",
        help="pyvista is the released default; scipy is a diagnostic alternative",
    )
    parser.add_argument(
        "--clip-01",
        action="store_true",
        help=(
            "Explicitly np.clip the aligned array to [0,1]. The released host "
            "grids are stored without clipping even though some values exceed 1."
        ),
    )
    parser.add_argument(
        "--recenter-backend",
        choices=("auto", "sugar", "native", "none"),
        default="auto",
        help="Host layout only; training pairs keep their shared coordinates",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Copy/select/recenter and emit manifests without xTB/Multiwfn/interpolation",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name, value in (
        ("--grid-spacing", args.grid_spacing),
        ("--extension", args.extension),
        ("--step-size", args.step_size),
    ):
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.num_points <= 0 or args.xtb_iterations <= 0:
        parser.error("--num-points and --xtb-iterations must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.layout == "training" and args.name:
        parser.error("--name applies only to --layout host")
    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve(strict=False)
    args.workdir = (
        args.workdir.expanduser().resolve(strict=False)
        if args.workdir
        else args.output / "_work"
    )
    try:
        run_pipeline(args)
    except PreprocessingError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
