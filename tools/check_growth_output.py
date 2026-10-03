#!/usr/bin/env python3
"""Validate the output of a fragment-growth run.

Checks performed:

* ``output.sdf`` exists and every record can be parsed by RDKit;
* ``ligED.npy`` has the N x 4 grid layout and the documented coordinate
  convention (origin -8.225, last coordinate ~7.88229);
* in binding mode, xTB produced finite single-point JSON files and every
  optimized complex topology kept exactly two fragments.

Malformed inputs are reported through ``growth_check.json`` with
``"pass": false`` instead of raising; the exit status is 0 on success.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from rdkit import Chem

LIGED_ORIGIN = -8.225
LIGED_LAST_COORDINATE = 7.882291666666665


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--mode", choices=("binding", "nonbinding"), required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()

    report: dict[str, object] = {"output_dir": str(root), "mode": args.mode}
    passed = True

    output_sdf = root / "output.sdf"
    if not output_sdf.is_file():
        report["sdf_records"] = 0
        report["rdkit_valid_molecules"] = 0
        passed = False
    else:
        records = list(Chem.SDMolSupplier(str(output_sdf), removeHs=False))
        n_valid = sum(1 for molecule in records if molecule is not None)
        report["sdf_records"] = len(records)
        report["rdkit_valid_molecules"] = n_valid
        report["unreadable_records"] = len(records) - n_valid
        passed &= bool(records) and n_valid == len(records)

    liged_path = root / "ligED.npy"
    if not liged_path.is_file():
        report["liged_shape"] = None
        passed = False
    else:
        liged = np.load(liged_path)
        shape_ok = liged.ndim == 2 and liged.shape[1] >= 4
        report["liged_shape"] = list(liged.shape)
        if shape_ok:
            report["liged_origin"] = liged[0, :3].tolist()
            report["liged_last_coordinate"] = liged[-1, :3].tolist()
            passed &= bool(
                liged.shape == (48**3, 4)
                and np.allclose(liged[0, :3], LIGED_ORIGIN)
                and np.allclose(liged[-1, :3], LIGED_LAST_COORDINATE)
            )
        else:
            passed = False

    if args.mode == "binding":
        xtb_root = root / "xtb_tmp"
        json_paths = sorted(xtb_root.rglob("xtbout.json")) if xtb_root.is_dir() else []
        energies: list[float | None] = []
        unreadable = 0
        for path in json_paths:
            try:
                energies.append(float(json.loads(path.read_text())["total energy"]))
            except Exception:
                energies.append(None)
                unreadable += 1
        finite = [e for e in energies if e is not None and math.isfinite(e)]
        complex_topologies = (
            sorted(xtb_root.glob("*_grow_mols/*/xtbtopo.mol")) if xtb_root.is_dir() else []
        )
        fragment_counts = []
        for path in complex_topologies:
            molecule = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=False)
            fragment_counts.append(len(Chem.GetMolFrags(molecule)) if molecule else None)
        report.update(
            {
                "xtbout_json_count": len(json_paths),
                "xtbout_unreadable": unreadable,
                "xtbout_all_finite": bool(finite) and len(finite) == len(json_paths),
                "successful_complex_topologies": len(complex_topologies),
                "complex_fragment_counts": fragment_counts,
            }
        )
        passed &= bool(
            len(json_paths) >= 3
            and unreadable == 0
            and finite
            and complex_topologies
            and all(count == 2 for count in fragment_counts)
        )

    report["pass"] = bool(passed)
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    (root / "growth_check.json").write_text(payload, encoding="utf-8")
    print(payload, end="")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
