#!/usr/bin/env python3
"""Validate the output of a fragment-growth run.

Checks performed:

* ``output.sdf`` exists and its molecules can be parsed by RDKit;
* ``ligED.npy`` has the N x 4 grid layout and the documented coordinate
  convention (origin -8.225, last coordinate ~7.88229);
* in binding mode, xTB produced finite single-point JSON files and every
  optimized complex topology kept exactly two fragments.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from rdkit import Chem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--mode", choices=("binding", "nonbinding"), required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()

    output_sdf = root / "output.sdf"
    molecules = [
        molecule for molecule in Chem.SDMolSupplier(str(output_sdf), removeHs=False) if molecule
    ]
    liged = np.load(root / "ligED.npy")
    report: dict[str, object] = {
        "output_dir": str(root),
        "mode": args.mode,
        "rdkit_valid_molecules": len(molecules),
        "liged_shape": list(liged.shape),
        "liged_origin": liged[0, :3].tolist(),
        "liged_last_coordinate": liged[-1, :3].tolist(),
    }
    passed = bool(
        molecules
        and liged.shape == (48**3, 4)
        and np.allclose(liged[0, :3], -8.225)
        and np.allclose(liged[-1, :3], 7.882291666666665)
    )

    if args.mode == "binding":
        json_paths = sorted((root / "xtb_tmp").rglob("xtbout.json"))
        energies = [float(json.loads(path.read_text())["total energy"]) for path in json_paths]
        complex_topologies = sorted((root / "xtb_tmp").glob("*_grow_mols/*/xtbtopo.mol"))
        fragment_counts = []
        for path in complex_topologies:
            molecule = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=False)
            fragment_counts.append(len(Chem.GetMolFrags(molecule)) if molecule else None)
        report.update(
            {
                "xtbout_json_count": len(json_paths),
                "xtbout_all_finite": bool(
                    energies and all(math.isfinite(value) for value in energies)
                ),
                "successful_complex_topologies": len(complex_topologies),
                "complex_fragment_counts": fragment_counts,
            }
        )
        passed &= bool(
            len(json_paths) >= 3
            and energies
            and all(math.isfinite(value) for value in energies)
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
