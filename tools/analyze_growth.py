#!/usr/bin/env python3
"""Summarize a fragment-growth run.

Reads ``output.sdf`` from a growth run directory and reports:

* validity (RDKit-readable molecules) and uniqueness (canonical SMILES);
* internal diversity: 1 - mean pairwise ECFP4 Tanimoto similarity over a
  random sample of the molecules;
* property distributions (MW, logP, HBD, HBA, rotatable bonds, TPSA,
  heavy-atom count, SA score, Lipinski violations);
* the fraction of molecules within typical drug-like thresholds
  (logP <= 5, SA score <= 8, heavy atoms <= 50, rotatable bonds <= 5,
  MW <= 300), matching the analysis thresholds of the reference runs;
* optionally, binding-energy statistics from a CSV/XLSX file that contains a
  ``binding_energy`` column (the sentinel value 999 marks failed xTB runs and
  negative values are treated as valid energies).

Outputs ``analysis_summary.json`` and ``molecule_properties.csv`` inside the
run directory (or ``--output-dir``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, RDConfig, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import BulkTanimotoSimilarity

RDLogger.DisableLog("rdApp.*")
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402


DRUG_LIKE_THRESHOLDS = {
    "logp_max": 5.0,
    "sa_score_max": 8.0,
    "heavy_atoms_max": 50,
    "rotatable_bonds_max": 5,
    "mw_max": 300.0,
}


def load_molecules(sdf_path: Path) -> tuple[int, list[Chem.Mol]]:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    total = len(supplier)
    molecules = [mol for mol in supplier if mol is not None]
    return total, molecules


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(array.mean()) if array.size else float("nan"),
        "std": float(array.std()) if array.size else float("nan"),
        "min": float(array.min()) if array.size else float("nan"),
        "median": float(np.median(array)) if array.size else float("nan"),
        "max": float(array.max()) if array.size else float("nan"),
    }


def sanitize_for_json(payload):
    """Replace non-finite floats with ``null`` for strict JSON output."""

    if isinstance(payload, dict):
        return {key: sanitize_for_json(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [sanitize_for_json(value) for value in payload]
    if isinstance(payload, float) and not math.isfinite(payload):
        return None
    return payload


def internal_diversity(molecules: list[Chem.Mol], sample_cap: int, seed: int) -> dict[str, float]:
    """1 - mean pairwise ECFP4 Tanimoto similarity over a random sample."""

    if len(molecules) < 2:
        return {"molecules_used": len(molecules), "mean_similarity": float("nan"), "internal_diversity": float("nan")}
    rng = random.Random(seed)
    sample = molecules if len(molecules) <= sample_cap else rng.sample(molecules, sample_cap)
    fingerprints = [
        GetMorganFingerprintAsBitVect(Chem.RemoveHs(mol), 2, nBits=2048) for mol in sample
    ]
    similarities = []
    for index, fp in enumerate(fingerprints):
        others = fingerprints[:index]
        if not others:
            continue
        similarities.extend(BulkTanimotoSimilarity(fp, others))
    mean_similarity = float(np.mean(similarities))
    return {
        "molecules_used": len(sample),
        "mean_similarity": mean_similarity,
        "internal_diversity": 1.0 - mean_similarity,
    }


def binding_energy_stats(path: Path) -> dict[str, float]:
    values: list[float] = []
    if path.suffix.lower() in {".xlsx", ".xls"}:
        import pandas as pd

        frame = pd.read_excel(path)
        if "binding_energy" not in frame.columns:
            raise ValueError(f"binding_energy column not found in {path}")
        numeric = pd.to_numeric(frame["binding_energy"], errors="coerce").dropna()
        values = [float(value) for value in numeric.tolist()]
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "binding_energy" not in (reader.fieldnames or []):
                raise ValueError(f"binding_energy column not found in {path}")
            for row in reader:
                try:
                    values.append(float(row["binding_energy"]))
                except (TypeError, ValueError):
                    continue

    valid = [value for value in values if value < 0]
    failed = [value for value in values if value == 999.0]
    stats = describe(valid)
    stats.update(
        {
            "rows": len(values),
            "valid_energies": len(valid),
            "failed_sentinel_999": len(failed),
        }
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Fragment-growth run directory")
    parser.add_argument("--output-dir", type=Path, help="Defaults to run_dir")
    parser.add_argument(
        "--diversity-sample",
        type=int,
        default=1000,
        help="Maximum number of molecules used for the diversity estimate",
    )
    parser.add_argument("--diversity-seed", type=int, default=20260807)
    parser.add_argument(
        "--binding-energies",
        type=Path,
        help="Optional CSV/XLSX with a binding_energy column",
    )
    parser.add_argument(
        "--histograms",
        action="store_true",
        help="Also write property histograms (requires matplotlib)",
    )
    args = parser.parse_args()

    if args.diversity_sample < 1:
        parser.error("--diversity-sample must be at least 1")
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or args.run_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = run_dir / "output.sdf"
    if not sdf_path.is_file():
        raise SystemExit(f"output.sdf not found in {run_dir}")

    total, molecules = load_molecules(sdf_path)
    heavy_molecules = [Chem.RemoveHs(mol) for mol in molecules]

    smiles = [Chem.MolToSmiles(mol) for mol in heavy_molecules]
    unique = set(smiles)

    properties = {
        "mw": [Descriptors.MolWt(mol) for mol in heavy_molecules],
        "logp": [Crippen.MolLogP(mol) for mol in heavy_molecules],
        "hba": [Lipinski.NumHAcceptors(mol) for mol in heavy_molecules],
        "hbd": [Lipinski.NumHDonors(mol) for mol in heavy_molecules],
        "rotatable_bonds": [Lipinski.NumRotatableBonds(mol) for mol in heavy_molecules],
        "tpsa": [rdMolDescriptors.CalcTPSA(mol) for mol in heavy_molecules],
        "heavy_atoms": [mol.GetNumHeavyAtoms() for mol in heavy_molecules],
        "sa_score": [sascorer.calculateScore(mol) for mol in heavy_molecules],
    }
    n_violations = [
        sum(
            [
                properties["mw"][i] > 500,
                properties["logp"][i] > 5,
                properties["hba"][i] > 10,
                properties["hbd"][i] > 5,
            ]
        )
        for i in range(len(heavy_molecules))
    ]
    properties["lipinski_violations"] = n_violations

    n_molecules = len(heavy_molecules)
    thresholds_report: dict[str, float] = {}
    if n_molecules:
        thresholds_report = {
            "fraction_logp_le_5": sum(v <= DRUG_LIKE_THRESHOLDS["logp_max"] for v in properties["logp"]) / n_molecules,
            "fraction_sa_score_le_8": sum(v <= DRUG_LIKE_THRESHOLDS["sa_score_max"] for v in properties["sa_score"]) / n_molecules,
            "fraction_heavy_atoms_le_50": sum(v <= DRUG_LIKE_THRESHOLDS["heavy_atoms_max"] for v in properties["heavy_atoms"]) / n_molecules,
            "fraction_rotatable_bonds_le_5": sum(v <= DRUG_LIKE_THRESHOLDS["rotatable_bonds_max"] for v in properties["rotatable_bonds"]) / n_molecules,
            "fraction_mw_le_300": sum(v <= DRUG_LIKE_THRESHOLDS["mw_max"] for v in properties["mw"]) / n_molecules,
        }

    summary = {
        "run_dir": str(run_dir),
        "records_in_sdf": total,
        "rdkit_valid_molecules": n_molecules,
        "validity": n_molecules / total if total else float("nan"),
        "unique_molecules": len(unique),
        "uniqueness": len(unique) / n_molecules if n_molecules else float("nan"),
        "internal_diversity": internal_diversity(molecules, args.diversity_sample, args.diversity_seed),
        "properties": {key: describe(values) for key, values in properties.items()},
        "drug_like_fractions": thresholds_report,
        "thresholds": DRUG_LIKE_THRESHOLDS,
    }
    if args.binding_energies is not None:
        summary["binding_energy_kcal_mol"] = binding_energy_stats(args.binding_energies.resolve())

    (output_dir / "analysis_summary.json").write_text(
        json.dumps(sanitize_for_json(summary), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    molecule_csv = output_dir / "molecule_properties.csv"
    with molecule_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "smiles",
                "mw",
                "logp",
                "hba",
                "hbd",
                "rotatable_bonds",
                "tpsa",
                "heavy_atoms",
                "sa_score",
                "lipinski_violations",
            ]
        )
        for i, mol_smiles in enumerate(smiles):
            writer.writerow(
                [
                    i,
                    mol_smiles,
                    f"{properties['mw'][i]:.4f}",
                    f"{properties['logp'][i]:.4f}",
                    properties["hba"][i],
                    properties["hbd"][i],
                    properties["rotatable_bonds"][i],
                    f"{properties['tpsa'][i]:.4f}",
                    properties["heavy_atoms"][i],
                    f"{properties['sa_score'][i]:.4f}",
                    properties["lipinski_violations"][i],
                ]
            )

    if args.histograms:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise SystemExit(f"--histograms requires matplotlib: {exc}")
        for key, label in (
            ("logp", "logP"),
            ("mw", "Molecular weight"),
            ("rotatable_bonds", "Rotatable bonds"),
            ("heavy_atoms", "Heavy atoms"),
            ("sa_score", "SA score"),
        ):
            values = properties[key]
            if not values:
                continue
            plt.figure(figsize=(6.4, 4.8))
            plt.hist(values, bins=10, edgecolor="black")
            plt.xlabel(label)
            plt.ylabel("Count")
            plt.savefig(output_dir / f"hist_{key}.png", dpi=200, bbox_inches="tight")
            plt.close()

    print(json.dumps(sanitize_for_json(summary), indent=2, ensure_ascii=False, allow_nan=False))
    if not math.isfinite(summary["validity"]) or n_molecules == 0:
        raise SystemExit("No RDKit-readable molecules found in output.sdf")


if __name__ == "__main__":
    main()
