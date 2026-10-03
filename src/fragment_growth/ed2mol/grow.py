"""ED-guided fragment growth (no binding-energy screening)."""

from __future__ import annotations

import os
import random
from traceback import print_exc
from typing import List, Tuple

import numpy as np
import torch as th
from rdkit import Chem
from scipy.spatial import KDTree
from tqdm import tqdm

from .models.Mol_Generator import (
    GrowthPointPredictionModel,
    TorsionAnglePredictionModel,
)
from .utils.CorePlacement import CoreId, GetClusterCenters, RetainMaxValues
from .utils.DataProcessing import Attach, DatasetGrowthPoint, GrowthPointFilter
from .utils.Score import Scorer
from .utils.Utils import (
    CheckAtomCol,
    CleanMol,
    GetResCoords,
    MarkAxialBonds,
    MaxIteration,
    MolBucket,
    SortMolByDist,
)


def load_models(
    configs,
) -> Tuple[GrowthPointPredictionModel, TorsionAnglePredictionModel]:
    gppm_path = configs.get("model", "GPPM")
    tapm_path = configs.get("model", "TAPM")
    device_type = configs.get("sample", "device_type")

    device = th.device(device_type)

    gppm = GrowthPointPredictionModel().to(device)
    gppm.load_state_dict(th.load(gppm_path, map_location=th.device(device_type)))
    gppm.eval()

    tapm = TorsionAnglePredictionModel().to(device)
    pretrained_dict = th.load(tapm_path, map_location=th.device(device_type))
    model_dict = tapm.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    tapm.load_state_dict(model_dict)
    tapm.eval()

    return gppm, tapm


def load_fragment_libraries(configs) -> Tuple[List[Chem.Mol], List[Chem.Mol]]:
    cores = Chem.SDMolSupplier(configs.get("lib", "Cores"))
    cores = [mol for mol in cores]
    frags = Chem.SDMolSupplier(configs.get("lib", "Frags"))
    frags = [mol for mol in frags]
    return cores, frags


def run_generation(ed_grid: np.ndarray, configs) -> None:
    """Main growth loop.

    Every iteration attaches fragments to the current population, filters the
    candidates by predicted growth points and the QScore density criterion,
    keeps a diverse subset via molecular-fingerprint bucketing, and retains the
    best molecules for the next iteration.
    """

    initial_cores, frags = load_fragment_libraries(configs)
    gppm, tapm = load_models(configs)
    scorer = Scorer(configs)

    ref_core = configs.get("sample", "reference_core")
    tolerance = (
        float(configs.get("sample", "tolerence"))
        if configs.has_option("sample", "tolerence")
        else 0.0
    )

    try:
        growth_point_pos = [
            float(configs.get("sample", "x")),
            float(configs.get("sample", "y")),
            float(configs.get("sample", "z")),
        ]
    except Exception:
        growth_point_pos = None

    iteration = MaxIteration(ed_grid, configs.get("sample", "iteration"))
    print(f"**********A {iteration}-step Molecule Generation**********")

    generate_molecules: List[Chem.Mol] = []
    cores: List[Chem.Mol] = []

    for step in range(iteration):
        pop_mols: List[List[Chem.Mol]] = []

        if step == 0:
            if not ref_core:
                ed_value = th.tensor(ed_grid)[:, -1].reshape(1, 48, 48, 48)
                max_grid = RetainMaxValues(ed_value)
                merged = np.hstack((ed_grid[:, :-1], max_grid.numpy().reshape(-1, 1)))
                cluster_centers = GetClusterCenters(merged)

                for core in tqdm(initial_cores[:], desc="Placing Initial Fragments"):
                    core_enumeration = CoreId(core, cluster_centers)
                    core_mol_list = scorer.QScore(core_enumeration, ed_grid, tol=tolerance)
                    pop_mols.append(core_mol_list[:1])
            else:
                CheckAtomCol(Chem.SDMolSupplier(ref_core)[0])
                pop_mols = [
                    [c]
                    for c in tqdm(
                        Chem.SDMolSupplier(ref_core), desc="Placing Reference Fragments"
                    )
                    if c
                ]

        else:
            for core in tqdm(cores, desc=f"Growing Fragments--Step {step + 1}"):
                try:
                    cleancores = Attach(
                        tapm, frags, core, ed_grid, configs, last_step=(step == (iteration - 1))
                    )
                    scoredcores = scorer.QScore(cleancores, ed_grid, tol=tolerance)
                    pop_mols.append(scoredcores)
                except Exception:
                    print_exc()

        grow_mols: List[Chem.Mol] = []
        candidate_mols: List[Chem.Mol] = []
        scored_mols: List[Chem.Mol] = []

        sim_threshold = float(configs.get("sample", "sim_threshold")) if step else 1.0
        bucket_size = int(float(configs.get("sample", "bucket_size")))

        kdtree = KDTree(ed_grid[:, :3])
        for core_mol_list in pop_mols:
            core_mol_list = [MarkAxialBonds(CleanMol(mol)) for mol in core_mol_list]
            cores_dataset = DatasetGrowthPoint(core_mol_list, ed_grid[:, -1], kdtree)

            gpcores, nogpcores = GrowthPointFilter(
                configs,
                core_mol_list,
                gppm,
                cores_dataset,
                step,
                growth_point_pos,
            )

            bucket = MolBucket(bucket_size=bucket_size, sim_threshold=sim_threshold)
            interact_res = configs.get("sample", "key_res")

            if nogpcores:
                nogp_keep = bucket.add(nogpcores)
                if interact_res:
                    nogp_keep_scored = scorer.InteractionScore(nogp_keep)
                    nogp_keep = [_c for _c, score in nogp_keep_scored if score == 1]
                scored_mols.extend(nogp_keep)

            if gpcores:
                gp_keep = bucket.add(gpcores)
                if interact_res:
                    gp_scored = scorer.InteractionScore(gp_keep)
                    for _c, score in gp_scored:
                        if score:
                            grow_mols.append(_c)
                        else:
                            candidate_mols.append(_c)
                else:
                    grow_mols.extend(gp_keep)
                scored_mols.extend(gp_keep)

        if step:
            keep = (
                int(configs.get("sample", "retain_mols_num"))
                if configs.has_option("sample", "retain_mols_num")
                else 40
            )
        else:
            keep = (
                int(configs.get("sample", "retain_cores_num"))
                if configs.has_option("sample", "retain_cores_num")
                else 40
            )

        if len(grow_mols) > keep:
            cores = random.sample(grow_mols, keep)
        else:
            cores = list(grow_mols)
            remaining = keep - len(grow_mols)
            if len(candidate_mols) > remaining:
                if interact_res:
                    receptor_path = configs.get("sample", "receptor")
                    query_coords = GetResCoords(interact_res, receptor_path)
                    cores.extend(SortMolByDist(candidate_mols, query_coords)[:remaining])
                else:
                    cores.extend(candidate_mols[:remaining])
            else:
                cores.extend(candidate_mols)

        if step:
            generate_molecules.extend(scored_mols)

    num_molecules = (
        int(configs.get("sample", "output_mols_num"))
        if configs.has_option("sample", "output_mols_num")
        else 1000
    )
    selected = (
        random.sample(generate_molecules, num_molecules)
        if len(generate_molecules) >= num_molecules
        else generate_molecules
    )

    out_path = os.path.join(configs.get("sample", "output_dir"), "output.sdf")
    writer = Chem.SDWriter(out_path)
    for idx, mol in enumerate(selected):
        mol.SetProp("generation number", str(idx))
        writer.write(CleanMol(mol))
    writer.close()

    print("**********Complete Molecule Generation**********")
