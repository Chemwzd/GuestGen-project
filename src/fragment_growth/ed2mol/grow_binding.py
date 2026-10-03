"""ED-guided fragment growth with xTB binding-energy screening.

The growth loop mirrors :mod:`ed2mol.grow`; after each iteration the
candidates are ranked by a GFN2-xTB host-guest binding energy
``(E_complex - E_frag1 - E_frag2) * 627.510`` kcal/mol. Candidates whose
optimization fails, or whose optimized complex does not remain exactly two
fragments, are recorded as the sentinel value ``999.0`` and sorted last.
"""

from __future__ import annotations

import json
import os
import random
import shlex
import shutil
import subprocess
from multiprocessing import Pool
from traceback import print_exc
from typing import List, Tuple

import numpy as np
import torch as th
from rdkit import Chem
from rdkit.Chem import Mol, RWMol, rdmolops
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
    MarkAxialBonds,
    MaxIteration,
    MolBucket,
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


def run_xtb(
    xtb_path,
    structure_file,
    calculation_dir=None,
    opt=False,
    level="normal",
    method="gfn2",
    solvent=None,
    charge=None,
    outjson=True,
    other_keywords=None,
):
    """Run xTB on ``structure_file`` inside ``calculation_dir``.

    Returns the captured ``(stdout, stderr)`` text.
    """

    if level not in ["crude", "sloppy", "loose", "normal", "tight", "verytight"]:
        raise ValueError(
            "level must be one of ['crude', 'sloppy', 'loose', 'normal', 'tight', 'verytight']"
        )
    if method not in ["gfn1", "gfn2", "gfnff"]:
        raise ValueError("method must be one of ['gfn1', 'gfn2', 'gfnff']")

    xtb_method_flags = {
        "gfn1": ["--gfn", "1"],
        "gfn2": ["--gfn", "2"],
        "gfnff": ["--gfnff"],
    }

    calculation_dir = calculation_dir or os.path.dirname(os.path.abspath(structure_file))
    os.makedirs(calculation_dir, exist_ok=True)
    file_basename = os.path.basename(structure_file)
    destination = os.path.join(calculation_dir, file_basename)
    if not os.path.exists(destination):
        shutil.copyfile(structure_file, destination)

    command = [str(xtb_path), file_basename]
    if opt:
        command += ["--opt", level]
    command += xtb_method_flags[method]
    if solvent:
        command += ["--alpb", solvent]
    if charge:
        command += ["--chrg", str(charge)]
    if outjson:
        command += ["--json"]
    if other_keywords:
        command += shlex.split(other_keywords)

    process = subprocess.run(
        command,
        cwd=calculation_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return process.stdout, process.stderr


def concat_molecules(mol1, mol2):
    concat_mol = rdmolops.CombineMols(mol1, mol2)
    return concat_mol


def get_disconnected_molecules(mol):
    fragments = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    return fragments


def cal_one_binding(
    host_mol: Chem.Mol,
    guest_mol: Chem.Mol,
    cal_dir: str,
    xtb_path: str,
    method: str = "gfnff",
    level: str = "normal",
    preopt=True,
    solvent=None,
) -> float:
    """Compute the host-guest binding energy in kcal/mol.

    Returns the sentinel ``999.0`` when any xTB step fails or when the
    optimized complex is not exactly two fragments.
    """

    os.makedirs(cal_dir, exist_ok=True)
    complex_mol = concat_molecules(host_mol, guest_mol)
    complex_mol_file = os.path.join(cal_dir, "complex.mol")
    Chem.MolToMolFile(complex_mol, complex_mol_file)

    if method == "gfn2" and preopt:
        try:
            run_xtb(
                xtb_path,
                complex_mol_file,
                calculation_dir=cal_dir,
                opt=True,
                level="loose",
                method="gfnff",
                solvent=None,
                charge=None,
                outjson=False,
                other_keywords=None,
            )
            pre_opt_mol_file = os.path.join(cal_dir, "xtb_preopt.mol")
            # The GFN-FF optimization writes xtbopt.mol.
            shutil.move(os.path.join(cal_dir, "xtbopt.mol"), pre_opt_mol_file)
            os.remove(os.path.join(cal_dir, "gfnff_topo"))
            os.remove(os.path.join(cal_dir, "xtbopt.log"))
        except Exception:
            return 999.00

        run_xtb(
            xtb_path,
            os.path.join(cal_dir, "xtb_preopt.mol"),
            calculation_dir=cal_dir,
            opt=True,
            level=level,
            method=method,
            solvent=solvent,
            charge=None,
            outjson=False,
            other_keywords=None,
        )
    elif method == "gfnff":
        run_xtb(
            xtb_path,
            complex_mol_file,
            calculation_dir=cal_dir,
            opt=True,
            level=level,
            method=method,
            solvent=None,
            charge=None,
            outjson=False,
            other_keywords=None,
        )
    else:
        run_xtb(
            xtb_path,
            complex_mol_file,
            calculation_dir=cal_dir,
            opt=True,
            level=level,
            method=method,
            solvent=solvent,
            charge=None,
            outjson=False,
            other_keywords=None,
        )

    binding_energy = 999.00
    opt_mol_file = os.path.join(cal_dir, "xtbopt.mol")

    # The optimized complex must keep exactly two fragments (host + guest);
    # GFN2 bond perception may merge or split them.
    if method == "gfn2":
        opto_mol_file = os.path.join(cal_dir, "xtbtopo.mol")

        if not os.path.exists(opto_mol_file):
            return binding_energy

        xtbtopo_mol = Chem.MolFromMolFile(opto_mol_file, removeHs=False, sanitize=False)

        if xtbtopo_mol is None:
            return binding_energy

        opto_frags = Chem.GetMolFrags(xtbtopo_mol)

        if len(opto_frags) != 2:
            print(f"{cal_dir.split('/')[-1]}: num_frags: {len(opto_frags)}")
            return binding_energy

    if not os.path.exists(opt_mol_file):
        return binding_energy

    if method == "gfnff":
        os.remove(os.path.join(cal_dir, "gfnff_topo"))
    os.remove(os.path.join(cal_dir, "xtbopt.log"))
    try:
        opt_mol = Chem.MolFromMolFile(opt_mol_file, removeHs=False)
        frags = get_disconnected_molecules(opt_mol)
        frag1_mol_file = os.path.join(cal_dir, "frag1.mol")
        Chem.MolToMolFile(frags[0], frag1_mol_file)
        frag2_mol_file = os.path.join(cal_dir, "frag2.mol")
        Chem.MolToMolFile(frags[1], frag2_mol_file)

        for i, m_file in enumerate([opt_mol_file, frag1_mol_file, frag2_mol_file]):
            frag_cal_dir = f"{cal_dir}/{i}"
            os.makedirs(frag_cal_dir, exist_ok=True)
            run_xtb(
                xtb_path,
                m_file,
                calculation_dir=frag_cal_dir,
                opt=False,
                level=level,
                method=method,
                solvent=None,
                charge=None,
                outjson=True,
                other_keywords=None,
            )
            if method == "gfnff":
                os.remove(os.path.join(frag_cal_dir, "gfnff_topo"))

        if method == "gfnff":
            json_file = "gfnff_lists.json"
        elif method == "gfn2":
            json_file = "xtbout.json"
        else:
            json_file = None

        with open(os.path.join(cal_dir, "0", json_file), "r") as f:
            complex_sp_out = json.load(f)
        with open(os.path.join(cal_dir, "1", json_file), "r") as f:
            frag1_sp_out = json.load(f)
        with open(os.path.join(cal_dir, "2", json_file), "r") as f:
            frag2_sp_out = json.load(f)
        binding_energy = (
            complex_sp_out["total energy"]
            - frag1_sp_out["total energy"]
            - frag2_sp_out["total energy"]
        )
        return binding_energy * 627.510

    except Exception:
        return 999.00


def replace_dummy_atoms(mol: Chem.Mol) -> Tuple[Mol, RWMol]:
    """Replace dummy atoms (``R``, ``*``, atomic number 0) with hydrogens."""

    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0 or atom.GetSymbol() in ("R", "*"):
            atom.SetAtomicNum(1)
            atom.SetFormalCharge(0)

    Chem.SanitizeMol(rw, catchErrors=True)
    return mol, rw


def _worker_cal_binding(args) -> Tuple[int, float]:
    """Pool worker: run one host-guest binding-energy calculation."""

    (
        idx,
        host_mol,
        guest_mol,
        cal_dir,
        xtb_path,
        method,
        level,
        preopt,
        solvent,
    ) = args

    os.makedirs(cal_dir, exist_ok=True)
    be = cal_one_binding(host_mol, guest_mol, cal_dir, xtb_path, method, level, preopt, solvent)

    return idx, be


def eval_binding_mp(
    host_mol: Chem.Mol,
    mols: List[Chem.Mol],
    dummy_mols: List[Chem.Mol],
    cal_dir: str,
    xtb_path: str,
    method: str = "gfnff",
    level: str = "normal",
    preopt: bool = True,
    solvent: str = None,
    n_procs: int = 8,
) -> List[tuple[float, Mol]]:
    """Compute binding energies for a batch of guests in parallel.

    Returns ``[(binding_energy, mol), ...]`` in the input order.
    """

    if not mols:
        return []

    os.makedirs(cal_dir, exist_ok=True)

    jobs = []
    for i, mol in enumerate(mols):
        job_dir = os.path.join(cal_dir, f"{i}")
        jobs.append((i, host_mol, mol, job_dir, xtb_path, method, level, preopt, solvent))

    with Pool(processes=max(1, min(int(n_procs), len(jobs)))) as pool:
        results = pool.map(_worker_cal_binding, jobs)

    results.sort(key=lambda x: x[0])

    binding_energies: List[tuple[float, Mol]] = []
    for idx, be in results:
        binding_energies.append((be, dummy_mols[idx]))

    return binding_energies


def run_generation_binding(ed_grid: np.ndarray, configs) -> None:
    """Main growth loop with binding-energy screening (see module docstring)."""

    initial_cores, frags = load_fragment_libraries(configs)
    gppm, tapm = load_models(configs)
    scorer = Scorer(configs)

    ref_core = configs.get("sample", "reference_core")
    tolerance = (
        float(configs.get("sample", "tolerence"))
        if configs.has_option("sample", "tolerence")
        else 0.0
    )
    use_binding_energy = configs.has_option(
        "sample", "use_binding_energy"
    ) and configs.getboolean("sample", "use_binding_energy")
    print(f"Using binding energy: {use_binding_energy}")
    binding_cal_dir = configs.get("sample", "binding_cal_dir")
    host_mol_file = configs.get("sample", "receptor")
    xtb_path = configs.get("sample", "xtb_path")
    xtb_method = configs.get("sample", "xtb_method")
    xtb_opt_level = configs.get("sample", "xtb_opt_level")
    preopt = configs.get("sample", "preopt")
    solvent = configs.get("sample", "solvent")
    binding_workers = (
        int(configs.get("sample", "binding_workers"))
        if configs.has_option("sample", "binding_workers")
        else 8
    )

    if use_binding_energy:
        print(f"Using xTB with {xtb_method} and opt level {xtb_opt_level}")
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

            if nogpcores:
                nogp_keep = bucket.add(nogpcores)
                scored_mols.extend(nogp_keep)

            if gpcores:
                gp_keep = bucket.add(gpcores)
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
        print(
            f"length of grow_mols: {len(grow_mols)}, keep: {keep}, "
            f"use_binding_energy: {use_binding_energy}"
        )

        if use_binding_energy:
            print("Starting Binding Energy...")
            host_mol = Chem.MolFromMolFile(host_mol_file, removeHs=False)
            if grow_mols:
                replaced_mols = [replace_dummy_atoms(mol) for mol in grow_mols]
                scored = eval_binding_mp(
                    host_mol,
                    [i[1] for i in replaced_mols],
                    [i[0] for i in replaced_mols],
                    os.path.join(binding_cal_dir, f"{step}_grow_mols"),
                    xtb_path,
                    xtb_method,
                    xtb_opt_level,
                    preopt,
                    solvent,
                    binding_workers,
                )
                # Lower binding energy is better.
                scored.sort(key=lambda x: x[0])
                grow_mols = [m for e, m in scored]

            if candidate_mols:
                replaced_mols = [replace_dummy_atoms(mol) for mol in candidate_mols]
                scored_cand = eval_binding_mp(
                    host_mol,
                    [i[1] for i in replaced_mols],
                    [i[0] for i in replaced_mols],
                    os.path.join(binding_cal_dir, f"{step}_cand_mols"),
                    xtb_path,
                    xtb_method,
                    xtb_opt_level,
                    preopt,
                    solvent,
                    binding_workers,
                )
                scored_cand.sort(key=lambda x: x[0])
                candidate_mols = [m for e, m in scored_cand]

        if len(grow_mols) > keep:
            if use_binding_energy:
                cores = grow_mols[:keep]
            else:
                cores = random.sample(grow_mols, keep)
        else:
            cores = list(grow_mols)
            remaining = keep - len(grow_mols)
            if len(candidate_mols) > remaining:
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
