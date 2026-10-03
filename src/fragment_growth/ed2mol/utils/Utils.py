"""Utility helpers used across the fragment-growth engine."""

from __future__ import annotations

import copy
import math
import multiprocessing
import os
import random
import configparser
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit.Chem.rdMolTransforms import (
    GetDihedralDeg,
    GetDihedralRad,
    SetBondLength,
    SetDihedralRad,
)
from rdkit.DataStructs import BulkTanimotoSimilarity


def SetSeed(seed: int) -> None:
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def GetConfigs(config_path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    return cfg


def MaxIteration(EDgrid: np.ndarray, assign_iter: Optional[str], vol: int = 1000) -> int:
    """Iteration count: explicit value, otherwise 2-5 scaled by the grid size."""

    if assign_iter:
        return int(assign_iter)
    grid_num = EDgrid[EDgrid[:, 3] > 0].shape[0]
    return min(max(round(grid_num / vol), 2), 5)


def GetAtoms(pdbf: str) -> List[str]:
    atomlines = []
    with open(pdbf, "r") as f:
        for line in f:
            if line[:6] in ["ATOM  ", "HETATM"]:
                atomlines.append(line)
    return atomlines


def GetRatom(mol: Chem.Mol) -> Chem.Atom:
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in ["R", "*"]:
            return atom
    raise ValueError("No attachment atom (R/*) found in molecule")


def BondNeighbors(mol: Chem.Mol, a1: int, a2: int) -> Tuple[List[int], List[int]]:
    """For a dihedral x-a1-a2-y, return the x neighbors of a1 and y of a2."""

    a1n = [n.GetIdx() for n in mol.GetAtomWithIdx(a1).GetNeighbors() if n.GetIdx() != a2]
    a2n = [n.GetIdx() for n in mol.GetAtomWithIdx(a2).GetNeighbors() if n.GetIdx() != a1]
    return a1n, a2n


def CalcAlpha(mol: Chem.Mol, a1: int, a2: int) -> float:
    """Dihedral aggregation used to define the torsion reference angle."""

    a1n, a2n = BondNeighbors(mol, a1, a2)
    s = []
    conf = mol.GetConformer()
    for x in a1n:
        for y in a2n:
            angle = GetDihedralDeg(conf, x, a1, a2, y)
            cosxy = math.cos(math.pi * angle / 180)
            sinxy = math.sin(math.pi * angle / 180)
            s.append([cosxy, sinxy])
    s = 1000000 * np.sum(np.array(s), 0)
    alpha = -math.atan2(s[0], s[1])
    return alpha


def ConnectMols(mol1: Chem.Mol, mol2: Chem.Mol, atom1: Chem.Atom, atom2: Chem.Atom):
    """Connect two molecules at their attachment atoms, removing both dummies.

    Returns the combined molecule and the updated indices of the attachment
    neighbors.
    """

    combined = Chem.CombineMols(mol1, mol2)
    emol = Chem.EditableMol(combined)

    neighbor1_idx = atom1.GetNeighbors()[0].GetIdx()
    neighbor2_idx = atom2.GetNeighbors()[0].GetIdx()
    atom1_idx = atom1.GetIdx()
    atom2_idx = atom2.GetIdx()
    bond_order = atom2.GetBonds()[0].GetBondType()

    emol.AddBond(neighbor1_idx, neighbor2_idx + mol1.GetNumAtoms(), order=bond_order)
    emol.RemoveAtom(atom2_idx + mol1.GetNumAtoms())
    emol.RemoveAtom(atom1_idx)

    mol = emol.GetMol()
    Chem.SanitizeMol(mol)

    if neighbor1_idx > atom1_idx:
        update_nei1_idx = neighbor1_idx - 1
    else:
        update_nei1_idx = neighbor1_idx

    if neighbor2_idx > atom2_idx:
        update_nei2_idx = mol1.GetNumAtoms() - 1 + neighbor2_idx - 1
    else:
        update_nei2_idx = mol1.GetNumAtoms() - 1 + neighbor2_idx

    return mol, update_nei1_idx, update_nei2_idx


def RotateFragment(
    core: Chem.Mol, frag: Chem.Mol, torsion_angle: float, Set_bond_length: float
) -> Chem.Mol:
    """Attach ``frag`` to ``core`` and set the new bond torsion and length."""

    atom1 = GetRatom(core)
    atom2 = GetRatom(frag)

    rdMolAlign.AlignMol(
        frag,
        core,
        atomMap=[
            (atom2.GetIdx(), atom1.GetNeighbors()[0].GetIdx()),
            (atom2.GetNeighbors()[0].GetIdx(), atom1.GetIdx()),
        ],
    )
    newmol, a1, a2 = ConnectMols(core, frag, atom1, atom2)
    Chem.SanitizeMol(newmol)

    conf = newmol.GetConformer()
    x = [n.GetIdx() for n in newmol.GetAtomWithIdx(a1).GetNeighbors() if n.GetIdx() != a2][0]
    y = [n.GetIdx() for n in newmol.GetAtomWithIdx(a2).GetNeighbors() if n.GetIdx() != a1][0]

    angle = -CalcAlpha(newmol, a1, a2) + torsion_angle
    alpha_ori = GetDihedralRad(conf, x, a1, a2, y)
    alpha = angle + alpha_ori

    SetDihedralRad(conf, x, a1, a2, y, alpha)
    SetBondLength(conf, a1, a2, Set_bond_length)
    return newmol


def CleanMol(mol: Chem.Mol) -> Chem.Mol:
    """Add explicit hydrogens and turn dummy atoms into hydrogens."""

    copy_mol = copy.deepcopy(mol)
    copy_mol = Chem.AddHs(copy_mol, addCoords=True)
    for a in copy_mol.GetAtoms():
        if a.GetSymbol() in ["R", "*", "H"]:
            a.SetAtomicNum(1)
    return copy_mol


def CheckAtomCol(mol: Chem.Mol) -> bool:
    """Steric clash check based on scaled sums of van-der-Waals radii."""

    mol = CleanMol(mol)
    dm = Chem.Get3DDistanceMatrix(mol)

    R = []
    n = mol.GetNumAtoms()
    for a1 in mol.GetAtoms():
        rvdw1 = Chem.GetPeriodicTable().GetRvdw(a1.GetAtomicNum())
        for a2 in mol.GetAtoms():
            rvdw2 = Chem.GetPeriodicTable().GetRvdw(a2.GetAtomicNum())
            R.append((rvdw1 + rvdw2) * 0.6)
    R = np.array(R).reshape(n, n)

    rings = mol.GetRingInfo().AtomRings()
    am = Chem.GetAdjacencyMatrix(mol) + np.eye(n, dtype=int)
    for r in rings:
        row = [i[0] for i in permutations(r, 2)]
        col = [i[1] for i in permutations(r, 2)]
        am[row, col] = 1

    coll = dm[np.logical_not(am)] < R[np.logical_not(am)]
    return coll.sum() == 0


def CalculateAngleBetweenVectors(vector1: np.ndarray, vector2: np.ndarray) -> float:
    cosine_angle = np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )
    angle_in_radians = np.arccos(cosine_angle)
    return float(np.degrees(angle_in_radians))


def Minimize(input_sdf: str, configs) -> str:
    """Placeholder for an optional geometry-minimization stage.

    The released protocol performs no minimization here, so the input path is
    returned unchanged.
    """

    return input_sdf


def MolOpt(mols: Sequence[Chem.Mol], configs) -> List[Chem.Mol]:
    """Run the (passthrough) minimization stage over a molecule batch."""

    if not mols:
        return []

    tmpdir = os.path.join(configs.get("sample", "output_dir"), "tmp")
    os.makedirs(tmpdir, exist_ok=True)

    num_cpu = int(configs.get("sample", "num_cpu"))
    num_cpu = max(1, min(num_cpu, len(mols)))

    chunksize = math.ceil(len(mols) / num_cpu)
    mol_chunks = [mols[i : i + chunksize] for i in range(0, len(mols), chunksize)]

    input_sdfs = []
    for index, chunk in enumerate(mol_chunks):
        path = os.path.join(tmpdir, f"{index}.sdf")
        w = Chem.SDWriter(path)
        for mol in chunk:
            w.write(mol)
        w.close()
        input_sdfs.append(path)

    pool = multiprocessing.Pool(processes=num_cpu)
    results = [pool.apply_async(Minimize, args=[p, configs]) for p in input_sdfs]
    pool.close()
    pool.join()

    optimized = []
    for result in results:
        output_sdf = result.get()
        if output_sdf is None:
            continue
        for mol in Chem.SDMolSupplier(output_sdf):
            if mol is not None:
                optimized.append(mol)
        try:
            os.remove(output_sdf)
        except OSError:
            pass

    for input_sdf in input_sdfs:
        try:
            os.remove(input_sdf)
        except OSError:
            continue

    return optimized


def DistFromMol2Coords(mol: Chem.Mol, coord_array: np.ndarray) -> float:
    mol_xyz = mol.GetConformer().GetPositions()
    distances = np.linalg.norm(coord_array - mol_xyz[:, np.newaxis], axis=2)
    min_distances = np.min(distances, axis=0)
    return float(np.min(min_distances))


def SortMolByDist(mols: Sequence[Chem.Mol], coord_array: np.ndarray) -> List[Chem.Mol]:
    all_min_distances = [DistFromMol2Coords(mol, coord_array) for mol in mols]
    sorted_mols = [
        mol for _, mol in sorted(zip(all_min_distances, mols), key=lambda x: -np.max(x[0]))
    ]
    return sorted_mols


def GetResCoords(res: str, recf: str) -> np.ndarray:
    """Extract the coordinates of a receptor residue.

    Parameters
    ----------
    res:
        String like ``"A809 sidechain"`` or ``"A809 backbone"``.
    recf:
        Receptor PDB file.
    """

    chainid = res.split()[0][0]
    resnum = res.split()[0][1:]

    bbcoords: List[List[float]] = []
    sccoords: List[List[float]] = []
    atmlines = GetAtoms(recf)

    for line in atmlines:
        if line[21] == chainid and line[22:26].strip() == resnum:
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            if line[12:16].strip() in ["N", "CA", "C", "O"]:
                bbcoords.append([x, y, z])
            else:
                sccoords.append([x, y, z])

    portion = res.split()[1] if len(res.split()) > 1 else "sidechain"
    coords = bbcoords if portion == "backbone" else sccoords
    return np.array(coords, dtype=float)


def MarkAxialBonds(mol: Chem.Mol) -> Chem.Mol:
    """Mark axial hydrogens of small rings with isotope label 2."""

    TargetSmiles = "C1****C1"
    mol = Chem.AddHs(mol, addCoords=True)
    substructure = Chem.MolFromSmarts(TargetSmiles)
    matches = mol.GetSubstructMatches(substructure)

    Positions = np.array(mol.GetConformer().GetPositions())
    from scipy.linalg import lstsq

    for match in matches:
        RingPositions = Positions[np.array(match).reshape(-1)]
        x, y, z = RingPositions.T
        A = np.column_stack((x, y, np.ones_like(x)))
        coefficients, _, _, _ = lstsq(A, z)
        a, b, _c = coefficients
        NormalVector = np.array([a, b, -1])

        for i in match:
            Atom = mol.GetAtoms()[i]
            AtomPosition = Positions[i]
            angles = []
            for Neighbor in Atom.GetNeighbors():
                if Neighbor.GetAtomicNum() == 1:
                    NeighborPosition = Positions[Neighbor.GetIdx()]
                    BondVector = NeighborPosition - AtomPosition
                    angle = CalculateAngleBetweenVectors(BondVector, NormalVector)
                    angle = abs(angle - 90)
                    angles.append(angle)
                    if angle > 60:
                        Neighbor.SetIsotope(2)

            if len(angles) == 2:
                AxialIndex = int(np.argmax(angles))
                index = 0
                for Neighbor in Atom.GetNeighbors():
                    if Neighbor.GetAtomicNum() == 1:
                        if AxialIndex == index:
                            Neighbor.SetIsotope(2)
                        index += 1

    return copy.deepcopy(mol)


class MolBucket:
    """Keep a diverse subset of molecules using an ECFP Tanimoto threshold."""

    def __init__(self, bucket_size: int, sim_threshold: float):
        self.bucket_size = bucket_size
        self.sim_threshold = sim_threshold

    def _maybe_add(
        self, idx: int, query_mol: Chem.Mol, bucket_fps: Dict[int, object]
    ) -> Dict[int, object]:
        qfp = GetMorganFingerprint(query_mol, 2)
        sim = BulkTanimotoSimilarity(qfp, list(bucket_fps.values()))
        if sorted(sim)[-1] <= self.sim_threshold:
            bucket_fps.update({idx: qfp})
        return bucket_fps

    def add(self, ranked_mols: Sequence[Chem.Mol]) -> List[Chem.Mol]:
        valid_mols = [(idx, mol) for idx, mol in enumerate(ranked_mols) if mol is not None]
        if not valid_mols:
            return []
        first_idx, first_mol = valid_mols[0]
        bucket_fps: Dict[int, object] = {
            first_idx: GetMorganFingerprint(Chem.RemoveHs(first_mol), 2)
        }
        for idx, m in valid_mols:
            m = Chem.RemoveAllHs(m)
            if len(bucket_fps) < self.bucket_size:
                bucket_fps = self._maybe_add(idx, m, bucket_fps)
            else:
                break
        return [ranked_mols[idx] for idx in bucket_fps]
