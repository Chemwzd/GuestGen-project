"""Graph featurization and model-driven growth-point / torsion prediction."""

from __future__ import annotations

import copy
from collections import defaultdict
from math import pi

import numpy as np
import torch as th
from rdkit import Chem, RDLogger
from torch_geometric import loader
from torch_geometric.data import Data, Dataset

from .Utils import CheckAtomCol, MolOpt, RotateFragment

RDLogger.DisableLog("rdApp.*")


class Mol2GraphGrowthPoint:
    def __init__(self, mol, edval, tree):
        self.mol = mol
        self.env = self.calc_ed_env(mol, tree, edval)

    def _onehotencoding(self, lst, i):
        return list(map(lambda x: 1 if x == i else 0, lst))

    def _atom_featurization(self):
        atom_vector = []
        conformer = self.mol.GetConformer()
        positions = conformer.GetPositions()
        for idx, a in enumerate(self.mol.GetAtoms()):
            atomtype_encoding = self.env[idx]
            atom_vector.append(atomtype_encoding)
        atom_coords = th.from_numpy(positions)
        atom_vector = th.tensor(atom_vector, dtype=th.float)
        return atom_coords, atom_vector

    def _bond_featurization(self):
        bond_vector = []
        for b in self.mol.GetBonds():
            b1 = [b.GetBeginAtomIdx()]
            b2 = [b.GetEndAtomIdx()]
            bondtype_encoding = self._onehotencoding(
                [1.0, 2.0, 3.0, 1.5, 0.0], b.GetBondTypeAsDouble()
            )
            bond_is_in_ring = [int(b.IsInRing())]
            bond_is_conjugated = [int(b.GetIsConjugated())]
            bond_vector.append(b1 + b2 + bondtype_encoding + bond_is_in_ring + bond_is_conjugated)
            bond_vector.append(b2 + b1 + bondtype_encoding + bond_is_in_ring + bond_is_conjugated)
        return th.tensor(bond_vector, dtype=th.long)

    def _atom_label(self):
        label_vector = []
        for idx, a in enumerate(self.mol.GetAtoms()):
            if a.GetSymbol() in ["R", "*"]:
                label_vector.append([1])
            elif a.GetSymbol() in ["H"]:
                label_vector.append([0])
            else:
                label_vector.append([2])
        return th.tensor(label_vector, dtype=th.long)

    def _create_graph(self):
        bond_vector = self._bond_featurization()
        atom_coords, atom_vector = self._atom_featurization()
        edge_index, edge_attr = bond_vector[:, :2], bond_vector[:, 2:]
        edge_index = edge_index.permute(1, 0)
        atom_vector = atom_vector.float()
        edge_attr = edge_attr.float()
        atom_label = self._atom_label()
        graph = Data(
            x=atom_vector,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=atom_coords,
            y=atom_label,
        )
        return graph

    def calc_ed_env(self, mol, tree, edval):
        """10-dimensional density-environment feature per atom.

        For every atom, the fraction of grid points within 1.5, 2.0, 2.5, 3.0
        Angstrom that lie inside (``acc_ratio``) or outside (``ex_ratio``) the
        density region is recorded.
        """

        ha_coors = []
        at_coors = {}
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            pos = mol.GetConformer().GetAtomPosition(i)
            if atom.GetSymbol() not in ["R", "*", "H"]:
                ha_coors.append([pos.x, pos.y, pos.z])
                at_coors[i] = [pos.x, pos.y, pos.z]
            else:
                at_coors[i] = [pos.x, pos.y, pos.z]

        if at_coors:
            ind1 = tree.query_ball_point(ha_coors, 1.5)
            atomexgrid = []
            for i in ind1:
                atomexgrid.extend(i)
            atomexgrid = np.array(list(set(atomexgrid)))
            atomexgrid = atomexgrid[edval[atomexgrid] > 0]

            env = defaultdict(list)
            for idx in at_coors:
                for r in np.arange(1, 3.5, 0.5):
                    try:
                        ind = tree.query_ball_point(at_coors[idx], r)
                        ind = np.array(ind)
                        ingrid = ind[edval[ind] > 0]
                        ex_ratio = np.intersect1d(atomexgrid, ingrid).size / ind.size
                        in_ratio = ingrid.size / ind.size
                        acc_ratio = in_ratio - ex_ratio
                        env[idx].extend([ex_ratio, acc_ratio])
                    except Exception:
                        env[idx].extend([0, 0])
        else:
            raise AssertionError("No attachment point")
        return env


class DatasetGrowthPoint(Dataset):
    def __init__(self, data_list, edval, tree):
        super(DatasetGrowthPoint, self).__init__()
        self.data_list = data_list
        self.edval = edval
        self.tree = tree

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        mol = Chem.AddHs(self.data_list[idx], addCoords=True)
        core2graph = Mol2GraphGrowthPoint(mol, self.edval, self.tree)
        return core2graph._create_graph()

    def __len__(self):
        return self.len()

    def __getitem__(self, idx):
        return self.get(idx)


def GrowthPointFilter(configs, cores, model, cores_dataset, step=0, growth_point_pos=None):
    """Classify candidate cores by their predicted growth point."""

    batch_size = int(configs.get("sample", "batch_size"))
    device_type = configs.get("sample", "device_type")
    trainloader = loader.DataLoader(
        cores_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        prefetch_factor=8,
    )
    device = th.device(device_type)
    model = model.to(device)
    model.eval()
    pass_cores_filter = []
    unpass_cores_filter = []

    if (not step) and (growth_point_pos is not None):
        # Lead optimization: keep one ligand core and mark the atom closest to
        # the requested growth position.
        mol = copy.deepcopy(cores[0])
        mol = Chem.AddHs(mol, addCoords=True)
        growth_point_index = 0
        min_dist = 999
        for idx, atom in enumerate(mol.GetAtoms()):
            if atom.GetSymbol() in ["R", "*", "H"]:
                atom_pos = mol.GetConformer().GetAtomPosition(idx)
                dist = np.linalg.norm(np.array(atom_pos) - np.array(growth_point_pos))
                if dist < min_dist:
                    min_dist = dist
                    growth_point_index = idx

        print(f"the min distance between the growth point and the atom is {min_dist}")
        try:
            mol.GetAtoms()[growth_point_index].SetAtomicNum(0)
            pass_cores_filter.append(mol)
        except Exception:
            unpass_cores_filter.append(mol)
            print("Error: growth point is not in the molecule")

    else:
        with th.no_grad():
            for batch, mols in zip(
                trainloader,
                [cores[i : i + batch_size] for i in range(0, len(cores), batch_size)],
            ):
                batch = batch.to(device)
                outputs = model(batch.edge_index, batch.x, batch.pos, batch.edge_attr)
                probabilities, predicted_labels = outputs.cpu().max(dim=1)
                initial_index = 0
                for _mol in mols:
                    mol = copy.deepcopy(_mol)
                    mol = Chem.AddHs(mol, addCoords=True)
                    best_idx = None
                    best_score = None

                    for idx, atom in enumerate(mol.GetAtoms()):
                        atom_idx = initial_index + idx
                        if atom.GetSymbol() in ["R", "*", "H"]:
                            if (
                                predicted_labels[atom_idx].item() == 1
                                and atom.GetIsotope() != 2
                            ):
                                score = probabilities[atom_idx].item()
                                if (best_score is None) or (score > best_score):
                                    best_score = score
                                    best_idx = idx

                    try:
                        # No candidate -> unpass. A candidate whose best score
                        # is <= 0.5 is discarded entirely, as in the original.
                        if best_score is None:
                            raise IndexError
                        if best_score > 0.5:
                            growth_point_index = best_idx
                            mol.GetAtoms()[growth_point_index].SetAtomicNum(0)
                            pass_cores_filter.append(mol)
                    except Exception:
                        unpass_cores_filter.append(mol)

                    initial_index += idx + 1

    return pass_cores_filter, unpass_cores_filter


class Mol2GraphTorsionAngle:
    def __init__(self, mol, c):
        self.mol = mol
        self.c = c

    def _onehotencoding(self, lst, i):
        return list(map(lambda x: 1 if x == i else 0, lst))

    def _atom_featurization(self):
        atom_vector = []
        conformer = self.mol.GetConformer()
        positions = conformer.GetPositions()
        for idx, a in enumerate(self.mol.GetAtoms()):
            atomtype_encoding = self._onehotencoding(
                ["C", "N", "O", "S", "F", "R"], a.GetSymbol()
            )
            atomdegree_encoding = self._onehotencoding([1, 2, 3, 4], a.GetDegree())
            atomhybrid_encoding = self._onehotencoding(
                ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "UNSPECIFIED", "OTHER"],
                str(a.GetHybridization()),
            )
            atomcharge_encoding = self._onehotencoding(
                [-3, -2, -1, 0, 1, 2, 3], a.GetFormalCharge()
            )
            atomhydrogen_encoding = self._onehotencoding([0, 1, 2, 3, 4], a.GetTotalNumHs())
            atom_is_in_aromatic = [int(a.GetIsAromatic())]
            atom_is_in_ring = [int(a.IsInRing())]
            atom_coord = conformer.GetAtomPosition(idx)
            if self.c is not None:
                center = self.c
                atom_position = list(
                    (
                        np.array(
                            [
                                atom_coord.x - center[0],
                                atom_coord.y - center[1],
                                atom_coord.z - center[2],
                            ]
                        )
                        + np.array([12, 12, 12])
                    )
                    / 24
                )
                atom_vector.append(
                    atomtype_encoding
                    + atomdegree_encoding
                    + atomhybrid_encoding
                    + atomcharge_encoding
                    + atomhydrogen_encoding
                    + atom_is_in_aromatic
                    + atom_is_in_ring
                    + atom_position
                )
            else:
                atom_vector.append(
                    atomtype_encoding
                    + atomdegree_encoding
                    + atomhybrid_encoding
                    + atomcharge_encoding
                    + atomhydrogen_encoding
                    + atom_is_in_aromatic
                    + atom_is_in_ring
                )

        atom_coords = th.from_numpy(positions)
        atom_vector = th.tensor(atom_vector, dtype=th.float)
        return atom_coords, atom_vector

    def _bond_featurization(self):
        bond_vector = []
        for b in self.mol.GetBonds():
            b1 = [b.GetBeginAtomIdx()]
            b2 = [b.GetEndAtomIdx()]
            bondtype_encoding = self._onehotencoding(
                [1.0, 2.0, 3.0, 1.5, 0.0], b.GetBondTypeAsDouble()
            )
            bond_is_in_ring = [int(b.IsInRing())]
            bond_is_conjugated = [int(b.GetIsConjugated())]
            bond_vector.append(b1 + b2 + bondtype_encoding + bond_is_in_ring + bond_is_conjugated)
            bond_vector.append(b2 + b1 + bondtype_encoding + bond_is_in_ring + bond_is_conjugated)
        return th.tensor(bond_vector, dtype=th.long)

    def _create_graph(self):
        bond_vector = self._bond_featurization()
        atom_coords, atom_vector = self._atom_featurization()
        edge_index, edge_attr = bond_vector[:, :2], bond_vector[:, 2:]
        edge_index = edge_index.permute(1, 0)
        atom_vector = atom_vector.float()
        edge_attr = edge_attr.float()
        graph = Data(
            x=atom_vector,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=atom_coords,
        )
        return graph


class DatasetTorsionAngle(Dataset):
    def __init__(self, frags, core, center, ED):
        super(DatasetTorsionAngle, self).__init__()
        self.frags = frags
        self.core = core
        self.center = center
        self.ED = ED

    def len(self):
        return len(self.frags)

    def get(self, idx):
        frag2graph = Mol2GraphTorsionAngle(self.frags[idx], None)
        core2graph = Mol2GraphTorsionAngle(self.core, self.center)
        frag_graph = frag2graph._create_graph()
        core_graph = core2graph._create_graph()
        temp = [core_graph, frag_graph, self.ED]
        return temp

    def __len__(self):
        return self.len()

    def __getitem__(self, idx):
        return self.get(idx)


def Attach(model, frags, core, EDgrid, configs, last_step=False):
    """Attach fragments to a core at the top-k predicted torsion angles."""

    device_type = configs.get("sample", "device_type")
    ED_center = EDgrid[:, :3].mean(0)
    ED_Tensor = th.tensor(EDgrid[:, -1].reshape(-1, 48, 48, 48))
    angles = TorsionAnglePred(
        model,
        frags,
        core,
        ED_center,
        ED_Tensor,
        device_type,
        int(configs.get("sample", "batch_size")),
    )

    outcores = []
    for i, frag in zip(angles, frags):
        for TosionAngle in i:
            attachmol = RotateFragment(core, frag, TosionAngle, 1.5)
            mol = Chem.AddHs(attachmol, addCoords=True)

            mol.SetProp("_Name", mol.GetProp("_Name") + "_" + frag.GetProp("_Name"))
            outcores.append(mol)

    outcores = MolOpt(outcores, configs) if last_step else outcores

    cleancores = [core for core in outcores if CheckAtomCol(core)]
    return cleancores


def TorsionAnglePred(model, frags, core, center, grid, device_type, batch_size, k=3):
    """Top-k torsion angles (radians) predicted for every fragment."""

    device = th.device(device_type)
    dataset = DatasetTorsionAngle(frags, core, center, grid)

    val_loader = loader.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        prefetch_factor=1,
    )

    tensor_list = []
    with th.no_grad():
        for _, batch in enumerate(val_loader):
            edge_index = batch[1].edge_index.to(device)
            x = batch[1].x.to(device)
            pos = batch[1].pos.to(device)
            edge_attr = batch[1].edge_attr.to(device)
            b = batch[1].batch.to(device)
            edge_index1 = batch[0].edge_index.to(device)
            x1 = batch[0].x.to(device)
            coords1 = batch[0].pos.to(device)
            edge_feat1 = batch[0].edge_attr.to(device)
            batch1 = batch[0].batch.to(device)
            grid2 = batch[2].float().to(device)
            val_pred = model(
                edge_index,
                x,
                pos,
                edge_attr,
                b,
                edge_index1,
                x1,
                coords1,
                edge_feat1,
                batch1,
                grid2,
            )
            tensor_list.append(val_pred.cpu())

    new_tensor = th.cat(tensor_list, dim=0)
    values, indices = th.topk(new_tensor, 10, dim=1)
    topk_angle = (np.array(indices) * 10 + 5) / 360 * 2 * pi - pi
    return topk_angle[:, :k]
