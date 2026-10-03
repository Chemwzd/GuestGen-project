"""Scoring of candidate molecules against the electron density and the receptor."""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
from rdkit import Chem
from scipy.spatial import cKDTree


class Scorer:
    def __init__(self, configs):
        self.protf = configs.get("sample", "receptor")
        self.query_res = configs.get("sample", "key_res")
        self.tmpdir = os.path.join(configs.get("sample", "output_dir"), "tmp")
        self.num_cpu = int(configs.get("sample", "num_cpu"))

    def QScore(self, mols, density, tol):
        """Filter molecules by density coverage and rank them by QScore.

        Molecules whose heavy atoms land on zero-density voxels for more than
        ``tol`` of their atoms are discarded. The remaining molecules are
        sorted by ``n_atoms * sum(Z * rho) / sum(Z)`` in descending order.
        """

        voxpoints = np.asarray(density[:, :-1])
        rho_all = np.asarray(density[:, -1])
        T = cKDTree(voxpoints)

        def process_mol(mol):
            z = []
            coords = []
            conf = mol.GetConformer()
            for atom in mol.GetAtoms():
                sym = atom.GetSymbol()
                if sym not in ["R", "*", "H"]:
                    z.append(atom.GetAtomicNum())
                    p = conf.GetAtomPosition(atom.GetIdx())
                    coords.append((p.x, p.y, p.z))

            if not coords:
                return None

            z = np.asarray(z, dtype=np.float64)
            coords = np.asarray(coords, dtype=np.float64)

            _, idx = T.query(coords, k=1)
            rho = rho_all[idx]

            if (rho == 0).sum() / rho.size > tol:
                return None

            q_score = coords.shape[0] * (z * rho).sum() / z.sum()
            return mol, q_score

        with ThreadPoolExecutor(max_workers=self.num_cpu) as ex:
            results = [r for r in ex.map(process_mol, mols) if r is not None]

        for mol, q in results:
            mol.SetProp("qscore", str(q))

        results.sort(key=lambda x: x[1], reverse=True)
        scoredmols = [mol for mol, _ in results]
        return scoredmols

    def InteractionMatch(self, compf, query_res, lig_idx):
        """Check whether the ligand forms an interaction with ``query_res``."""

        from plip.structure.preparation import PDBComplex

        mol = PDBComplex()
        mol.load_pdb(compf)
        bsid = "UNL:Z:1"
        mol.analyze()
        interaction = mol.interaction_sets.get(bsid)
        if interaction is None:
            # No binding site under the expected identifier => no match.
            return lig_idx, False

        interaction_res = []
        for i in interaction.all_itypes:
            i_info = i._asdict()
            if i_info["restype"] not in ["LIG", "HOH"]:
                if i_info.get("sidechain", True):
                    portion = "sidechain"
                else:
                    portion = "backbone"
                res = "".join([str(i.reschain), str(i.resnr)])
                interaction_res.append(res + " " + portion)
        interaction_res = list(set(interaction_res))

        if query_res not in interaction_res:
            return lig_idx, False
        return lig_idx, True

    def InteractionScore(self, ligs):
        """Label each molecule with 1/0 depending on the query-residue contact.

        Requires PLIP (interaction detection) to be installed; it is imported
        lazily when this method is called.
        """

        with open(self.protf, "r") as handle:
            reclines = [
                line for line in handle if line[:6] in ["ATOM  ", "HETATM"]
            ]
        lig_hit = []
        plipdir = os.path.join(self.tmpdir, "plip")
        os.makedirs(plipdir, exist_ok=True)

        with ProcessPoolExecutor(max_workers=self.num_cpu) as ex:
            futures = []
            for lig_idx, l in enumerate(ligs):
                lname = str(lig_idx)
                outfname = os.path.join(plipdir, lname + ".pdb")
                l = Chem.RemoveHs(l)
                ligpdb = Chem.PDBWriter(outfname)
                ligpdb.write(l)
                ligpdb.close()
                with open(outfname, "r") as handle:
                    ligatoms = [line for line in handle if line[:6] == "HETATM"]
                with open(outfname, "r") as handle:
                    ligconect = [line for line in handle if line[:6] == "CONECT"]

                with open(outfname, "w") as comp:
                    for _, l in enumerate(ligatoms):
                        comp.writelines(l[:21] + "Z" + l[22:])
                    for _, l in enumerate(reclines):
                        serial = _ + len(ligatoms) + 1
                        comp.writelines(
                            l[:6] + (5 - len(str(serial))) * " " + str(serial) + l[11:]
                        )
                    comp.writelines(ligconect)
                    comp.writelines("END")

                futures.append(
                    ex.submit(self.InteractionMatch, outfname, self.query_res, lig_idx)
                )

            for future in as_completed(futures):
                lig_idx, hit = future.result()
                lig_hit.append([lig_idx, hit])

            mol_plipscore = []
            for idx, hit in lig_hit:
                plipscore = 1 if hit else 0
                l = ligs[idx]
                mol_plipscore.append([l, plipscore])

        shutil.rmtree(plipdir)
        return mol_plipscore
