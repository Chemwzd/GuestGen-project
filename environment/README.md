# Environments

Two conda environments are required. They must remain separate: the ED-VAE
side uses a recent PyTorch/RDKit stack while the fragment-growth side depends
on PyTorch 1.13 with CUDA 11.7 and older compiled packages.

## ED-VAE, data augmentation and host ED

```bash
conda env create -f environment/environment-edvae.yml
conda activate guestgen
```

Key runtime versions: Python 3.10, PyTorch 2.9.1, NumPy 1.26.4, SciPy 1.13.1,
RDKit 2025.09.1, LMDB 1.7.5, PyVista 0.46.4, supramolecular-sugar 0.6,
SwanLab 0.7.5 (only needed for `--monitor_backend swanlab`).

## Fragment growth

```bash
conda env create -f environment/environment-fragment-growth.yml
conda activate e2m
```

Key runtime versions: Python 3.8, PyTorch 1.13.1 (CUDA 11.7),
torch-geometric 2.3.0, NumPy 1.24.4, SciPy 1.10.1, scikit-learn 1.3.2,
PLIP 2.3.0.

## External programs

| Program | Purpose | Override |
|---|---|---|
| xTB 6.7.1 | molden files for host/guest ED; GFN-FF/GFN2 optimization in binding mode | `XTB_BIN` / `--xtb` |
| Multiwfn 3.8(dev) | electron-density export on the 0.5 Bohr grid | `MULTIWFN_BIN` / `--multiwfn`, `MULTIWFN_SETTINGS` / `--multiwfn-settings` |
| Open Babel 3.1.1 | MOL -> XYZ conversion for training pairs; MOL -> SDF receptor conversion | `OBABEL_BIN` / `--obabel` |

If several programs with the same name are installed (for example a
Schrödinger `xtb` wrapper, which has an incompatible command line), make sure
`XTB_BIN` points at the real xTB binary.

Multiwfn's `settings.ini` is always copied into the calculation directory and
`aug3D` is modified only in that copy. The `supramolecular-sugar` package may
require an external installation channel.

## Python resolution in the pipeline scripts

`scripts/_common.sh` resolves each interpreter in this order:

1. `EDVAE_PYTHON` / `FRAG_PYTHON` environment variables;
2. a conda environment named `guestgen` / `e2m` under `~/anaconda3`,
   `~/miniconda3` or `~/miniforge3`;
3. `python` from `PATH` (activate the intended environment yourself).
