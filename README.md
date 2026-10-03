# GuestGen

Code, released model weights and lightweight examples for **electron-density
(ED) guided generation of guest molecules** for host–guest complexes.

The pipeline has two learned/generative stages:

1. **ED-VAE** — a 3D convolutional variational autoencoder that maps a host's
   electron density on a 48³ grid to the guest's electron density on the same
   grid (`host ED -> guest ED`).
2. **ED-guided fragment growth (FragGrow)** — places and grows fragments of a
   guest inside the predicted density, optionally ranking the candidates by a
   GFN2-xTB host–guest binding energy ("with binding" mode).

Five runnable stages are provided end to end:

```text
preprocessing            training              inference            growth
┌─────────────────┐   ┌───────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ augmentation    │   │ ED-VAE        │   │ host ED ->       │   │ without binding: │
│ xTB -> Multiwfn │──▶│ training on   │──▶│ guest ED         │──▶│ density-guided   │──▶ analysis
│ 48³ grids       │   │ the LMDB set  │   │ (released ckpt)  │   │ growth           │
│ LMDB dataset    │   └───────────────┘   └──────────────────┘   │ with binding:    │
└─────────────────┘                                              │ + GFN2 ranking   │
                                                                 └──────────────────┘
```

## Repository layout

```text
GuestGen-project/
├── configs/protocol.json        # reference parameters of the released pipeline
├── environment/                 # conda environments for the two stages
├── examples/                    # small self-contained inputs + run_examples.sh
├── models/checkpoints/          # released ED-VAE checkpoint
├── scripts/                     # pipeline entry points (00–07)
├── src/
│   ├── preprocessing/           # augmentation, xTB/Multiwfn ED, LMDB building
│   ├── edvae/                   # ED-VAE training and guest ED inference
│   └── fragment_growth/         # ED-guided fragment-growth engine (+ weights)
├── tests/                       # unit tests for the preprocessing package
└── tools/                       # dataset inspection, comparison and analysis CLIs
```

## Installation

Model weights (`*.pth`, `*.pt`) are stored with Git LFS: install
[git-lfs](https://git-lfs.com) before cloning, or run `git lfs pull` inside the
clone if `src/fragment_growth/weights/TAPM.pth` is still a pointer file.

Two conda environments are required and must stay separate (see
`environment/README.md` for details):

```bash
conda env create -f environment/environment-edvae.yml          # env: guestgen
conda env create -f environment/environment-fragment-growth.yml # env: e2m
```

The external programs xTB, Multiwfn and Open Babel are needed for the
data-preparation, host-ED and binding-growth steps. Their locations can be
given through environment variables (`XTB_BIN`, `MULTIWFN_BIN`,
`MULTIWFN_SETTINGS`, `OBABEL_BIN`) or the corresponding CLI options. The
pipeline scripts find the two Python environments automatically (see
`scripts/_common.sh`); alternatively set `EDVAE_PYTHON` and `FRAG_PYTHON`.

Verify a machine with:

```bash
bash scripts/00_check_environment.sh
```

## Data

Large assets (the full training LMDB, the augmented MOL files, the host/guest
ED grids) are distributed separately from this repository. Every script takes
the data location as an option or environment variable, so no fixed layout is
required. The formats are:

| Asset | Format | Notes |
|---|---|---|
| Training set | LMDB directory (`data.mdb`) | ASCII integer keys `0..N-1`; each value is a pickled dict with `base_id` (str), `aug_idx` (int), `host_ed` / `guest_ed` (float32, `48×48×48`) |
| Augmented pairs | MOL files `<base>_1_<aug>.mol` / `<base>_2_<aug>.mol` | one directory per augmentation stage |
| Processed ED pairs | `<base>_<aug>/property_aligned.pkl` | pickle dict with `host_aligned_ed` / `guest_aligned_ed` (float64 `48³`); input of the LMDB builder |
| Host ED | `<case>/property_aligned_scale_0_1.pkl` | pickle dict with key `aligned_ed` (float32 `48³`); stored without clipping |
| Guest ED | `<case>.npy` | float32 `48³`; output of the ED-VAE inference |
| Growth input | `ligED.npy` | float32 `(48³, 4)` columns `x y z rho`, masked at `rho ≤ 0.01`, Fortran-order flattening |

The released training set contains 4,937 host/guest complexes (4,085
experimental + 852 virtual), 5 augmentations each (24,685 pairs), and 24,678
LMDB entries (7 pairs rejected by the validation rules). `configs/protocol.json`
registers the reference parameters.

## Quick start: the bundled examples

The `examples/` directory contains small self-contained inputs (4 training
pairs, a 40-entry LMDB subset, three host/guest ED cases and one initial core)
so that every stage can be exercised without the full dataset:

```bash
bash scripts/00_check_environment.sh              # verify environments and assets

bash examples/run_examples.sh data                # augmentation + 1 xTB/Multiwfn pair + LMDB
bash examples/run_examples.sh train               # ED-VAE smoke training (5 steps)
bash examples/run_examples.sh infer               # guest ED inference (seed 42)
bash examples/run_examples.sh grow                # growth without binding screening
bash examples/run_examples.sh grow-binding        # growth with GFN2-xTB screening (needs xTB)
bash examples/run_examples.sh analyze             # validity/diversity/property summary

bash examples/run_examples.sh all                 # all of the above in order
```

See `examples/README.md` for what each example contains and its runtime.

## Pipeline reference

All scripts accept environment variables / options to scale up to the full
dataset; the defaults target the bundled examples.

### 1. Training-data preparation

```bash
bash scripts/01_prepare_training_data.sh augment   # 5x augmentation per pair
bash scripts/01_prepare_training_data.sh ed        # xTB -> Multiwfn -> 48³ interpolation
bash scripts/01_prepare_training_data.sh lmdb      # build the training LMDB
```

Useful variables: `TRAINING_SOURCE_DIR`, `RUN_ROOT`, `PAIR_LIMIT` (default 4),
`NUM_AUG` (5), `ED_PAIR_LIMIT` (1), `LMDB_LIMIT` (32). `DRY_RUN=1` plans the ED
stage without running xTB/Multiwfn. Input structures are always copied into an
isolated work directory first; sources are never modified.

For a full run, point `TRAINING_SOURCE_DIR` at the complete pair collection and
raise the limits; the LMDB stage is then invoked on the resulting
`processed_48` tree.

### 2. ED-VAE training

```bash
bash scripts/02_train_edvae.sh --smoke                          # 5 steps, bundled subset
bash scripts/02_train_edvae.sh --full --data-dir /path/to/lmdb  # full protocol, full set
```

Training parameters (full protocol): max_channels/latent_dim 128/128, dropout
0.3, Adam with lr 3e-4, linear warm-up 200 steps from 3e-5, cosine annealing
with `T_max=3800` and `eta_min=3e-6`, batch size 32, 5,000 steps, KL weight
1e-5, seed 42, validation every 50 steps, log every 10 steps, 90/10 split
grouped by `base_id` (all augmentations of one complex stay on one side).
Checkpoints are written every validation interval as
`guestgen_step_<n>.pt` (`models/checkpoints/guestgen_step_5000.pt` is the
released checkpoint).

`--monitor-backend swanlab` additionally logs to SwanLab (requires an API key).

### 3. Host ED preparation

```bash
bash scripts/03_prepare_host_ed.sh bezcoa                 # one case
bash scripts/03_prepare_host_ed.sh bezcoa levpuz qupyik   # several cases
```

The host MOL is copied to a work directory, recentered (heavy-atom centroid;
`sugar` backend by default, `--recenter-backend native` as a fallback), passed
through xTB (`--molden --iterations 999`) and Multiwfn (density on a 0.5 Bohr
grid), then interpolated onto the 48³ grid at 0.35 Å.

### 4. Guest ED inference

```bash
bash scripts/04_infer_guest_ed.sh bezcoa levpuz qupyik           # random sampling
INFERENCE_SEED=42 bash scripts/04_infer_guest_ed.sh bezcoa       # reproducible
```

Reads host ED pickles from `examples/host_ed` (override with `INPUT_DIR`) and
writes `<case>.npy` plus `inference_manifest.json` (checkpoint hash, seed,
statistics). The VAE samples its latent code, so a run without `--seed` is a
fresh sample each time; a seed with `--deterministic` makes repeated runs
numerically identical on the same hardware.

### 5. Fragment growth

```bash
# without binding-energy screening
bash scripts/05_grow_without_binding.sh bezcoa --execute

# with GFN2-xTB binding-energy screening (requires xTB)
bash scripts/06_grow_with_binding.sh bezcoa --execute
```

Without `--execute` the case is only prepared (inputs, `config.ini`,
`run_manifest.json`) so everything can be inspected first. The scripts forward
any additional `run_case.py` options; the most useful ones are:

| Option | Meaning |
|---|---|
| `--guest-ed PATH` | guest ED `.npy` (default `examples/guest_ed/<target>.npy`) |
| `--host-mol PATH` | host MOL (default `examples/host_mol/<target>.mol`) |
| `--output-dir DIR` | run directory |
| `--initial-core SDF` | seed growth from an SDF instead of the core library (used by the examples) |
| `--iteration N`, `--retain-mols-num N`, `--output-mols-num N` | growth sizes (full protocol: 3 / 80 / 2000) |
| `--bucket-size N`, `--sim-threshold X`, `--seed N` | diversity bucketing and RNG |
| `--num-cpu N`, `--batch-size N`, `--binding-workers N` | resources |
| `--tolerance X`, `--mask X`, `--step-size X` | density masking (full protocol: 0.3 / 0.01 / 0.35) |

Set `EXAMPLE=0` to use the full-protocol defaults of `run_case.py`
(3 iterations, 80 retained, 2,000 output molecules, 32 CPU workers, 8 binding
workers, seed 1234).

Full-protocol growth of the manuscript cases:

```bash
EXAMPLE=0 bash scripts/06_grow_with_binding.sh qupyik \
  --guest-ed /path/to/guest_ed/qupyik.npy --execute
```

Outputs (per run directory): `output.sdf` (generated guests with `qscore` and
`generation number` properties), `ligED.npy`, `<target>.sdf` (receptor),
`config.ini`, `run_manifest.json`, and in binding mode `xtb_tmp/` with the raw
xTB outputs. Binding energy in kcal/mol is
`(E_complex − E_frag1 − E_frag2) × 627.510`; failed candidates are kept as the
sentinel `999.0` and sorted last. The optimized complex must remain exactly two
fragments.

### 6. Result analysis

```bash
bash scripts/07_analyze_growth.sh runs/examples/fragment_growth/without_binding/bezcoa
bash scripts/07_analyze_growth.sh <run_dir> --histograms --binding-energies metrics.xlsx
```

Reports validity, uniqueness, internal diversity (1 − mean pairwise ECFP4
Tanimoto similarity over a random sample), descriptor distributions (MW, logP,
HBD, HBA, rotatable bonds, TPSA, heavy atoms, SA score, Lipinski violations),
drug-like fractions (logP ≤ 5, SA ≤ 8, heavy atoms ≤ 50, rotatable bonds ≤ 5,
MW ≤ 300) and, optionally, binding-energy statistics from a CSV/XLSX file with
a `binding_energy` column. Writes `analysis_summary.json` and
`molecule_properties.csv` into the run directory.

### Auxiliary tools

```bash
# copy a small subset of an LMDB (used for the training example)
python tools/subset_lmdb.py <full_lmdb> <subset_lmdb> --entries 40

# inspect / validate an LMDB (entry count, shapes, base-id scan)
python tools/inspect_lmdb.py <lmdb_dir> --samples 5 --expected-entries 24678 \
    --expected-shape 48,48,48 --expected-dtype float32 --scan-base-ids

# compare two electron-density arrays (shape/finite, MAE, RMSE, Pearson, mask IoU)
python tools/compare_ed.py reference.npy candidate.npy --output compare.json

# validate the outputs of one growth run
python tools/check_growth_output.py <run_dir> --mode binding
```

## Reproduction notes

* **Preprocessing** — augmentation and tools are pinned (xTB 6.7.1, Multiwfn
  3.8, Open Babel 3.1.1). The interpolation values can differ in low-order
  bits across VTK/PyVista versions; validate regenerated grids with
  `tools/compare_ed.py` (shape, Pearson correlation, MAE/RMSE, threshold IoU)
  rather than byte equality.
* **Training** — the split is deterministic given the LMDB and seed 42, and all
  sampling is seeded, but CUDA kernels and cuDNN autotuning are not bitwise
  deterministic. Re-trained models should match the released protocol in loss
  magnitude and curve shape; exact checkpoint hashes are not guaranteed.
* **Inference** — the VAE samples `z ~ N(μ, σ)`. For repeatable inference pass
  `--seed` (optionally `--deterministic`); without a seed every run produces a
  new sample by design.
* **Growth** — Python/NumPy/PyTorch are seeded (default 1234). In binding mode,
  parallel xTB completion order and the numerical optimizers can change which
  individual candidates survive; compare distributions (counts, validity,
  energies) rather than molecule identity.
* **Checksums** (also verified by `scripts/00_check_environment.sh`):

| Asset | SHA-256 |
|---|---|
| `models/checkpoints/guestgen_step_5000.pt` | `2579ce1e4b06ef84701a0c787995ba672c695e5f4bbf3f1726f996bc3310eb81` |
| `src/fragment_growth/weights/GPPM.pth` | `d8d2a31c1005b8d4b003d4cd3572942fb6588d3344d905cee6235abcaf7d7293` |
| `src/fragment_growth/weights/TAPM.pth` | `a1a670e3de8bd8723f6a3d1068a7c1b69c6a6ff84552f7fc3b80364536a03163` |
| `src/fragment_growth/frag_lib/Cores.sdf` | `7412e8c85b63a010ea23070c8f5b713340110b1040353cacd2fcf33a49782fe4` |
| `src/fragment_growth/frag_lib/Frags.sdf` | `d381d1a757642f56026194f117dcef06cade98fec48b116673f26afccae3f573` |

## Tests

```bash
python -m pytest tests -q
```

The test suite covers pair discovery, augmentation isolation (sources are never
written), the recentering fallback, the ED grid protocol, Multiwfn stdin and
settings handling, and the LMDB value rules.

## License and attribution

Released under the MIT License (see `LICENSE`). The ED-VAE architecture follows
the ED generator of [ED2Mol](https://github.com/pineappleK/ED2Mol) (MIT
License); the fragment-growth engine is adapted from the same project and
reorganized as a package. See `THIRD_PARTY_NOTICES.md` for details.
