# Examples

Small, self-contained inputs that exercise every stage of the pipeline. All
example runs write to `runs/examples/` and never modify these inputs.

## Contents

| Path | Description |
|---|---|
| `training_source/experimental/` | 4 host/guest complex pairs (`<base>_1.mol` / `<base>_2.mol`) for the augmentation example |
| `training_source/virtual/` | 2 additional pairs (virtual complexes) |
| `training_lmdb_subset/` | 40-entry LMDB subset of the full training set, used by the training example |
| `training_preprocessed/` | one precomputed `processed_48/<base>_<aug>/property_aligned.pkl` sample for the LMDB-building example |
| `host_mol/` | host MOL files (24 complexes; `bezcoa`, `levpuz`, `qupyik` are the manuscript cases) |
| `host_ed/` | precomputed host ED grids (48^3) for `bezcoa`, `levpuz`, `qupyik` |
| `guest_ed/` | precomputed guest ED grids (48^3 `.npy`) for the same three cases, used as growth inputs |
| `initial_core.sdf` | one small core used to seed the fast growth examples |

Precomputed ED files are provided so that the training/inference/growth
examples can run on machines without xTB/Multiwfn. Re-computing them with
`scripts/03_prepare_host_ed.sh` (or the `ed` stage below) requires those
programs.

## Quick start

```bash
# 1. preprocessing: augmentation + one xTB/Multiwfn ED pair + LMDB build
bash examples/run_examples.sh data

# 2. ED-VAE smoke training on the bundled LMDB subset (5 steps)
bash examples/run_examples.sh train

# 3. guest ED inference with the released checkpoint (seed 42)
bash examples/run_examples.sh infer

# 4. fragment growth, without binding screening
bash examples/run_examples.sh grow

# 5. fragment growth, with GFN2-xTB binding screening (requires xTB)
bash examples/run_examples.sh grow-binding

# 6. summarize both growth runs
bash examples/run_examples.sh analyze

# or run all stages in order
bash examples/run_examples.sh all
```

`TARGET=levpuz` (or `qupyik`) switches the case for the growth stages.

## Runtimes (reference: single workstation, one RTX 4090)

| Stage | Cost |
|---|---|
| data (augmentation) | seconds |
| data (one ED pair: 2 xTB + 2 Multiwfn runs) | a few minutes |
| data (LMDB build) | seconds |
| train (5 steps) | under a minute on GPU |
| infer (3 cases) | seconds |
| grow (2 rounds, small pools) | a few minutes |
| grow-binding (2 rounds, 2 workers) | minutes to ~1 h depending on xTB speed |
| analyze | seconds |

Full-scale runs use the same scripts with the full dataset and protocol
parameters; see the main README.
