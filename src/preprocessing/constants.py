"""Shared constants for the GuestGen preprocessing pipeline.

The defaults below are the values used to prepare the released training
dataset: 5x structural augmentation, xTB -> Multiwfn electron-density grids
interpolated to 48^3 points at 0.35 Angstrom, and the LMDB value policy that
maps every density value > 1 to 0.
"""

# Structural augmentation (src/preprocessing/legacy_augment.py).
DEFAULT_NUM_AUG = 5
DEFAULT_ROTATION_STEP_SIZE = 5.0
DEFAULT_TRANSLATION_STEP_SIZE = 20.0
DEFAULT_AUGMENTATION_SEED = 123

# Multiwfn export settings (both lengths are in Bohr inside Multiwfn).
DEFAULT_MULTIWFN_GRID_SPACING = 0.5
DEFAULT_MULTIWFN_AUG3D = 6.0

# Interpolation grid: 48 points per axis, adjacent points 0.35 Angstrom apart,
# every axis spanning [-8.225, +8.225] Angstrom.
DEFAULT_NUM_POINTS = 48
DEFAULT_TARGET_STEP_SIZE = 0.35

# xTB command used to produce the molden file for Multiwfn.
DEFAULT_XTB_ITERATIONS = 999

# The LMDB converter rejects negative or non-finite density values and maps
# every value > 1 to 0 (values above 1 are not clipped to 1).
DEFAULT_LMDB_HIGH_VALUE_POLICY = "zero"
