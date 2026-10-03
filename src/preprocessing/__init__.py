"""Preprocessing for the GuestGen training data.

The package is import-light: optional scientific packages (``sugar``,
``pyvista`` and ``lmdb``) are imported only by the stages that need them, so
every command can always provide ``--help`` and useful dependency messages.
"""

from .constants import (
    DEFAULT_AUGMENTATION_SEED,
    DEFAULT_LMDB_HIGH_VALUE_POLICY,
    DEFAULT_MULTIWFN_AUG3D,
    DEFAULT_MULTIWFN_GRID_SPACING,
    DEFAULT_NUM_AUG,
    DEFAULT_NUM_POINTS,
    DEFAULT_ROTATION_STEP_SIZE,
    DEFAULT_TARGET_STEP_SIZE,
    DEFAULT_TRANSLATION_STEP_SIZE,
    DEFAULT_XTB_ITERATIONS,
)

__all__ = [
    "DEFAULT_AUGMENTATION_SEED",
    "DEFAULT_LMDB_HIGH_VALUE_POLICY",
    "DEFAULT_MULTIWFN_AUG3D",
    "DEFAULT_MULTIWFN_GRID_SPACING",
    "DEFAULT_NUM_AUG",
    "DEFAULT_NUM_POINTS",
    "DEFAULT_ROTATION_STEP_SIZE",
    "DEFAULT_TARGET_STEP_SIZE",
    "DEFAULT_TRANSLATION_STEP_SIZE",
    "DEFAULT_XTB_ITERATIONS",
]
