"""Loading of the ligand electron-density grid used by the growth engine."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


def load_ligand_ed(liged_dir: str) -> Optional[np.ndarray]:
    """Load a precomputed ligand ED grid (``ligED.npy``) if present."""

    liged_path = os.path.join(liged_dir, "ligED.npy")
    if os.path.exists(liged_path):
        return np.load(liged_path)
    return None
