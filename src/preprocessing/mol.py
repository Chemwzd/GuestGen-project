"""MOL helpers used on work copies inside a run directory.

Two centering backends are supported:

``sugar``
    Uses ``supramolecular-sugar`` (``HostMolecule``), the backend used for the
    released dataset.

``native``
    A V2000-only fallback that subtracts the host heavy-atom centroid from both
    molecules of a pair. It is provided for machines without a CSD/sugar
    installation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .common import PreprocessingError


def _v2000_atoms(path: Path) -> Tuple[List[str], int, np.ndarray, List[str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(True)
    if len(lines) < 5:
        raise PreprocessingError(f"Not a valid MOL file (too few lines): {path}")
    if "V3000" in lines[3]:
        raise PreprocessingError(
            f"The native recentering backend supports V2000 only: {path}. "
            "Install `sugar` or convert the work copy to V2000 first."
        )
    try:
        atom_count = int(lines[3][0:3])
    except ValueError as exc:
        raise PreprocessingError(f"Cannot parse V2000 atom count in {path}") from exc
    atom_lines = lines[4 : 4 + atom_count]
    if len(atom_lines) != atom_count:
        raise PreprocessingError(f"Truncated atom block in {path}")

    coordinates = []
    symbols = []
    for line in atom_lines:
        try:
            coordinates.append(
                [float(line[0:10]), float(line[10:20]), float(line[20:30])]
            )
            symbol = line[31:34].strip()
            if not symbol:
                symbol = line.split()[3]
        except (ValueError, IndexError) as exc:
            raise PreprocessingError(
                f"Cannot parse V2000 atom line in {path}: {line!r}"
            ) from exc
        symbols.append(symbol)
    return lines, atom_count, np.asarray(coordinates, dtype=float), symbols


def _render_shifted_v2000(
    lines: Sequence[str], atom_count: int, coordinates: np.ndarray, shift: Sequence[float]
) -> str:
    """Return the MOL text with the atom block shifted by ``shift``."""

    shifted = coordinates + np.asarray(shift, dtype=float)
    output = list(lines)
    for index in range(atom_count):
        original = output[4 + index]
        suffix = original[30:] if len(original) >= 30 else "\n"
        output[4 + index] = (
            f"{shifted[index, 0]:10.4f}{shifted[index, 1]:10.4f}"
            f"{shifted[index, 2]:10.4f}{suffix}"
        )
    return "".join(output)


def _replace_file(path: Path, text: str) -> None:
    """Atomically replace ``path`` with ``text`` (write-then-rename)."""

    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def recenter_pair_native(host_path: Path, guest_path: Path) -> None:
    """Subtract the host heavy-atom centroid from both work-copy MOL files.

    Both files are parsed before either one is written, so a malformed pair
    cannot leave the two files in different coordinate frames.
    """

    host_lines, host_count, host_coordinates, host_symbols = _v2000_atoms(host_path)
    guest_lines, guest_count, guest_coordinates, _ = _v2000_atoms(guest_path)
    heavy_indices = [
        i for i, symbol in enumerate(host_symbols) if symbol.upper() != "H"
    ]
    if not heavy_indices:
        raise PreprocessingError(f"Host contains no non-hydrogen atoms: {host_path}")
    host_centroid = host_coordinates[heavy_indices].mean(axis=0)
    shift = -host_centroid
    _replace_file(
        host_path, _render_shifted_v2000(host_lines, host_count, host_coordinates, shift)
    )
    _replace_file(
        guest_path, _render_shifted_v2000(guest_lines, guest_count, guest_coordinates, shift)
    )


def recenter_single_native(mol_path: Path) -> None:
    """Center one host work-copy on its heavy-atom centroid."""

    lines, atom_count, coordinates, symbols = _v2000_atoms(mol_path)
    heavy_indices = [i for i, symbol in enumerate(symbols) if symbol.upper() != "H"]
    if not heavy_indices:
        raise PreprocessingError(f"Molecule contains no non-hydrogen atoms: {mol_path}")
    _replace_file(
        mol_path,
        _render_shifted_v2000(
            lines, atom_count, coordinates, -coordinates[heavy_indices].mean(axis=0)
        ),
    )


def recenter_pair_sugar(host_path: Path, guest_path: Path) -> None:
    """Center a host/guest pair with the `sugar` backend.

    The host heavy-atom centroid is moved to the origin and the guest keeps its
    relative displacement, matching the centering of the released dataset.
    Both files are rewritten through temporary copies and only replaced after
    both conversions succeeded.
    """

    try:
        from sugar.molecule import HostMolecule
    except ImportError as exc:
        raise PreprocessingError(
            "The `sugar` package is unavailable. Activate the ED-VAE environment "
            "(guestgen), or use --recenter-backend native (V2000 fallback)."
        ) from exc

    sugar_host = HostMolecule.init_from_mol_file(str(host_path))
    sugar_guest = HostMolecule.init_from_mol_file(str(guest_path))
    host_centroid = sugar_host.get_centroid_remove_h()
    guest_centroid = sugar_guest.get_centroid_remove_h()
    guest_relative_centroid = guest_centroid - host_centroid
    moved_host = sugar_host.translate_to_new_origin([0.0, 0.0, 0.0])
    moved_guest = sugar_guest.translate_to_new_origin(guest_relative_centroid)

    temporary_host = host_path.with_name(f"{host_path.stem}.tmp{host_path.suffix}")
    temporary_guest = guest_path.with_name(f"{guest_path.stem}.tmp{guest_path.suffix}")
    moved_host.write_to_mol_file(
        output_path=str(host_path.parent), file_name=temporary_host.name, convert=True
    )
    moved_guest.write_to_mol_file(
        output_path=str(guest_path.parent), file_name=temporary_guest.name, convert=True
    )
    os.replace(temporary_host, host_path)
    os.replace(temporary_guest, guest_path)


def recenter_single_sugar(mol_path: Path) -> None:
    try:
        from sugar.molecule import HostMolecule
    except ImportError as exc:
        raise PreprocessingError(
            "The `sugar` package is unavailable. Activate the ED-VAE environment "
            "(guestgen), or use --recenter-backend native (V2000 fallback)."
        ) from exc
    molecule = HostMolecule.init_from_mol_file(str(mol_path))
    moved = molecule.translate_to_new_origin([0.0, 0.0, 0.0])
    temporary = mol_path.with_name(f"{mol_path.stem}.tmp{mol_path.suffix}")
    moved.write_to_mol_file(
        output_path=str(mol_path.parent), file_name=temporary.name, convert=True
    )
    os.replace(temporary, mol_path)


def choose_recenter_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import sugar.molecule  # noqa: F401
    except ImportError:
        return "native"
    return "sugar"
