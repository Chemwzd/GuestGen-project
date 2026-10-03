"""Unit tests for the preprocessing package.

Run with:  python -m pytest tests/ -q
"""

from __future__ import annotations

import hashlib
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from preprocessing.augmentation import discover_pairs, main as augmentation_main  # noqa: E402
from preprocessing.common import PreprocessingError  # noqa: E402
from preprocessing.electron_density import (  # noqa: E402
    clip_01,
    discover_training_pairs,
    multiwfn_stdin,
    patch_multiwfn_settings,
    target_coordinates,
)
from preprocessing.lmdb_builder import load_processed_sample  # noqa: E402
from preprocessing.mol import recenter_pair_native  # noqa: E402


def write_v2000(path: Path, atoms):
    lines = [
        "test\n",
        "  GuestGen\n",
        "\n",
        f"{len(atoms):3d}{0:3d}  0  0  0  0            999 V2000\n",
    ]
    for symbol, x, y, z in atoms:
        lines.append(
            f"{x:10.4f}{y:10.4f}{z:10.4f} {symbol:<3s} 0  0  0  0  0  0  0  0  0  0  0  0\n"
        )
    lines.append("M  END\n")
    path.write_text("".join(lines), encoding="utf-8")


def coordinates(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    count = int(lines[3][:3])
    return np.asarray(
        [
            [float(line[0:10]), float(line[10:20]), float(line[20:30])]
            for line in lines[4 : 4 + count]
        ]
    )


class DiscoveryTests(unittest.TestCase):
    def test_colocated_pair_discovery_is_sorted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for base in ("zeta", "alpha_with_underscore"):
                write_v2000(root / f"{base}_1.mol", [("C", 0, 0, 0)])
                write_v2000(root / f"{base}_2.mol", [("O", 1, 0, 0)])
            pairs = discover_pairs(None, None, root)
            self.assertEqual(
                [pair.base_id for pair in pairs], ["alpha_with_underscore", "zeta"]
            )

    def test_training_pair_parser_preserves_base_underscores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_v2000(root / "WP6_6_1_4.mol", [("C", 0, 0, 0)])
            write_v2000(root / "WP6_6_2_4.mol", [("O", 1, 0, 0)])
            pairs = discover_training_pairs(root)
            self.assertEqual(pairs[0].base_id, "WP6_6")
            self.assertEqual(pairs[0].aug_idx, 4)


class RecenterAndAugmentTests(unittest.TestCase):
    def test_native_pair_centering_uses_host_heavy_atoms(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            host = root / "h.mol"
            guest = root / "g.mol"
            write_v2000(host, [("C", 1, 0, 0), ("O", 3, 0, 0), ("H", 100, 0, 0)])
            write_v2000(guest, [("N", 4, 0, 0)])
            recenter_pair_native(host, guest)
            np.testing.assert_allclose(coordinates(host)[:2, 0], [-1, 1])
            self.assertAlmostEqual(coordinates(guest)[0, 0], 2.0)

    def test_augmentation_copies_source_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            host = source / "case_1.mol"
            guest = source / "case_2.mol"
            write_v2000(host, [("C", 0, 0, 0), ("O", 1, 0, 0)])
            write_v2000(guest, [("N", 2, 0, 0)])
            hashes_before = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (host, guest)
            }
            result = augmentation_main(
                [
                    "--input-dir",
                    str(source),
                    "--output",
                    str(output),
                    "--num-aug",
                    "1",
                    "--recenter-backend",
                    "native",
                ]
            )
            self.assertEqual(result, 0)
            self.assertTrue((output / "molecules" / "case_1_0.mol").is_file())
            self.assertTrue((output / "molecules" / "case_2_0.mol").is_file())
            hashes_after = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (host, guest)
            }
            self.assertEqual(hashes_before, hashes_after)


class ElectronDensityTests(unittest.TestCase):
    def test_target_grid_is_exact_protocol_grid(self):
        grid = target_coordinates(48, 0.35)
        self.assertEqual(grid.shape, (48**3, 3))
        axis = np.unique(grid[:, 0])
        self.assertAlmostEqual(axis[0], -8.225)
        self.assertAlmostEqual(axis[-1], 8.225)
        np.testing.assert_allclose(np.diff(axis), 0.35)

    def test_clip_is_explicit_conventional_clip(self):
        result = clip_01(np.asarray([-1.0, 0.5, 2.0], dtype=np.float32))
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_multiwfn_protocol_and_settings_copy(self):
        self.assertEqual(multiwfn_stdin(0.5), "\n5\n1\n4\n0.5\n3\n\n")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.ini"
            destination = root / "work" / "settings.ini"
            source.write_text("nthreads= 4\naug3D= 2.0\n", encoding="utf-8")
            original = source.read_bytes()
            provenance = patch_multiwfn_settings(source, destination, 6.0)
            self.assertEqual(source.read_bytes(), original)
            self.assertIn("aug3D= 6", destination.read_text(encoding="utf-8"))
            self.assertEqual(provenance["effective_aug3D"], 6.0)


class LmdbPayloadTests(unittest.TestCase):
    def test_high_values_become_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            sample_dir = Path(temporary) / "base_with_under_3"
            sample_dir.mkdir()
            source = sample_dir / "property_aligned.pkl"
            host = np.full((2, 2, 2), 0.5, dtype=np.float64)
            guest = np.full((2, 2, 2), 0.25, dtype=np.float64)
            host[0, 0, 0] = 1.5
            with source.open("wb") as handle:
                pickle.dump(
                    {"host_aligned_ed": host, "guest_aligned_ed": guest}, handle
                )
            payload = load_processed_sample(source, num_points=2)
            self.assertEqual(payload["base_id"], "base_with_under")
            self.assertEqual(payload["aug_idx"], 3)
            self.assertEqual(payload["host_ed"][0, 0, 0], 0.0)
            self.assertEqual(payload["host_ed"].dtype, np.float32)

    def test_negative_density_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            sample_dir = Path(temporary) / "base_0"
            sample_dir.mkdir()
            source = sample_dir / "property_aligned.pkl"
            bad = np.zeros((2, 2, 2), dtype=np.float32)
            bad[0, 0, 0] = -0.1
            with source.open("wb") as handle:
                pickle.dump(
                    {"host_aligned_ed": bad, "guest_aligned_ed": np.zeros_like(bad)},
                    handle,
                )
            with self.assertRaises(PreprocessingError):
                load_processed_sample(source, num_points=2)


class CommandLineTests(unittest.TestCase):
    def test_all_modules_offer_help_without_optional_dependencies(self):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(SRC)
        for module in (
            "preprocessing.augmentation",
            "preprocessing.electron_density",
            "preprocessing.lmdb_builder",
        ):
            completed = subprocess.run(
                [sys.executable, "-m", module, "--help"],
                cwd=str(PROJECT_ROOT),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("usage:", completed.stdout.lower())


if __name__ == "__main__":
    unittest.main()
