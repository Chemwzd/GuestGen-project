"""Structural augmentation kernel (frozen).

This module is the augmentation implementation used to prepare the released
training set. It is kept byte-for-byte behaviorally compatible with the
original script: both molecules of a host/guest pair are rotated by the same
random rotation matrix and translated by the same random vector, and the
result is rounded to four decimal places.

Two properties of the kernel are intentional and must be preserved when
re-running the protocol:

* ``random`` and ``numpy`` are re-seeded inside every pair, so each pair uses
  the same sequence of random draws;
* the rotation angle is computed as ``random() * 2 - rotation_step_size``
  (operator precedence as written), which for ``rotation_step_size=5`` yields
  angles in about ``[-5, -3]`` rad.
"""

from __future__ import annotations

import os
import random

import numpy as np


def cal_rotation_matrix(axis, angle):
    """Return the rotation matrix around ``axis`` by ``angle`` (radians)."""

    axis = np.asarray(axis) / np.linalg.norm(axis)
    rot_matrix = np.array(
        [
            [
                axis[0] ** 2 * (1 - np.cos(angle)) + np.cos(angle),
                axis[0] * axis[1] * (1 - np.cos(angle)) - axis[2] * np.sin(angle),
                axis[0] * axis[2] * (1 - np.cos(angle)) + axis[1] * np.sin(angle),
            ],
            [
                axis[0] * axis[1] * (1 - np.cos(angle)) + axis[2] * np.sin(angle),
                axis[1] ** 2 * (1 - np.cos(angle)) + np.cos(angle),
                axis[1] * axis[2] * (1 - np.cos(angle)) - axis[0] * np.sin(angle),
            ],
            [
                axis[0] * axis[2] * (1 - np.cos(angle)) - axis[1] * np.sin(angle),
                axis[1] * axis[2] * (1 - np.cos(angle)) + axis[0] * np.sin(angle),
                axis[2] ** 2 * (1 - np.cos(angle)) + np.cos(angle),
            ],
        ]
    )
    return np.array(rot_matrix).reshape((3, 3))


def rotation_around_axis(position, rot_mat):
    """Rotate positions with a rotation matrix (``rot_mat @ position``)."""

    return np.dot(rot_mat, position)


def translation(position, move):
    """Translate positions by ``move``."""

    position = np.array(position)
    new_position = np.add(position, np.array(move))
    return new_position


def norm_vector(vector):
    return np.array(vector) / np.linalg.norm(vector)


def get_mol_contents(molfile):
    with open(molfile, "r") as f:
        mol_contents = f.readlines()
    return mol_contents


def get_coordinates(mol_contents):
    """Read the coordinates of a MOL file given as a list of lines."""

    num_atoms = mol_contents[3][:3].rstrip()
    num_atoms = int(num_atoms)
    atom_contents = mol_contents[4 : 4 + num_atoms]
    coordinates = [[float(coord) for coord in i.split()[:3]] for i in atom_contents]
    return np.array(coordinates)


def get_aug_coordinates_single(
    mol_contents, num_aug=10, rotation_step_size=2, translation_step_size=1, seed=123
):
    """Augment a single MOL file."""

    random.seed(seed)
    np.random.seed(seed)
    init_coordinates = get_coordinates(mol_contents)
    aug_coordinates = []

    for i in range(num_aug):
        rot_angles = random.random() * 2 - 1 * rotation_step_size
        theta = 2 * np.pi * random.random()
        phi = np.arccos(2 * random.random() - 1)
        rotation_axis = np.array(
            [
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi),
            ]
        )
        rot_mat = cal_rotation_matrix(rotation_axis, rot_angles)

        rand_vector = norm_vector(np.random.rand(3) * 2 - 1)
        translation_vector = rand_vector * translation_step_size
        translation_vector = translation_vector[np.newaxis, :]

        new_positions = rotation_around_axis(init_coordinates.T, rot_mat).T
        aug_coordinates.append(translation(new_positions, translation_vector))
    return np.round(aug_coordinates, 4)


def get_aug_coordinates_double(
    host_mol_contents,
    guest_mol_contents,
    num_aug=10,
    rotation_step_size=2,
    translation_step_size=1,
    seed=123,
):
    """Augment a pair of host/guest MOL files with one shared transform."""

    random.seed(seed)
    np.random.seed(seed)
    init_host_coordinates = get_coordinates(host_mol_contents)
    init_guest_coordinates = get_coordinates(guest_mol_contents)
    aug_host_coordinates = []
    aug_guest_coordinates = []

    for i in range(num_aug):
        rot_angles = random.random() * 2 - 1 * rotation_step_size
        theta = 2 * np.pi * random.random()
        phi = np.arccos(2 * random.random() - 1)
        rotation_axis = np.array(
            [
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi),
            ]
        )
        rot_mat = cal_rotation_matrix(rotation_axis, rot_angles)

        rand_vector = norm_vector(np.random.rand(3) * 2 - 1)
        translation_vector = rand_vector * translation_step_size
        translation_vector = translation_vector[np.newaxis, :]

        new_host_positions = rotation_around_axis(init_host_coordinates.T, rot_mat).T
        new_guest_positions = rotation_around_axis(init_guest_coordinates.T, rot_mat).T
        aug_host_coordinates.append(translation(new_host_positions, translation_vector))
        aug_guest_coordinates.append(
            translation(new_guest_positions, translation_vector)
        )
    return np.round(aug_host_coordinates, 4), np.round(aug_guest_coordinates, 4)


def revise_coordinates(mol_contents, new_coordinates):
    """Rewrite the atom block of a MOL file with new coordinates."""

    num_atoms = mol_contents[3][:3].rstrip()
    num_atoms = int(num_atoms)
    head_contents = mol_contents[:4]
    atom_contents = mol_contents[4 : 4 + num_atoms]
    end_contents = mol_contents[4 + num_atoms :]

    new_atom_contents = []
    for i, atom_content in enumerate(atom_contents):
        new_atom_content = [
            new_coordinates[i][0],
            new_coordinates[i][1],
            new_coordinates[i][2],
            atom_content.split()[3],
            atom_content.split()[-1],
        ]
        formatted_string = "{:>10}{:>10}{:>10} {:<2} {:>6}".format(
            new_atom_content[0],
            new_atom_content[1],
            new_atom_content[2],
            new_atom_content[3],
            new_atom_content[4],
        )
        formatted_string += "\n"
        new_atom_contents.append(formatted_string)
    head_contents.extend(new_atom_contents)
    head_contents.extend(end_contents)
    return head_contents


def generate_aug_files_single(
    molfile,
    outdir,
    out_prefix,
    num_aug=10,
    rotation_step_size=2,
    translation_step_size=20,
    seed=123,
):
    """Write ``num_aug`` augmented copies of one MOL file.

    Output files are named ``{out_prefix}_{i}.mol`` under ``outdir``.
    """

    os.chdir(outdir)

    mol_contents = get_mol_contents(molfile)
    aug_coordinates = get_aug_coordinates_single(
        mol_contents,
        num_aug=num_aug,
        rotation_step_size=rotation_step_size,
        translation_step_size=translation_step_size,
        seed=seed,
    )
    for i in range(num_aug):
        new_mol_contents = revise_coordinates(mol_contents, aug_coordinates[i])
        with open(f"{out_prefix}_{i}.mol", "w") as f:
            f.writelines(new_mol_contents)
    return None


def generate_aug_files_double(
    host_molfile,
    guest_molfile,
    outdir,
    out_prefix,
    num_aug=10,
    rotation_step_size=2,
    translation_step_size=20,
    seed=123,
):
    """Write ``num_aug`` augmented host/guest pairs.

    For a pair ``{out_prefix}_1.mol`` / ``{out_prefix}_2.mol`` the outputs are
    ``{out_prefix}_1_{i}.mol`` and ``{out_prefix}_2_{i}.mol`` under ``outdir``.
    """

    host_mol_contents = get_mol_contents(host_molfile)
    guest_mol_contents = get_mol_contents(guest_molfile)
    aug_host_coordinates, aug_guest_coordinates = get_aug_coordinates_double(
        host_mol_contents,
        guest_mol_contents,
        num_aug=num_aug,
        rotation_step_size=rotation_step_size,
        translation_step_size=translation_step_size,
        seed=seed,
    )
    for i, contents in enumerate(zip(aug_host_coordinates, aug_guest_coordinates)):
        new_host_mol_contents = revise_coordinates(host_mol_contents, contents[0])
        new_guest_mol_contents = revise_coordinates(guest_mol_contents, contents[1])
        host_file_path = os.path.join(outdir, f"{out_prefix}_1_{i}.mol")
        guest_file_path = os.path.join(outdir, f"{out_prefix}_2_{i}.mol")
        with open(host_file_path, "w") as f:
            f.writelines(new_host_mol_contents)
        with open(guest_file_path, "w") as f:
            f.writelines(new_guest_mol_contents)
    return None
