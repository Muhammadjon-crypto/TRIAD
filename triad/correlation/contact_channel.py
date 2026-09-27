"""
triad.correlation.contact_channel
=====================================
Properly integrated per-residue-type FFT correlation channel for the
knowledge-based contact potential (triad.potentials.contact_potential),
replacing the earlier post-hoc re-scoring approach used in Part 6/11 of
docs/TRIAD_v1_MANIFEST.md.

MATHEMATICAL TRICK (per manifest Part 2.2's original design): the total
contact-potential score for a candidate translation tau is, in principle,

    Score(tau) = sum_{i,j}  potential(i,j) * correlate(Receptor_i, Ligand_j)(tau)

summed over all 20x20 residue-type pairs (i = receptor residue type,
j = ligand residue type). Computed naively this needs up to 400 separate
FFT correlations per rotation -- far too slow. Using linearity of
correlation:

    sum_j potential(i,j) * correlate(Receptor_i, Ligand_j)
        = correlate( Receptor_i,  sum_j potential(i,j)*Ligand_j )
        = correlate( Receptor_i,  L'_i )

where L'_i(x) = sum_j potential(i,j) * Ligand_j(x) is a "weighted ligand
combination grid" for receptor type i -- a cheap weighted SUM of the 20
ligand-type occupancy grids (no correlation needed to build it), computed
ONCE per rotation. This reduces the whole channel to exactly 20 FFT
correlations (one per receptor residue type) instead of up to 400 --
the same asymptotic cost class as the shape or electrostatic channels.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from triad.correlation.grid import voxelize_atoms
from triad.correlation.fft_dock import fft_correlate_3d
from triad.potentials.contact_potential import STANDARD_AA, ContactPotential


def build_residue_type_grids(
    coords: np.ndarray, resnames: list[str], radii: np.ndarray,
    grid_shape: tuple[int, int, int], origin: np.ndarray, spacing: float,
    dilation_radius: float = 0.0,
) -> dict[str, np.ndarray]:
    """One binary occupancy grid per standard residue type, built from only
    the atoms belonging to that type. Non-standard residues (if any) are
    silently excluded -- they carry no contact-potential score anyway.

    BUG FOUND AND FIXED (docs/TRIAD_v1_MANIFEST.md Part 33): the original
    version used raw atom-sphere occupancy with no dilation, which measures
    direct atom-sphere INTERPENETRATION -- the same definition appropriate
    for clash detection, where a valid, non-clashing pose correctly shows
    ZERO overlap. Real touching-but-not-clashing atoms at van der Waals
    contact distance (~3.4 A apart, ~1.7 A radii) do not interpenetrate at
    all, so this definition gave native poses a contact-potential score of
    essentially zero regardless of how good the real interface was, while
    rewarding poses with more incidental interpenetration (borderline
    clashing) as if that were favorable chemical contact. `dilation_radius`
    (in the same units as `spacing`, e.g. Angstroms) expands each residue-
    type's occupied region outward before overlap is computed, so contact
    between two residues near but not literally overlapping is correctly
    captured. A dilation_radius of 0.0 (the original, buggy default)
    reproduces the old interpenetration-only behavior and should not be
    used for contact-potential scoring going forward.
    """
    grids = {}
    resnames_arr = np.array(resnames)
    dilation_voxels = int(np.ceil(dilation_radius / spacing)) if dilation_radius > 0 else 0
    structure = ndimage.generate_binary_structure(3, 1) if dilation_voxels > 0 else None
    for aa in STANDARD_AA:
        mask = resnames_arr == aa
        if not np.any(mask):
            grids[aa] = np.zeros(grid_shape, dtype=np.float64)
            continue
        occ = voxelize_atoms(coords[mask], radii[mask], grid_shape, origin, spacing)
        binary_occ = occ > 0
        if dilation_voxels > 0:
            binary_occ = ndimage.binary_dilation(binary_occ, structure=structure, iterations=dilation_voxels)
        grids[aa] = binary_occ.astype(np.float64)
    return grids


def build_weighted_ligand_combination_grids(
    ligand_type_grids: dict[str, np.ndarray],
    potential: ContactPotential,
) -> dict[str, np.ndarray]:
    """For each receptor residue type i, precompute L'_i = sum_j
    potential(i,j) * Ligand_j -- the weighted combination grid that makes
    the linearity trick work. Returns one combination grid per receptor
    residue type.
    """
    grid_shape = next(iter(ligand_type_grids.values())).shape
    combos = {}
    for aa_i in STANDARD_AA:
        combo = np.zeros(grid_shape, dtype=np.float64)
        for aa_j in STANDARD_AA:
            weight = potential.get(aa_i, aa_j)
            if weight != 0.0:
                combo += weight * ligand_type_grids[aa_j]
        combos[aa_i] = combo
    return combos


def contact_potential_correlation(
    receptor_type_grids: dict[str, np.ndarray],
    ligand_type_grids: dict[str, np.ndarray],
    potential: ContactPotential,
) -> np.ndarray:
    """The full contact-potential correlation score for every translation
    simultaneously: sum over the 20 receptor residue types of
    correlate(Receptor_i, L'_i). Returns a real-valued grid the same shape
    as the input grids, directly addable to the shape/electrostatic
    correlation grids for a combined score.
    """
    combos = build_weighted_ligand_combination_grids(ligand_type_grids, potential)
    grid_shape = next(iter(receptor_type_grids.values())).shape
    total = np.zeros(grid_shape, dtype=np.float64)
    for aa_i in STANDARD_AA:
        if np.any(receptor_type_grids[aa_i]) and np.any(combos[aa_i]):
            total += fft_correlate_3d(receptor_type_grids[aa_i], combos[aa_i])
    return total
