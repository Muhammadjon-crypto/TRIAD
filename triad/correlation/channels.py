"""
triad.correlation.channels
=============================
Katchalski-Katzir-style shape-complementarity grid construction.

CORRECTED formulation (an earlier version of this module got this wrong --
see below): the receptor's "surface reward" applies to the EMPTY VOXELS
immediately surrounding its solid volume (the thin shell of space where a
well-packed docking partner's atoms should sit), NOT to voxels at the
receptor's own surface-atom positions. The "interior penalty" applies to
solid voxels deep inside the body (fully surrounded by other solid voxels),
representing a genuine steric clash if the ligand's atoms land there.

MISTAKE FOUND AND FIXED DURING DEVELOPMENT: an earlier version defined
"surface" as voxels AT atom positions classified as surface-exposed (using
a neighbor-count heuristic). This is wrong: those voxels are still part of
the receptor's own solid body, so rewarding the ligand for occupying them
was rewarding a mild clash, not genuine complementary packing. Confirmed by
testing on real 5T35 data: the true native pose scored far below other,
non-native translations under the old (wrong) definition. The fix below
uses voxel-adjacency geometry directly (correct per Katchalski-Katzir 1992),
not atom-level burial classification.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from triad.correlation.grid import voxelize_atoms

INTERIOR_PENALTY = -15.0
SURFACE_REWARD = 1.0


def build_receptor_shape_grid(
    coords: np.ndarray, radii: np.ndarray,
    grid_shape: tuple[int, int, int], origin: np.ndarray, spacing: float,
) -> np.ndarray:
    """Fixed-body shape grid, correct Katchalski-Katzir definition:
      - solid voxels fully surrounded by other solid voxels (26-connectivity)
        -> INTERIOR_PENALTY (deep inside the body; any ligand atom landing
        here is a genuine steric clash)
      - empty voxels directly adjacent to the solid body -> SURFACE_REWARD
        (the shell where a complementary partner should sit)
      - everything else -> 0
    """
    solid = voxelize_atoms(coords, radii, grid_shape, origin, spacing) > 0

    # a solid voxel is "interior" if ALL 26 neighbors are also solid (no
    # boundary showing in any direction) -- computed via binary erosion
    interior = ndimage.binary_erosion(solid, structure=np.ones((3, 3, 3)))

    # the surface shell is the empty voxels touching the solid body --
    # computed via binary dilation of the solid, then subtracting the solid
    # itself, leaving only the newly-added (previously empty) boundary layer
    dilated = ndimage.binary_dilation(solid, structure=np.ones((3, 3, 3)))
    surface_shell = dilated & ~solid

    grid = np.zeros(grid_shape, dtype=np.float64)
    grid[surface_shell] = SURFACE_REWARD
    grid[interior] = INTERIOR_PENALTY
    return grid


def build_ligand_shape_grid(
    coords: np.ndarray, radii: np.ndarray,
    grid_shape: tuple[int, int, int], origin: np.ndarray, spacing: float,
) -> np.ndarray:
    """Mobile-body shape grid: simple binary occupancy (any atom present = 1)."""
    occ = voxelize_atoms(coords, radii, grid_shape, origin, spacing)
    return (occ > 0).astype(np.float64)


def build_charge_grid(
    coords: np.ndarray, charges: np.ndarray,
    grid_shape: tuple[int, int, int], origin: np.ndarray, spacing: float,
) -> np.ndarray:
    """Point charges deposited onto the nearest grid voxel (nearest-neighbor
    assignment -- simple and adequate at the ~1.5 A spacing used here;
    higher-order charge spreading, e.g. cloud-in-cell, would reduce grid
    discretization artifacts but isn't necessary at this resolution).
    """
    grid = np.zeros(grid_shape, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    voxel_idx = np.round((coords - origin) / spacing).astype(int)
    for idx, q in zip(voxel_idx, charges):
        if q == 0:
            continue
        if np.all((idx >= 0) & (idx < np.array(grid_shape))):
            grid[tuple(idx)] += q
    return grid


def build_coulomb_kernel(grid_shape: tuple[int, int, int], spacing: float) -> np.ndarray:
    """1/r Coulomb kernel on the same periodic grid convention as
    everything else here (see triad.correlation.grid.periodic_distance_grid).
    The r=0 singularity is regularized to a minimum distance of half a
    voxel spacing -- a standard, simple regularization avoiding a
    divide-by-zero while keeping the self-term finite and small relative to
    genuine near-neighbor interactions.
    """
    from triad.correlation.grid import periodic_distance_grid

    dist_voxels = periodic_distance_grid(grid_shape, center=(0, 0, 0))
    dist_angstrom = dist_voxels * spacing
    dist_angstrom = np.maximum(dist_angstrom, spacing * 0.5)  # regularize r=0
    return 1.0 / dist_angstrom


def build_receptor_potential_grid(
    coords: np.ndarray, charges: np.ndarray,
    grid_shape: tuple[int, int, int], origin: np.ndarray, spacing: float,
) -> np.ndarray:
    """Electrostatic potential field generated by the receptor's formal
    charges, via free-space Coulomb convolution:

        potential(x) = k * sum_y charge(y) / |x - y|
                      = k * (charge_grid CONVOLVED WITH 1/r kernel)(x)

    Uses the SAME COULOMB_CONSTANT as triad.scoring.electrostatics for
    consistency, and the same formal-charge scheme -- this is the grid-based
    counterpart of that module's pairwise Coulomb sum, not a different model.
    """
    from triad.correlation.fft_dock import fft_convolve_3d
    from triad.scoring.electrostatics import COULOMB_CONSTANT

    charge_grid = build_charge_grid(coords, charges, grid_shape, origin, spacing)
    kernel = build_coulomb_kernel(grid_shape, spacing)
    potential = fft_convolve_3d(charge_grid, kernel)
    return COULOMB_CONSTANT * potential
