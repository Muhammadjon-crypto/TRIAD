"""
triad.correlation.grid
========================
Voxelization: converting atom coordinates (or, for synthetic testing,
simple analytic shapes) into 3D occupancy/property grids for FFT
correlation.

Grids here are treated as PERIODIC (a torus), matching the circular
correlation computed by triad.correlation.fft_dock -- shapes are defined
using wraparound-aware distance, and real docking use requires padding
grids generously beyond the two molecules' combined extent so that
translations of physical interest don't spuriously wrap around and
contaminate the correlation (see module docstring in fft_dock.py).
"""
from __future__ import annotations

import numpy as np


def periodic_distance_grid(grid_shape: tuple[int, int, int], center: tuple[int, int, int]) -> np.ndarray:
    """For each voxel, the minimum-image (periodic/wraparound) distance in
    voxel units to `center`. Used to define shapes consistently with the
    circular correlation's periodic assumption.
    """
    nx, ny, nz = grid_shape
    cx, cy, cz = center

    ix = np.arange(nx)
    iy = np.arange(ny)
    iz = np.arange(nz)

    def wrap_delta(idx, c, n):
        d = idx - c
        return np.minimum(np.abs(d), n - np.abs(d))

    dx = wrap_delta(ix, cx, nx)[:, None, None]
    dy = wrap_delta(iy, cy, ny)[None, :, None]
    dz = wrap_delta(iz, cz, nz)[None, None, :]

    return np.sqrt(dx.astype(np.float64) ** 2 + dy.astype(np.float64) ** 2 + dz.astype(np.float64) ** 2)


def voxelize_sphere(
    grid_shape: tuple[int, int, int],
    center: tuple[int, int, int],
    radius_voxels: float,
) -> np.ndarray:
    """A simple binary occupancy grid: 1.0 within `radius_voxels` of
    `center` (periodic/wraparound distance), 0.0 elsewhere. Used for
    synthetic validation of the correlation search before any real atom
    data is involved.
    """
    dist = periodic_distance_grid(grid_shape, center)
    return (dist <= radius_voxels).astype(np.float64)


def voxelize_atoms(
    coords: np.ndarray,
    atom_radii: np.ndarray,
    grid_shape: tuple[int, int, int],
    origin: np.ndarray,
    spacing: float,
) -> np.ndarray:
    """Real atom-based occupancy grid: each atom marks all voxels within its
    radius as occupied (value 1.0; overlapping atoms don't double-count).

    `origin` is the physical (Angstrom) coordinate of voxel index (0,0,0).
    `spacing` is Angstrom per voxel. This is NOT periodic-aware (real
    molecules should be placed well within the grid with generous padding,
    not relying on wraparound) -- periodic wraparound is a property of the
    correlation math, not something real atom placement should depend on.
    """
    grid = np.zeros(grid_shape, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    voxel_coords = (coords - origin) / spacing  # atom positions in voxel-index space

    for pos, radius in zip(voxel_coords, atom_radii):
        r_vox = radius / spacing
        lo = np.maximum(np.floor(pos - r_vox).astype(int), 0)
        hi = np.minimum(np.ceil(pos + r_vox).astype(int) + 1, grid_shape)
        if np.any(lo >= hi):
            continue  # atom entirely outside the grid -- skip, don't error
        ix = np.arange(lo[0], hi[0])
        iy = np.arange(lo[1], hi[1])
        iz = np.arange(lo[2], hi[2])
        dx = (ix[:, None, None] - pos[0]) ** 2
        dy = (iy[None, :, None] - pos[1]) ** 2
        dz = (iz[None, None, :] - pos[2]) ** 2
        dist2 = dx + dy + dz
        mask = dist2 <= r_vox ** 2
        sub = grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        sub[mask] = 1.0
        grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = sub

    return grid
