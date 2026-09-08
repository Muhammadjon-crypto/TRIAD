"""
triad.geometry.transforms
==========================
Rigid-body operations on 3D coordinate arrays, and deterministic near-uniform
sampling of SO(3) for the protein-protein rotational search.

All coordinate arrays are (N, 3) float64 NumPy arrays. All rotations are
represented as (3, 3) proper rotation matrices (det = +1).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def apply_transform(coords: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply a rigid-body transform x -> R @ x + t to an (N, 3) coordinate array.

    Parameters
    ----------
    coords : (N, 3) array
    R : (3, 3) proper rotation matrix
    t : (3,) translation vector

    Returns
    -------
    (N, 3) transformed coordinates
    """
    coords = np.asarray(coords, dtype=np.float64)
    return coords @ R.T + t


def centroid(coords: np.ndarray) -> np.ndarray:
    """Geometric centroid of an (N, 3) coordinate array."""
    return np.asarray(coords, dtype=np.float64).mean(axis=0)


def center_on_origin(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate coords so their centroid sits at the origin.

    Returns (centered_coords, original_centroid) so the translation can be
    inverted later.
    """
    c = centroid(coords)
    return coords - c, c


def fibonacci_sphere(n_points: int) -> np.ndarray:
    """Generate n_points quasi-uniformly distributed unit vectors on S^2
    using the Fibonacci lattice. Used as rotation axes for the orientation
    search grid — far better packing than uniform (theta, phi) sampling,
    which clusters points at the poles.
    """
    indices = np.arange(0, n_points, dtype=np.float64) + 0.5
    phi = np.arccos(1 - 2 * indices / n_points)
    golden_angle = np.pi * (1 + 5 ** 0.5)
    theta = golden_angle * indices

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def sample_rotations(n_axes: int = 200, n_angles_per_axis: int = 12) -> np.ndarray:
    """Deterministic near-uniform sampling of SO(3) via axis-angle
    parameterization: axes on a Fibonacci-sphere lattice, angles evenly
    spaced around each axis.

    This is the rotational search grid for docking one rigid body (e.g. the
    E3 ligase) around another (the target), analogous in spirit to the
    spherical-harmonic rotational grids used in TomoMiner's fast rotational
    matching, but implemented directly in axis-angle space since our search
    space per orientation is small enough not to need FFT acceleration yet.

    BUG FOUND AND FIXED (docs/TRIAD_v1_MANIFEST.md Part 8): the angle grid
    used to start at 0 (`np.linspace(0, 2*pi, n, endpoint=False)`), and a
    rotation by 0 degrees about ANY axis is identity regardless of which
    axis was chosen. That meant every one of the n_axes sampled axes
    produced an identical angle=0 rotation, wasting n_axes of the
    n_axes*n_angles_per_axis total samples on exact duplicates of identity
    instead of genuine orientation diversity -- for n_axes=30, that's 1 in
    6 of the entire "180 rotation" search silently testing the same
    orientation over and over. Fixed by shifting the angle grid by half a
    bin-width, so no sampled angle is ever exactly 0 (or equivalently 2*pi)
    -- every (axis, angle) pair now gives a genuinely distinct rotation,
    with no change to the output array's shape or to any other property
    (properness, orthogonality) that existing tests check.

    Returns
    -------
    (n_axes * n_angles_per_axis, 3, 3) array of rotation matrices
    """
    axes = fibonacci_sphere(n_axes)
    half_bin = np.pi / n_angles_per_axis
    angles = np.linspace(0, 2 * np.pi, n_angles_per_axis, endpoint=False) + half_bin

    rotvecs = []
    for axis in axes:
        for angle in angles:
            rotvecs.append(axis * angle)
    rotvecs = np.asarray(rotvecs)

    return Rotation.from_rotvec(rotvecs).as_matrix()


def random_rotation(rng: np.random.Generator | None = None) -> np.ndarray:
    """A single uniformly random rotation matrix (Haar measure on SO(3)).
    Used to generate scrambled decoy orientations for the benchmark suite.
    """
    rng = rng or np.random.default_rng()
    return Rotation.random(random_state=rng).as_matrix()
