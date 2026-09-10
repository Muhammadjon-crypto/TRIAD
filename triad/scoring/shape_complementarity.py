"""
triad.scoring.shape_complementarity
=======================================
Real shape complementarity (Sc) statistic, per Lawrence & Colman (1993,
J. Mol. Biol. 234:946-950) -- the standard measure of how well two protein
surfaces "nest" against each other at their true contact patch, as opposed
to any whole-molecule aggregate statistic (all six of which were tested
and found insufficient in docs/TRIAD_v1_MANIFEST.md Parts 14-19).

METHOD: generate molecular surface points with outward normal vectors for
each body (reusing the same Shrake-Rupley accessible-point logic already
validated in triad.scoring.sasa, extended here to also return each
accessible point's position and normal, not just the total area). Restrict
to the INTERFACE PATCH (surface points near the other body). For each
interface point on body A, find its nearest neighbor point on body B's
interface patch and compute the dot product of A's normal with the
NEGATIVE of B's normal (complementary surfaces point oppositely where they
pack well -- a perfect nested fit gives a dot product of +1). Sc is the
average of the median dot product computed in each direction (A-to-B and
B-to-A).
"""
from __future__ import annotations

import numpy as np

from triad.geometry.transforms import fibonacci_sphere
from triad.scoring.clash import VDW_RADII, DEFAULT_VDW

PROBE_RADIUS = 1.4


def get_surface_points_and_normals(
    coords: np.ndarray, elements: list[str], n_points: int = 100, probe_radius: float = PROBE_RADIUS,
) -> tuple[np.ndarray, np.ndarray]:
    """Accessible surface points (in isolation) and their outward unit
    normal vectors, for one molecular body. Reuses the same accessibility
    logic as triad.scoring.sasa.compute_sasa, but returns the actual
    surviving points/normals instead of collapsing them to a scalar area.
    """
    coords = np.asarray(coords, dtype=np.float64)
    radii = np.array([VDW_RADII.get(e.upper(), DEFAULT_VDW) for e in elements]) + probe_radius
    sphere_dirs = fibonacci_sphere(n_points)
    max_r = radii.max() if len(radii) else 0.0

    all_points, all_normals = [], []
    for i in range(len(coords)):
        r_i = radii[i]
        test_points = coords[i] + r_i * sphere_dirs
        d_others = np.linalg.norm(coords - coords[i], axis=1)
        neighbor_idx = np.where((d_others > 0) & (d_others < r_i + max_r))[0]
        accessible = np.ones(n_points, dtype=bool)
        for j in neighbor_idx:
            d = np.linalg.norm(test_points - coords[j], axis=1)
            accessible &= d > radii[j]
        if accessible.any():
            all_points.append(test_points[accessible])
            all_normals.append(sphere_dirs[accessible])  # outward unit normal = sphere direction

    if not all_points:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(all_points), np.concatenate(all_normals)


def restrict_to_interface(
    points: np.ndarray, normals: np.ndarray, other_coords: np.ndarray, cutoff: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only surface points within `cutoff` of any atom of the other
    body -- the interface-facing patch, not the whole molecular surface.
    """
    if len(points) == 0 or len(other_coords) == 0:
        return points, normals
    dists = np.linalg.norm(points[:, None, :] - other_coords[None, :, :], axis=-1)
    min_dist = dists.min(axis=1)
    mask = min_dist <= cutoff
    return points[mask], normals[mask]


def _directional_sc(points_a, normals_a, points_b, normals_b) -> float | None:
    """Median normal-dot-product from A's interface points to their
    nearest neighbor on B's interface patch.
    """
    if len(points_a) == 0 or len(points_b) == 0:
        return None
    dists = np.linalg.norm(points_a[:, None, :] - points_b[None, :, :], axis=-1)
    nearest_idx = np.argmin(dists, axis=1)
    dots = np.sum(normals_a * (-normals_b[nearest_idx]), axis=1)
    return float(np.median(dots))


def shape_complementarity(
    coords_a: np.ndarray, elements_a: list[str],
    coords_b: np.ndarray, elements_b: list[str],
    n_points: int = 100, interface_cutoff: float = 6.0,
) -> float | None:
    """The Sc statistic for two bodies at a given (typically native) pose.
    Returns None if either body has no interface-facing surface points
    within the cutoff (bodies not in contact).

    PERFORMANCE NOTE: pre-filters each body to only atoms within a
    generous margin of the other body BEFORE generating surface points --
    generating full-chain surface points for large proteins (hundreds of
    residues) and only filtering to the interface afterward caused an
    out-of-memory kill on real data during development. Filtering atoms
    first (interface residues are typically only a few dozen atoms, not
    hundreds) makes this tractable without changing the result, since
    atoms far from the other body can never contribute an interface-patch
    surface point anyway.
    """
    margin = interface_cutoff + PROBE_RADIUS + 3.0  # generous: probe + largest plausible vdw radius
    elements_a = np.array(elements_a)
    elements_b = np.array(elements_b)

    dists_a_to_b = np.linalg.norm(coords_a[:, None, :] - coords_b[None, :, :], axis=-1)
    near_a = dists_a_to_b.min(axis=1) <= margin
    near_b = dists_a_to_b.min(axis=0) <= margin

    coords_a_near, elements_a_near = coords_a[near_a], list(elements_a[near_a])
    coords_b_near, elements_b_near = coords_b[near_b], list(elements_b[near_b])

    if len(coords_a_near) == 0 or len(coords_b_near) == 0:
        return None

    pts_a, norm_a = get_surface_points_and_normals(coords_a_near, elements_a_near, n_points)
    pts_b, norm_b = get_surface_points_and_normals(coords_b_near, elements_b_near, n_points)

    pts_a_if, norm_a_if = restrict_to_interface(pts_a, norm_a, coords_b_near, interface_cutoff)
    pts_b_if, norm_b_if = restrict_to_interface(pts_b, norm_b, coords_a_near, interface_cutoff)

    sc_ab = _directional_sc(pts_a_if, norm_a_if, pts_b_if, norm_b_if)
    sc_ba = _directional_sc(pts_b_if, norm_b_if, pts_a_if, norm_a_if)

    if sc_ab is None or sc_ba is None:
        return None
    return (sc_ab + sc_ba) / 2.0
