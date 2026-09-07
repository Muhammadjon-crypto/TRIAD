"""
triad.scoring.sasa
====================
Solvent-accessible surface area (SASA) via the Shrake-Rupley algorithm
(Shrake & Rupley, J. Mol. Biol. 1973, 79:351-371), and buried surface area
between two rigid bodies -- the standard protein-protein docking interface
metric (used as a core scoring term in ZDOCK, HADDOCK, RosettaDock, and
essentially every serious docking tool).

This directly replaces the naive "count of atom pairs within a distance
window" interface proxy tried earlier, which was found (empirically, on
5T35's rotational search) to actively reward non-specific crowding over
genuine complementary binding -- adding it made near-native pose ranking
dramatically WORSE, not better. Burial fixes this because it accounts for
solvent exclusion geometrically: a point on one body's surface only counts
as "buried" if it is genuinely occluded by the other body, not merely
nearby.

Method: for each atom, sample points on a sphere of radius (vdW + probe)
around it (reusing the already-verified `fibonacci_sphere` point
distribution from triad.geometry.transforms). A point is "accessible" if no
other atom's expanded sphere covers it. Sum over all points, scaled by each
atom's share of full sphere area, gives per-atom SASA; summing over atoms
gives total SASA.
"""
from __future__ import annotations

import numpy as np

from triad.geometry.transforms import fibonacci_sphere
from triad.scoring.clash import VDW_RADII, DEFAULT_VDW

PROBE_RADIUS = 1.4  # water probe radius, Angstrom -- standard SASA convention


def _radii_for_elements(elements: list[str]) -> np.ndarray:
    return np.array([VDW_RADII.get(e.upper(), DEFAULT_VDW) for e in elements])


def compute_sasa(
    coords: np.ndarray,
    elements: list[str],
    n_points: int = 100,
    probe_radius: float = PROBE_RADIUS,
) -> np.ndarray:
    """Per-atom SASA for a single set of atoms (e.g. one isolated protein).

    Returns an (N,) array of per-atom SASA in square Angstroms. Total SASA
    is the sum.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n_atoms = len(coords)
    radii = _radii_for_elements(elements) + probe_radius
    sphere_points = fibonacci_sphere(n_points)  # unit vectors

    sasa = np.zeros(n_atoms)
    max_r = radii.max() if n_atoms else 0.0

    for i in range(n_atoms):
        r_i = radii[i]
        test_points = coords[i] + r_i * sphere_points  # (n_points, 3)

        dists_to_others = np.linalg.norm(coords - coords[i], axis=1)
        neighbor_mask = (dists_to_others > 0) & (dists_to_others < r_i + max_r)
        neighbor_idx = np.where(neighbor_mask)[0]

        accessible = np.ones(n_points, dtype=bool)
        for j in neighbor_idx:
            d = np.linalg.norm(test_points - coords[j], axis=1)
            accessible &= d > radii[j]
            if not accessible.any():
                break  # fully buried already, no need to check remaining neighbors

        sasa[i] = accessible.sum() / n_points * 4 * np.pi * r_i ** 2

    return sasa


def buried_surface_area(
    coords_a: np.ndarray, elements_a: list[str],
    coords_b: np.ndarray, elements_b: list[str],
    n_points: int = 100,
) -> float:
    """Buried surface area (BSA) between two rigid bodies:

        BSA = (SASA_a_alone + SASA_b_alone - SASA_complex) / 2

    This is THE standard protein-protein interface size metric. Larger BSA
    means more surface area is mutually occluded -- a genuine geometric
    signature of a complementary, packed interface, unlike a raw distance-
    window contact count which can't distinguish real burial from two
    surfaces merely passing near each other without actually excluding
    solvent between them.
    """
    sasa_a_alone = compute_sasa(coords_a, elements_a, n_points).sum()
    sasa_b_alone = compute_sasa(coords_b, elements_b, n_points).sum()

    combined_coords = np.concatenate([coords_a, coords_b])
    combined_elements = elements_a + elements_b
    sasa_complex = compute_sasa(combined_coords, combined_elements, n_points).sum()

    return (sasa_a_alone + sasa_b_alone - sasa_complex) / 2.0
