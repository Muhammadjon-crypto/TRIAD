"""
triad.scoring.clash
=====================
Steric clash detection between two sets of atoms (e.g. target protein vs.
candidate-oriented E3 ligase). This is the first, cheapest filter in the
docking pipeline: a pose with severe steric overlap is physically impossible
regardless of how good its interface score might otherwise look, so it
should be rejected before any more expensive scoring is computed.

Van der Waals radii are standard literature values (Bondi, J. Phys. Chem.
1964, 68:441 -- the most widely used VdW radii table in structural biology
and cheminformatics).
"""
from __future__ import annotations

import numpy as np

VDW_RADII: dict[str, float] = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
    "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98, "ZN": 1.39,
    "MG": 1.73, "CA": 2.31, "NA": 2.27, "K": 2.75, "FE": 2.00,
    "MN": 2.00, "CU": 1.40, "NI": 1.63, "CO": 2.00, "SE": 1.90,
}
DEFAULT_VDW = 1.70  # fallback for unlisted elements (typical carbon-ish value)


def count_clashes(
    coords_a: np.ndarray,
    elements_a: list[str],
    coords_b: np.ndarray,
    elements_b: list[str],
    overlap_tolerance: float = 0.4,
) -> tuple[int, np.ndarray]:
    """Count clashing atom pairs between two atom sets.

    A pair (i in A, j in B) is a clash if their distance is less than
    (vdw_radius_i + vdw_radius_j - overlap_tolerance). The tolerance allows
    for the normal, small van der Waals overlaps seen even in correctly
    solved crystal structures (real atoms are "soft," not hard spheres) --
    0.4 A is a standard permissive tolerance used in structural validation
    tools (e.g. MolProbity-style clash analysis uses a comparable margin).

    Returns
    -------
    (n_clashes, clash_mask) where clash_mask is a (len(A), len(B)) boolean
    array, so callers can inspect exactly which atoms are clashing.
    """
    coords_a = np.asarray(coords_a, dtype=np.float64)
    coords_b = np.asarray(coords_b, dtype=np.float64)
    radii_a = np.array([VDW_RADII.get(e.upper(), DEFAULT_VDW) for e in elements_a])
    radii_b = np.array([VDW_RADII.get(e.upper(), DEFAULT_VDW) for e in elements_b])

    diffs = coords_a[:, None, :] - coords_b[None, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    cutoffs = radii_a[:, None] + radii_b[None, :] - overlap_tolerance

    clash_mask = dists < cutoffs
    return int(clash_mask.sum()), clash_mask


def clash_score(
    coords_a: np.ndarray,
    elements_a: list[str],
    coords_b: np.ndarray,
    elements_b: list[str],
    overlap_tolerance: float = 0.4,
) -> float:
    """Continuous clash penalty: sum of (overlap_depth)^2 over all clashing
    pairs, rather than a raw count. This is more useful for ranking poses
    than a bare count, since it distinguishes "barely touching" clashes from
    "atoms on top of each other" ones -- a rotational search will produce
    many candidate poses with a handful of minor clashes, and this score
    lets those be ranked sensibly instead of all tied at the same count.
    """
    coords_a = np.asarray(coords_a, dtype=np.float64)
    coords_b = np.asarray(coords_b, dtype=np.float64)
    radii_a = np.array([VDW_RADII.get(e.upper(), DEFAULT_VDW) for e in elements_a])
    radii_b = np.array([VDW_RADII.get(e.upper(), DEFAULT_VDW) for e in elements_b])

    diffs = coords_a[:, None, :] - coords_b[None, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    cutoffs = radii_a[:, None] + radii_b[None, :] - overlap_tolerance

    overlap_depth = np.clip(cutoffs - dists, a_min=0.0, a_max=None)
    return float(np.sum(overlap_depth ** 2))
