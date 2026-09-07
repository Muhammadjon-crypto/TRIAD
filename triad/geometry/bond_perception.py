"""
triad.geometry.bond_perception
================================
Distance-based covalent bond detection from 3D coordinates, using explicit
covalent radii rather than RDKit's built-in `proximityBonding` heuristic.

Why this exists: RDKit's `Chem.MolFromPDBBlock(..., proximityBonding=True)`
was found (empirically, on the real MZ1/759 ligand from 5T35) to over-trigger
bonds in compact fused polycyclic systems — it added spurious bonds at
distances of 2.3-2.5 A, well outside the 1.3-1.8 A range of any real single
or double covalent bond, causing a downstream valence exception. Rather than
patch around that failure mode, this module implements the perception step
directly with covalent radii we can inspect and tune, and every value here
is checkable against a reference (Cordero et al. 2008).

This is deliberately NOT trying to be a general-purpose bond-order perceiver
(aromaticity, double vs single bonds) — only connectivity (which atoms are
bonded to which). Bond ORDER is assigned afterward by RDKit's sanitization
once connectivity is fixed, which is a much better-posed problem than doing
both at once.
"""
from __future__ import annotations

import numpy as np

# Covalent radii in Angstrom, single-bond values (Cordero et al., 2008,
# Dalton Trans., "Covalent radii revisited"). Only elements expected in the
# TRIAD benchmark ligands + common metals in E3 ligase structures.
COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "P": 1.07, "S": 1.05, "CL": 1.02, "BR": 1.20, "I": 1.39,
    "ZN": 1.22, "MG": 1.41, "CA": 1.76, "NA": 1.66, "K": 2.03,
    "FE": 1.32, "MN": 1.39, "CU": 1.32, "NI": 1.24, "CO": 1.26,
    "SE": 1.20,
}

DEFAULT_RADIUS = 0.75  # fallback for unrecognized elements — logged by caller
DEFAULT_TOLERANCE = 0.45  # additive slack in Angstrom (Open Babel convention)


def guess_element(atom_name: str) -> str:
    """LAST-RESORT fallback only — do not use this as the primary source of
    element identity. Ligand atom names are arbitrary labels, not systematic
    element codes: this WILL misclassify real atoms whose name happens to
    start with a two-letter element symbol (confirmed bug: ligand atoms
    named "CAV"/"CAT"/"CAU" are carbon, but this function reads their "CA"
    prefix as calcium). Prefer `Chain.all_elements` from
    `triad.io.pdb_parser`, which comes from the file's actual element
    column via BioPython, whenever it's available. Only reach for this
    function when parsing a source with no element information at all.
    """
    stripped = atom_name.strip().upper().lstrip("0123456789")
    if len(stripped) >= 2 and stripped[:2] in COVALENT_RADII:
        return stripped[:2]
    return stripped[:1]


def detect_bonds(
    coords: np.ndarray,
    elements: list[str],
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[tuple[int, int, float]]:
    """Detect covalent bonds from 3D coordinates using covalent-radii-sum
    distance cutoffs.

    A bond is placed between atoms i, j if:
        distance(i, j) <= radius(elem_i) + radius(elem_j) + tolerance

    Parameters
    ----------
    coords : (N, 3) array
    elements : length-N list of element symbols (already resolved, e.g. via
        `guess_element` — this function does not do name parsing itself)
    tolerance : additive slack in Angstrom. 0.45 A is the Open Babel default
        and was confirmed on real ligand data (see module docstring) to
        correctly separate real bonds (1.3-1.8 A) from ring-crowding
        false positives (>2.2 A) in a fused-polycyclic PROTAC warhead.

    Returns
    -------
    List of (i, j, distance) tuples, i < j, one entry per detected bond.
    """
    n = len(coords)
    coords = np.asarray(coords, dtype=np.float64)
    radii = np.array([COVALENT_RADII.get(e.upper(), DEFAULT_RADIUS) for e in elements])

    bonds = []
    for i in range(n):
        # vectorized distance from atom i to all atoms j > i
        diffs = coords[i + 1:] - coords[i]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        cutoffs = radii[i] + radii[i + 1:] + tolerance
        hits = np.where(dists <= cutoffs)[0]
        for h in hits:
            j = i + 1 + h
            bonds.append((i, int(j), float(dists[h])))
    return bonds


def bond_length_sanity_check(
    bonds: list[tuple[int, int, float]],
    min_length: float = 0.4,
) -> list[str]:
    """Flag any detected bond shorter than physically plausible (atoms
    placed implausibly close, e.g. overlapping altloc conformers that
    weren't properly separated upstream). Does not flag long bonds — the
    cutoff in `detect_bonds` already bounds those.
    """
    return [
        f"bond ({i},{j}) has implausibly short length {d:.3f} A "
        f"(possible overlapping/duplicate atoms)"
        for i, j, d in bonds if d < min_length
    ]
