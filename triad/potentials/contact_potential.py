"""
triad.potentials.contact_potential
=====================================
A knowledge-based residue-pair contact potential, derived empirically from
our own 15 verified benchmark structures -- NOT a hand-transcribed external
matrix (e.g. Miyazawa-Jernigan). This is a deliberate choice, not a
shortcut: recalling a 210-value published matrix from memory carries the
same transcription-error risk flagged for hand-written SMILES earlier in
this project (see triad/io/ligand_prep.py's module docstring) -- a wrong
number wouldn't crash, it would silently bias every future scoring result.

STATED LIMITATION, prominently: 15 structures (11 VHL-based, 4 CRBN-based,
many sharing near-identical ligase-binding contacts) is a SMALL, NON-
INDEPENDENT sample for deriving real statistics. A production-grade
potential needs a large, diverse protein-protein interface database (real
statistical potentials like Miyazawa-Jernigan are derived from thousands of
structures). This module produces a real, honestly-labeled empirical
potential from real data -- but its statistical power should not be
overstated, and its real value in this codebase is as a validated
methodology (contact extraction -> log-odds scoring -> biochemistry sanity
check) that could be re-run against a larger dataset later without any
change to the underlying approach.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

STANDARD_AA = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]


@dataclass
class ContactPotential:
    """Log-odds contact potential: score(i,j) = log(observed(i,j) / expected(i,j)).
    Positive = residue types i,j contact more often than chance in our
    dataset (favorable); negative = less often (unfavorable).
    """
    scores: dict[tuple[str, str], float]
    n_contacts_observed: int
    n_structures: int

    def get(self, resname_a: str, resname_b: str) -> float:
        key = tuple(sorted([resname_a, resname_b]))
        return self.scores.get(key, 0.0)  # unobserved pair -> neutral, not penalized


def extract_interface_contacts(
    target_coords: np.ndarray, target_resnames: list[str],
    ligase_coords: np.ndarray, ligase_resnames: list[str],
    contact_cutoff: float = 5.0,
) -> list[tuple[str, str]]:
    """All (target_residue_type, ligase_residue_type) pairs with at least
    one representative-atom contact within `contact_cutoff` Angstroms, in a
    REAL solved structure. One list entry per contacting atom pair (so
    residues with more contacting atoms contribute proportionally more --
    a reasonable weighting, since a larger contact patch is a stronger
    observation).
    """
    diffs = target_coords[:, None, :] - ligase_coords[None, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    contact_mask = dists <= contact_cutoff
    t_idx, l_idx = np.where(contact_mask)
    return [(target_resnames[i], ligase_resnames[j]) for i, j in zip(t_idx, l_idx)]


def classify_surface_residues(
    coords: np.ndarray, neighbor_radius: float = 8.0, burial_threshold: int = 15,
) -> np.ndarray:
    """Boolean array: True = surface-exposed (few neighbors within
    `neighbor_radius`). Used to restrict the background/expected-frequency
    population for the contact potential to residues that COULD physically
    appear at an interface -- an earlier version of this module used ALL
    residues (including deeply buried core) as background, which inflated
    the "expected by chance" frequency for hydrophobic residues (abundant
    in cores, never at interfaces) and made real hydrophobic packing
    incorrectly look unfavorable relative to chance. Confirmed by testing:
    switching to surface-only background fixed 6 of 8 basic biochemistry
    sanity checks (see tests/test_contact_potential.py).
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    is_surface = np.zeros(n, dtype=bool)
    for i in range(n):
        d = np.linalg.norm(coords - coords[i], axis=1)
        is_surface[i] = np.sum((d > 0) & (d < neighbor_radius)) < burial_threshold
    return is_surface


def derive_contact_potential(
    all_contacts: list[tuple[str, str]],
    all_background_resnames: list[str],
    n_structures: int,
    pseudocount: float = 1.0,
) -> ContactPotential:
    """Derive a log-odds contact potential from observed interface contacts
    across multiple structures.

    `all_background_resnames` should be every residue in every target and
    ligase chain used (not just contacting ones) -- this defines the
    "expected by chance" background composition (residue type frequencies),
    against which observed contact frequencies are compared.
    """
    symmetrized = [tuple(sorted(pair)) for pair in all_contacts]
    observed_counts = Counter(symmetrized)
    total_observed = sum(observed_counts.values())

    background_counts = Counter(all_background_resnames)
    total_background = sum(background_counts.values())
    freq = {aa: background_counts.get(aa, 0) / total_background for aa in STANDARD_AA}

    scores: dict[tuple[str, str], float] = {}
    for i, aa_i in enumerate(STANDARD_AA):
        for aa_j in STANDARD_AA[i:]:
            key = tuple(sorted([aa_i, aa_j]))
            obs = observed_counts.get(key, 0) + pseudocount
            obs_freq = obs / (total_observed + pseudocount * len(STANDARD_AA) ** 2)
            multiplicity = 1.0 if aa_i == aa_j else 2.0
            exp_freq = multiplicity * freq[aa_i] * freq[aa_j]
            if exp_freq > 0:
                scores[key] = float(np.log(obs_freq / exp_freq))

    return ContactPotential(
        scores=scores, n_contacts_observed=total_observed, n_structures=n_structures,
    )
