"""
triad.scoring.desolvation
============================
Atomic solvation potential (Eisenberg-McLachlan-style), estimating the
energetic cost of burying solvent-accessible surface area upon complex
formation. This is a real, missing physical effect: burying a nonpolar
(hydrophobic) atom is energetically FAVORABLE (the hydrophobic effect --
releasing ordered water), while burying a polar/charged atom without a
compensating polar partner is UNFAVORABLE (losing hydrogen-bonding/
solvation to water with nothing to replace it).

STATED SIMPLIFICATION, not hidden: real atomic solvation parameters
(Eisenberg & McLachlan 1986) are per-atom-type empirical values fit to
transfer free energies. Rather than hand-transcribe that published table
from memory -- the same transcription-error risk flagged for hand-written
SMILES in ligand_prep.py -- this uses a coarse, clearly-labeled two-category
model (nonpolar vs. polar/charged) with round, directionally-correct
parameter values. This is a legitimate physical model at reduced precision,
not a faithful reproduction of the literature table.

ENERGY = sum_atoms  sigma(element) * (SASA_free - SASA_complex)

sigma > 0 for nonpolar atoms: burying them (SASA_free > SASA_complex,
positive delta) REDUCES total unfavorable "isolated hydrophobic surface"
energy... actually stated directly: sigma > 0 means burying this atom type
LOWERS total energy (favorable), matching the real hydrophobic effect.
sigma < 0 for polar/charged atoms: burying them RAISES total energy
(unfavorable), matching the real cost of desolvating a polar group.
"""
from __future__ import annotations

import numpy as np

from triad.scoring.sasa import compute_sasa

NONPOLAR_ELEMENTS = {"C", "S"}
POLAR_ELEMENTS = {"N", "O"}

# Simplified, round, directionally-correct values (cal/(mol*A^2)), NOT a
# transcription of the exact Eisenberg-McLachlan table. Sign convention:
# positive sigma = favorable to bury (nonpolar); negative = unfavorable to
# bury (polar/charged). Order of magnitude matches the real effect being
# roughly comparable for nonpolar burial and polar desolvation cost.
SIGMA_NONPOLAR = 20.0
SIGMA_POLAR = -15.0
SIGMA_DEFAULT = 0.0  # unclassified elements (e.g. halogens, metals): neutral


def get_sigma(element: str) -> float:
    e = element.strip().upper()
    if e in NONPOLAR_ELEMENTS:
        return SIGMA_NONPOLAR
    if e in POLAR_ELEMENTS:
        return SIGMA_POLAR
    return SIGMA_DEFAULT


def desolvation_energy(
    coords_a: np.ndarray, elements_a: list[str], sasa_free_a: np.ndarray,
    coords_b: np.ndarray, elements_b: list[str], sasa_free_b: np.ndarray,
    n_points: int = 50,
) -> float:
    """Desolvation energy for a candidate complex, given PRECOMPUTED
    free-state per-atom SASA for each body (compute once, reuse across many
    candidate poses -- free-state SASA doesn't depend on the pose at all,
    only the complex-state SASA does).

    Returns total energy in the simplified sigma units above (favorable
    burial is NEGATIVE total energy contribution, matching a standard
    "lower is better" free-energy convention -- note this is the OPPOSITE
    sign convention from sigma itself, since sigma*delta_sasa is added as
    -sigma*delta_sasa isn't right either; see the explicit sign check in
    the docstring above and the validation tests before trusting this).
    """
    combined_coords = np.concatenate([coords_a, coords_b])
    combined_elements = elements_a + elements_b
    sasa_complex = compute_sasa(combined_coords, combined_elements, n_points)

    sasa_complex_a = sasa_complex[: len(coords_a)]
    sasa_complex_b = sasa_complex[len(coords_a):]

    delta_a = sasa_free_a - sasa_complex_a  # positive = buried upon complex formation
    delta_b = sasa_free_b - sasa_complex_b

    sigma_a = np.array([get_sigma(e) for e in elements_a])
    sigma_b = np.array([get_sigma(e) for e in elements_b])

    # burying a nonpolar atom (positive sigma, positive delta) should be
    # FAVORABLE -> negative energy contribution -> energy = -sigma*delta
    energy_a = -np.sum(sigma_a * delta_a)
    energy_b = -np.sum(sigma_b * delta_b)
    return float(energy_a + energy_b)
