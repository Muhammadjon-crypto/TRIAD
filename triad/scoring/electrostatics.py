"""
triad.scoring.electrostatics
==============================
Electrostatic interaction energy between two rigid bodies, using formal
charges on ionizable side-chain atoms (Asp/Glu carboxylates, Lys/Arg
cationic groups) and a distance-dependent dielectric Coulomb term.

STATED LIMITATION, not hidden: this is a deliberately simplified model, not
a force-field-derived partial-charge scheme (AMBER/CHARMM assign small
partial charges to every atom, including backbone and neutral side chains).
This model only charges the classic salt-bridge-forming groups, which
dominate specific electrostatic recognition at protein-protein interfaces
anyway -- but it will miss weaker polar contributions (backbone dipoles,
His tautomer-dependent charge, Ser/Thr/Tyr hydroxyl partial charges). This
is an intentional, stated scoping choice to get a real, checkable
electrostatic term without depending on an external force-field parameter
file or an MD engine.

Distance-dependent dielectric (epsilon_r = r) is a standard, documented
simplification for implicit solvent screening in fast scoring contexts,
reducing Coulomb's law to E = k*q_i*q_j / r^2. The Coulomb constant k=332.0637
(kcal*A/(mol*e^2)) is the standard textbook value used throughout molecular
mechanics force fields (AMBER, CHARMM) when charges are in elementary-charge
units and distances in Angstroms.
"""
from __future__ import annotations

import numpy as np

COULOMB_CONSTANT = 332.0637  # kcal*A / (mol*e^2), standard MM force-field value

# Formal charges on classic salt-bridge-forming side-chain atoms only.
# Histidine is treated as neutral (a stated simplification -- its real
# protonation state is pH- and environment-dependent, pKa ~6).
FORMAL_CHARGES: dict[tuple[str, str], float] = {
    ("ASP", "OD1"): -0.5, ("ASP", "OD2"): -0.5,
    ("GLU", "OE1"): -0.5, ("GLU", "OE2"): -0.5,
    ("LYS", "NZ"): 1.0,
    ("ARG", "NH1"): 0.5, ("ARG", "NH2"): 0.5,
}


def assign_formal_charges(residue_names: list[str], atom_names: list[str]) -> np.ndarray:
    """Per-atom formal charge array, zero everywhere except the ionizable
    side-chain atoms listed in FORMAL_CHARGES.
    """
    return np.array([
        FORMAL_CHARGES.get((rn, an), 0.0)
        for rn, an in zip(residue_names, atom_names)
    ])


def coulomb_energy(
    coords_a: np.ndarray, charges_a: np.ndarray,
    coords_b: np.ndarray, charges_b: np.ndarray,
    min_distance: float = 1.0,
) -> float:
    """Total Coulombic interaction energy between two charged atom sets,
    using a distance-dependent dielectric (epsilon_r = r):

        E = sum_ij  k * q_i * q_j / r_ij^2

    Negative E = net favorable (attractive) electrostatic interaction;
    positive E = net unfavorable (repulsive). Only atoms with nonzero
    charge contribute, so this is cheap even for large atom counts --
    typically only ~10-15% of protein atoms carry a nonzero formal charge
    under this scheme.

    `min_distance` guards against division blow-up for coincident atoms
    (shouldn't occur for real, non-clashing poses, but caps the term rather
    than returning inf if it does).
    """
    nonzero_a = charges_a != 0
    nonzero_b = charges_b != 0
    if not nonzero_a.any() or not nonzero_b.any():
        return 0.0

    ca = coords_a[nonzero_a]
    qa = charges_a[nonzero_a]
    cb = coords_b[nonzero_b]
    qb = charges_b[nonzero_b]

    diffs = ca[:, None, :] - cb[None, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    dists = np.clip(dists, a_min=min_distance, a_max=None)

    charge_products = qa[:, None] * qb[None, :]
    energy = COULOMB_CONSTANT * charge_products / (dists ** 2)
    return float(np.sum(energy))
