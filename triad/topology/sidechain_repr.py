"""
triad.topology.sidechain_repr
================================
Extracts a reduced but chemically meaningful atom representation per
residue: the backbone CA, the Cβ (general side-chain direction/bulk, absent
for Gly), and the specific functional heteroatoms that mediate charged/polar
recognition (e.g. Asp's OD1/OD2, Lys's NZ, Ser's OG).

This directly targets the failure diagnosed from two independent scoring
attempts (buried surface area and electrostatics) at CA-only resolution:
both failed to discriminate near-native poses, most likely because real
molecular recognition happens through side chains reaching outward from the
backbone -- a CA-only trace is geometrically blind to exactly where
specific contacts and salt bridges actually form. This representation
keeps compute tractable (2-5 atoms per residue, vs. 7-15 for full-atom)
while putting charges and shape-relevant atoms at their real 3D positions
instead of collapsed onto the backbone.

Explicitly NOT full-atom: aliphatic side-chain carbons between CB and the
functional terminus (e.g. Lys's CG/CD/CE) are omitted. This is a stated
simplification -- if this representation still fails to discriminate,
that's real evidence pointing toward needing full-atom detail, not proof
that side-chain information doesn't matter.
"""
from __future__ import annotations

import numpy as np

from triad.io.pdb_parser import Chain

REPRESENTATIVE_ATOMS: dict[str, list[str]] = {
    "ALA": ["CA", "CB"],
    "ARG": ["CA", "CB", "NE", "NH1", "NH2"],
    "ASN": ["CA", "CB", "OD1", "ND2"],
    "ASP": ["CA", "CB", "OD1", "OD2"],
    "CYS": ["CA", "CB", "SG"],
    "GLN": ["CA", "CB", "OE1", "NE2"],
    "GLU": ["CA", "CB", "OE1", "OE2"],
    "GLY": ["CA"],                          # no CB
    "HIS": ["CA", "CB", "ND1", "NE2"],
    "ILE": ["CA", "CB"],
    "LEU": ["CA", "CB"],
    "LYS": ["CA", "CB", "NZ"],
    "MET": ["CA", "CB", "SD"],
    "PHE": ["CA", "CB", "CZ"],
    "PRO": ["CA", "CB"],
    "SER": ["CA", "CB", "OG"],
    "THR": ["CA", "CB", "OG1"],
    "TRP": ["CA", "CB", "NE1"],
    "TYR": ["CA", "CB", "OH"],
    "VAL": ["CA", "CB"],
}


def extract_representative_atoms(
    chain: Chain,
) -> tuple[np.ndarray, list[str], list[str], list[int], list[str]]:
    """Filter a Chain's full atom list down to the representative subset.

    Returns (coords, elements, residue_names, residue_ids) for just the
    kept atoms, in their real 3D positions -- unlike a CA-only or
    charge-collapsed-to-CA representation, a Lys's NZ here is exactly where
    the crystal structure says it is.
    """
    keep_mask = [
        name in REPRESENTATIVE_ATOMS.get(resname, ["CA"])
        for name, resname in zip(chain.all_atom_names, chain.all_residue_names)
    ]
    coords = chain.all_coords[np.array(keep_mask, dtype=bool)]
    elements = [e for e, k in zip(chain.all_elements, keep_mask) if k]
    residue_names = [r for r, k in zip(chain.all_residue_names, keep_mask) if k]
    residue_ids = [r for r, k in zip(chain.all_residue_ids, keep_mask) if k]
    atom_names = [a for a, k in zip(chain.all_atom_names, keep_mask) if k]
    return coords, elements, residue_names, residue_ids, atom_names
