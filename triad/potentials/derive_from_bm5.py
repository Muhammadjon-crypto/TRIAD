"""
triad.potentials.derive_from_bm5
====================================
Derives a contact potential from the real Docking Benchmark 5 (Vreven et
al. 2015) -- 231 non-redundant, independent, diverse protein-protein
complexes, fetched via benchmark/fetch_bm5_interfaces.sh -- as a genuinely
large-sample alternative to the 15-structure, PROTAC-specific derivation in
triad.potentials.contact_potential.

STATUS (docs/TRIAD_v1_MANIFEST.md Part 13): built, run, and tested. The
resulting potential is statistically more robust in the categories that
generalize well (hydrophobic packing, like-charge repulsion both remain
correctly signed) but does NOT improve pose discrimination on 5T35 --
if anything, it's marginally worse than the small-sample PROTAC-specific
version at low weight. The most credible interpretation: BM5 is built
almost entirely from NATURAL, evolutionarily-selected interfaces, while a
PROTAC-induced ternary complex is a "neo-interface" never optimized by
evolution -- the statistical patterns may not transfer across that domain
boundary regardless of how much natural-PPI data is used. This suggests
the original small, PROTAC-specific dataset may be the more RELEVANT (if
still too small) source for this specific problem, not a limitation to
escape via generic scale.

Run this after benchmark/fetch_bm5_interfaces.sh has populated
BM5-clean/structures-matched/ alongside the triad repo (or adjust BM5_DIR).
"""
from __future__ import annotations

import numpy as np

from triad.io.pdb_parser import load_structure
from triad.topology.sidechain_repr import extract_representative_atoms
from triad.potentials.contact_potential import (
    extract_interface_contacts, classify_surface_residues, derive_contact_potential,
    ContactPotential,
)

BM5_DIR = "BM5-clean/structures-matched"


def derive_potential_from_bm5(bm5_dir: str = BM5_DIR, ids: list[str] | None = None) -> ContactPotential:
    """Process every complex in the BM5 directory (or a given subset of
    PDB IDs) into a single aggregated contact potential, using the SAME
    representative-atom extraction and surface classification as the
    original 15-structure derivation, for a fair, apples-to-apples
    comparison between the two data sources.
    """
    import os

    if ids is None:
        ids = sorted({
            f.split("_r_b-matched.pdb")[0]
            for f in os.listdir(bm5_dir) if f.endswith("_r_b-matched.pdb")
        })

    all_contacts, all_background = [], []
    n_used = 0

    for pdb_id in ids:
        try:
            r_struct = load_structure(f"{bm5_dir}/{pdb_id}_r_b-matched.pdb", structure_id=f"{pdb_id}_r")
            l_struct = load_structure(f"{bm5_dir}/{pdb_id}_l_b-matched.pdb", structure_id=f"{pdb_id}_l")

            r_coords_list, r_resn_list = [], []
            for cid in r_struct.protein_chain_ids():
                coords, elem, resn, resid, atomn = extract_representative_atoms(r_struct.chains[cid])
                r_coords_list.append(coords)
                r_resn_list.extend(resn)
            l_coords_list, l_resn_list = [], []
            for cid in l_struct.protein_chain_ids():
                coords, elem, resn, resid, atomn = extract_representative_atoms(l_struct.chains[cid])
                l_coords_list.append(coords)
                l_resn_list.extend(resn)

            if not r_coords_list or not l_coords_list:
                continue

            r_coords = np.concatenate(r_coords_list)
            l_coords = np.concatenate(l_coords_list)

            all_contacts.extend(extract_interface_contacts(r_coords, r_resn_list, l_coords, l_resn_list))
            r_surf = classify_surface_residues(r_coords)
            l_surf = classify_surface_residues(l_coords)
            all_background.extend([r for r, m in zip(r_resn_list, r_surf) if m])
            all_background.extend([r for r, m in zip(l_resn_list, l_surf) if m])
            n_used += 1
        except Exception:
            continue

    return derive_contact_potential(all_contacts, all_background, n_used)
