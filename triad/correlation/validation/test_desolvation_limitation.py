"""
Documents the honest finding from docs/TRIAD_v1_MANIFEST.md Part 11: adding
the validated desolvation term to shape+electrostatics ranking makes
native's rank WORSE, not better, among genuinely clash-filtered valid
candidates. Recorded as a permanent regression so this doesn't silently
change without notice.
"""
import os

import numpy as np
import pytest

from triad.io.pdb_parser import load_structure
from triad.io.ligand_prep import extract_ligand_mol
from triad.topology.pharmacophore import split_by_ligase_pharmacophore
from triad.topology.linker_geometry import compute_reach_from_warhead
from triad.topology.sidechain_repr import extract_representative_atoms
from triad.correlation.channels import (
    build_receptor_shape_grid, build_ligand_shape_grid,
    build_receptor_potential_grid, build_charge_grid, build_receptor_occupancy_grid,
)
from triad.correlation.fft_dock import fft_correlate_3d
from triad.correlation.search import build_reach_mask, build_overlap_mask
from triad.scoring.electrostatics import assign_formal_charges
from triad.scoring.sasa import compute_sasa
from triad.scoring.desolvation import desolvation_energy
from triad.benchmark.manifest import BENCHMARK_SET

PDB_DIR = "pdb_raw"
pytestmark = pytest.mark.skipif(not os.path.isdir(PDB_DIR), reason=f"{PDB_DIR}/ not present")


def test_desolvation_does_not_improve_native_ranking():
    """HONEST DOCUMENTED FINDING (manifest Part 11): adding desolvation to
    shape+electrostatics ranking, on the genuinely clash-filtered candidate
    pool at the correct rotation, makes native's rank worse as weight
    increases, not better. If this test starts failing (rank improves),
    investigate and update this test's expectations, don't just loosen it.
    """
    entry = BENCHMARK_SET["5T35"]
    s = load_structure(f"{PDB_DIR}/5T35.pdb", structure_id="5T35")
    chain = s.chains[[c for c in s.hetero_chain_ids()
                       if s.chains[c].all_residue_names[0] == entry.ligand_code][0]]
    r = extract_ligand_mol(chain)
    split = split_by_ligase_pharmacophore(r.mol, entry.ligase)
    reach = compute_reach_from_warhead(chain.all_coords, r.mol, split.ligase_warhead_atoms)
    target_chain = s.chains[entry.target_chains[0]]
    n_copy = len(entry.ligase_chains) // 2
    ligase_chains = [s.chains[c] for c in entry.ligase_chains[:n_copy]]
    t_coords, t_elem, t_resn, t_resid, t_atomn = extract_representative_atoms(target_chain)
    lig_parts = [extract_representative_atoms(c) for c in ligase_chains]
    lig_coords = np.concatenate([p[0] for p in lig_parts])
    lig_elem = sum([p[1] for p in lig_parts], [])
    lig_resn = sum([p[2] for p in lig_parts], [])
    lig_atomn = sum([p[4] for p in lig_parts], [])
    t_radii = np.full(len(t_coords), 1.7)
    lig_radii = np.full(len(lig_coords), 1.7)
    t_charges = assign_formal_charges(t_resn, t_atomn)
    lig_charges = assign_formal_charges(lig_resn, lig_atomn)

    mobile_anchor = reach.ligase_warhead_centroid
    fixed_anchor = chain.all_coords[reach.farthest_atom_idx]
    reach_dist = reach.straight_line_distance_angstrom
    lig_centered = lig_coords - mobile_anchor
    spacing, pad = 1.5, reach_dist + 15.0
    lo = np.minimum(t_coords.min(axis=0), fixed_anchor - reach_dist) - pad
    hi = np.maximum(t_coords.max(axis=0), fixed_anchor + reach_dist) + pad
    grid_shape = tuple(int(np.ceil((hi[i] - lo[i]) / spacing)) for i in range(3))

    shape_r = build_receptor_shape_grid(t_coords, t_radii, grid_shape, lo, spacing)
    pot_r = build_receptor_potential_grid(t_coords, t_charges, grid_shape, lo, spacing)
    occ_r = build_receptor_occupancy_grid(t_coords, t_radii, grid_shape, lo, spacing)
    embedded = lig_centered + fixed_anchor
    shape_l = build_ligand_shape_grid(embedded, lig_radii, grid_shape, lo, spacing)
    charge_l = build_charge_grid(embedded, lig_charges, grid_shape, lo, spacing)
    C_shape = fft_correlate_3d(shape_r, shape_l)
    C_elec = -fft_correlate_3d(pot_r, charge_l)
    C_combined = C_shape + 0.1 * C_elec

    tau, reach_mask = build_reach_mask(grid_shape, spacing, reach_dist)
    overlap_mask = build_overlap_mask(occ_r, shape_l)
    valid_mask = reach_mask & overlap_mask
    valid_idx = np.argwhere(valid_mask)

    scores = np.array([C_combined[tuple(c)] for c in valid_idx])
    order = np.argsort(-scores)
    top100 = valid_idx[order[:100]]

    sasa_free_t = compute_sasa(t_coords, t_elem, n_points=50)
    sasa_free_l = compute_sasa(lig_coords, lig_elem, n_points=50)

    desolv_scores = []
    for c in top100:
        posed = lig_centered + fixed_anchor + tau[tuple(c)]
        e = desolvation_energy(t_coords, t_elem, sasa_free_t, posed, lig_elem, sasa_free_l, n_points=50)
        desolv_scores.append(e)
    desolv_scores = np.array(desolv_scores)

    tau_native = mobile_anchor - fixed_anchor
    native_idx = next(
        (i for i, c in enumerate(top100) if np.allclose(tau[tuple(c)], tau_native, atol=spacing)),
        None,
    )
    assert native_idx is not None, "native pose not found in top-100 -- setup may have changed"

    top100_shape_scores = scores[order[:100]]
    rank_no_desolv = int(np.where(np.argsort(-top100_shape_scores) == native_idx)[0][0]) + 1

    combined_with_desolv = top100_shape_scores - 0.01 * desolv_scores
    rank_with_desolv = int(np.where(np.argsort(-combined_with_desolv) == native_idx)[0][0]) + 1

    assert rank_with_desolv >= rank_no_desolv, (
        f"expected desolvation to NOT improve rank (documented limitation), "
        f"but rank went from {rank_no_desolv} to {rank_with_desolv} -- if "
        f"this is a genuine improvement, update this test and manifest Part 11"
    )


if __name__ == "__main__":
    test_desolvation_does_not_improve_native_ranking()
    print("Desolvation limitation confirmed and locked in (manifest Part 11).")
