"""
Real-data test of the Phase 2 shape + electrostatic correlation channels
against 5T35, run at the CORRECT (native) rotation to isolate translational
discrimination power. This locks in an honest finding, not a success case:
shape alone, and shape+electrostatics combined, both fail to rank the true
native pose at the top even when restricted to the reach-constrained search
space -- consistent with, and explaining, why production docking tools add
a third knowledge-based pairwise-contact channel (see
docs/TRIAD_v1_MANIFEST.md Part 4 for the full discussion).
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
    build_receptor_potential_grid, build_charge_grid,
)
from triad.correlation.fft_dock import fft_correlate_3d
from triad.scoring.electrostatics import assign_formal_charges, COULOMB_CONSTANT
from triad.benchmark.manifest import BENCHMARK_SET

PDB_DIR = "pdb_raw"
pytestmark = pytest.mark.skipif(
    not os.path.isdir(PDB_DIR), reason=f"{PDB_DIR}/ not present"
)


def test_coulomb_kernel_matches_exact_freespace_physics():
    """The potential solver's 1/r decay must match exact Coulomb's law --
    the physics correctness gate, independent of any docking result.
    """
    grid_shape = (40, 40, 40)
    spacing = 1.0
    origin = np.array([0.0, 0.0, 0.0])
    coords = np.array([[20.0, 20.0, 20.0]])
    charges = np.array([1.0])
    pot = build_receptor_potential_grid(coords, charges, grid_shape, origin, spacing)

    for r in [1, 2, 4, 8]:
        idx = (20 + r, 20, 20)
        expected = COULOMB_CONSTANT * 1.0 / r
        np.testing.assert_allclose(pot[idx], expected, rtol=1e-6)


def test_shape_and_electrostatics_do_not_rank_native_best_at_5t35():
    """Documents the actual, honest finding: at the correct native
    rotation, restricted to reach-constrained translations, neither shape
    alone nor shape+electrostatics ranks the true native pose at rank 1.
    If this test starts failing because native suddenly IS best, verify
    it's a genuine improvement (e.g. a better charge model), not a bug,
    before updating this test and the manifest.
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
    lig_resn = sum([p[2] for p in lig_parts], [])
    lig_atomn = sum([p[4] for p in lig_parts], [])
    t_radii = np.full(len(t_coords), 1.7)
    lig_radii = np.full(len(lig_coords), 1.7)
    t_charges = assign_formal_charges(t_resn, t_atomn)
    lig_charges = assign_formal_charges(lig_resn, lig_atomn)

    all_coords = np.concatenate([t_coords, lig_coords])
    spacing, pad = 1.5, 10.0
    lo = all_coords.min(axis=0) - pad
    hi = all_coords.max(axis=0) + pad
    grid_shape = tuple(int(np.ceil((hi[i] - lo[i]) / spacing)) for i in range(3))

    shape_receptor = build_receptor_shape_grid(t_coords, t_radii, grid_shape, lo, spacing)
    shape_ligand = build_ligand_shape_grid(lig_coords, lig_radii, grid_shape, lo, spacing)
    C_shape = fft_correlate_3d(shape_receptor, shape_ligand)

    potential_receptor = build_receptor_potential_grid(t_coords, t_charges, grid_shape, lo, spacing)
    charge_ligand = build_charge_grid(lig_coords, lig_charges, grid_shape, lo, spacing)
    C_elec = -fft_correlate_3d(potential_receptor, charge_ligand)

    fixed_anchor = chain.all_coords[reach.farthest_atom_idx]
    mobile_anchor = reach.ligase_warhead_centroid
    reach_dist = reach.straight_line_distance_angstrom
    nx, ny, nz = grid_shape
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    tau_phys = np.stack([ix * spacing, iy * spacing, iz * spacing], axis=-1)
    for d, n in enumerate(grid_shape):
        half = n * spacing / 2
        tau_phys[..., d] = np.where(tau_phys[..., d] > half, tau_phys[..., d] - n * spacing, tau_phys[..., d])
    dist_to_fixed = np.linalg.norm(mobile_anchor + tau_phys - fixed_anchor, axis=-1)
    reach_mask = np.abs(dist_to_fixed - reach_dist) <= 3.0

    C_combined = C_shape + 0.1 * C_elec
    native_val = C_combined[0, 0, 0]
    rank = int(np.sum(C_combined[reach_mask] > native_val)) + 1
    n_candidates = int(reach_mask.sum())

    assert rank > 1, (
        f"native unexpectedly ranked #1 of {n_candidates} -- if this is a "
        f"genuine improvement, update this test and the manifest, don't "
        f"just loosen the assertion"
    )


if __name__ == "__main__":
    test_coulomb_kernel_matches_exact_freespace_physics()
    test_shape_and_electrostatics_do_not_rank_native_best_at_5t35()
    print("Phase 2 real-data regression tests passed (including the documented limitation).")
