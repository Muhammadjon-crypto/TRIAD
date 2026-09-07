"""
Tests for triad.correlation.search, locking in the two findings from
docs/TRIAD_v1_MANIFEST.md Part 7:
  1. The pose-application math is exact (verified to machine precision).
  2. The full rotation+translation search runs correctly (no plumbing bugs)
     but does not yet find a near-native pose, because scoring
     discrimination -- not search coverage -- is the limiting factor.
"""
import os

import numpy as np
import pytest

from triad.io.pdb_parser import load_structure
from triad.io.ligand_prep import extract_ligand_mol
from triad.topology.pharmacophore import split_by_ligase_pharmacophore
from triad.topology.linker_geometry import compute_reach_from_warhead
from triad.topology.sidechain_repr import extract_representative_atoms
from triad.correlation.channels import build_receptor_shape_grid, build_receptor_potential_grid
from triad.correlation.search import search_rotations, apply_search_result, SearchResult
from triad.geometry.rmsd import raw_rmsd
from triad.scoring.electrostatics import assign_formal_charges
from triad.benchmark.manifest import BENCHMARK_SET

PDB_DIR = "pdb_raw"
pytestmark = pytest.mark.skipif(not os.path.isdir(PDB_DIR), reason=f"{PDB_DIR}/ not present")


def _setup_5t35():
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
    ligase_ca_native = np.concatenate([c.ca_coords() for c in ligase_chains])

    return dict(
        t_coords=t_coords, t_resn=t_resn, t_atomn=t_atomn,
        lig_coords=lig_coords, lig_resn=lig_resn, lig_atomn=lig_atomn,
        ligase_ca_native=ligase_ca_native,
        mobile_anchor=reach.ligase_warhead_centroid,
        fixed_anchor=chain.all_coords[reach.farthest_atom_idx],
        reach_dist=reach.straight_line_distance_angstrom,
    )


def test_pose_application_is_exact_at_true_native_transform():
    """Feeding the TRUE native rotation and translation into
    apply_search_result must reproduce native CA positions to machine
    precision -- isolates pose math from search/scoring correctness.
    """
    d = _setup_5t35()
    true_tau = d["mobile_anchor"] - d["fixed_anchor"]
    fake_result = SearchResult(
        best_rotation=np.eye(3), best_translation=true_tau,
        best_score=0.0, n_rotations_tried=1,
    )
    posed = apply_search_result(
        d["ligase_ca_native"], d["mobile_anchor"], d["fixed_anchor"], fake_result,
    )
    rmsd = raw_rmsd(posed, d["ligase_ca_native"])
    assert rmsd < 1e-8, f"expected near-zero RMSD with true transform, got {rmsd}"


def test_full_rotation_search_runs_without_error_and_returns_valid_result():
    d = _setup_5t35()
    t_radii = np.full(len(d["t_coords"]), 1.7)
    lig_radii = np.full(len(d["lig_coords"]), 1.7)
    t_charges = assign_formal_charges(d["t_resn"], d["t_atomn"])
    lig_charges = assign_formal_charges(d["lig_resn"], d["lig_atomn"])

    spacing, pad = 1.5, d["reach_dist"] + 15.0
    lo = np.minimum(d["t_coords"].min(axis=0), d["fixed_anchor"] - d["reach_dist"]) - pad
    hi = np.maximum(d["t_coords"].max(axis=0), d["fixed_anchor"] + d["reach_dist"]) + pad
    grid_shape = tuple(int(np.ceil((hi[i] - lo[i]) / spacing)) for i in range(3))

    shape_r = build_receptor_shape_grid(d["t_coords"], t_radii, grid_shape, lo, spacing)
    pot_r = build_receptor_potential_grid(d["t_coords"], t_charges, grid_shape, lo, spacing)

    result = search_rotations(
        d["lig_coords"], lig_radii, lig_charges,
        d["mobile_anchor"], d["fixed_anchor"],
        shape_r, pot_r, grid_shape, lo, spacing, d["reach_dist"],
        n_axes=10, n_angles_per_axis=4,
    )

    assert abs(np.linalg.det(result.best_rotation) - 1.0) < 1e-6
    achieved_reach = np.linalg.norm(result.best_translation)
    assert abs(achieved_reach - d["reach_dist"]) <= 3.0


def test_discrimination_not_search_coverage_is_the_bottleneck():
    """HONEST DOCUMENTED FINDING (manifest Part 7): even at the TRUE native
    rotation, the FFT correlation's best-scoring translation is NOT the
    true native translation -- confirming scoring discrimination, not
    search coverage, is the limiting factor. If this ever finds the
    correct translation, investigate as a genuine improvement, don't just
    "fix" the test.
    """
    d = _setup_5t35()
    t_radii = np.full(len(d["t_coords"]), 1.7)
    lig_radii = np.full(len(d["lig_coords"]), 1.7)

    spacing, pad = 1.5, d["reach_dist"] + 15.0
    lo = np.minimum(d["t_coords"].min(axis=0), d["fixed_anchor"] - d["reach_dist"]) - pad
    hi = np.maximum(d["t_coords"].max(axis=0), d["fixed_anchor"] + d["reach_dist"]) + pad
    grid_shape = tuple(int(np.ceil((hi[i] - lo[i]) / spacing)) for i in range(3))

    shape_r = build_receptor_shape_grid(d["t_coords"], t_radii, grid_shape, lo, spacing)
    from triad.correlation.channels import build_ligand_shape_grid
    from triad.correlation.fft_dock import fft_correlate_3d
    from triad.correlation.search import build_reach_mask

    lig_centered = d["lig_coords"] - d["mobile_anchor"]
    embedded = lig_centered @ np.eye(3).T + d["fixed_anchor"]
    shape_l = build_ligand_shape_grid(embedded, lig_radii, grid_shape, lo, spacing)
    C_shape = fft_correlate_3d(shape_r, shape_l)

    tau, mask = build_reach_mask(grid_shape, spacing, d["reach_dist"])
    C_masked = np.where(mask, C_shape, -np.inf)
    best_idx = np.unravel_index(np.argmax(C_masked), C_masked.shape)
    found_tau = tau[best_idx]

    true_tau = d["mobile_anchor"] - d["fixed_anchor"]
    distance_from_true = np.linalg.norm(found_tau - true_tau)

    assert distance_from_true > 5.0, (
        f"found translation is only {distance_from_true:.2f} A from true native "
        f"-- if discrimination has improved, update this test's expectations"
    )


if __name__ == "__main__":
    test_pose_application_is_exact_at_true_native_transform()
    test_full_rotation_search_runs_without_error_and_returns_valid_result()
    test_discrimination_not_search_coverage_is_the_bottleneck()
    print("All rotation-search tests passed (infrastructure verified correct; "
          "discrimination limitation documented as expected).")
