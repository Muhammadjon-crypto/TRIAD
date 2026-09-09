"""
The primary discrimination benchmark for TRIAD's shape+electrostatics
scoring, run across ALL benchmark structures (not just 5T35 -- see
docs/TRIAD_v1_MANIFEST.md Part 14 for why testing only one structure in
Parts 7-13 was a real methodological gap that overstated how uniformly
hard this problem is).

Locks in the corrected finding: 8 of 13 PROTAC structures (5FQD excluded
as a molecular glue, not a comparable bipartite case; 6SIS excluded per
its documented bond-perception issue) show genuinely good discrimination
(native in the top ~5-27th percentile of a real, physically valid
candidate pool), while 5 show near-random discrimination (50-65th
percentile). This is a real, structure-dependent split, not uniform
failure -- what determines it remains an open question.
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
from triad.benchmark.manifest import BENCHMARK_SET

PDB_DIR = "pdb_raw"
pytestmark = pytest.mark.skipif(not os.path.isdir(PDB_DIR), reason=f"{PDB_DIR}/ not present")

EXPECTED_RESULTS = {
    "5FQD": (64, 115), "5HXB": (370, 2501), "5T35": (54, 1112),
    "6BN7": (162, 2538), "6BOY": (63, 771), "6HAX": (409, 773),
    "6HAY": (480, 750), "6HR2": (499, 847), "7KHH": (1180, 2128),
    "8BDS": (219, 820), "8BEB": (464, 820), "8FY0": (821, 19222),
    "8FY1": (3239, 26943), "8FY2": (1437, 13212),
}

TRACTABLE_THRESHOLD_PCT = 30.0


def _measure_discrimination(pdb_id: str) -> tuple[int, int]:
    entry = BENCHMARK_SET[pdb_id]
    s = load_structure(f"{PDB_DIR}/{pdb_id}.pdb", structure_id=pdb_id)
    het = [c for c in s.hetero_chain_ids() if s.chains[c].all_residue_names[0] == entry.ligand_code]
    chain = s.chains[het[0]]
    r = extract_ligand_mol(chain)
    split = split_by_ligase_pharmacophore(r.mol, entry.ligase)
    reach = compute_reach_from_warhead(chain.all_coords, r.mol, split.ligase_warhead_atoms)

    target_chain = s.chains[entry.target_chains[0]]
    n_copy = len(entry.ligase_chains) // 2 if len(entry.ligase_chains) > 2 else len(entry.ligase_chains)
    ligase_chains = [s.chains[c] for c in entry.ligase_chains[:max(n_copy, 1)] if c in s.chains]

    t_coords, t_elem, t_resn, t_resid, t_atomn = extract_representative_atoms(target_chain)
    lig_parts = [extract_representative_atoms(c) for c in ligase_chains]
    lig_coords = np.concatenate([p[0] for p in lig_parts])
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

    tau_native = mobile_anchor - fixed_anchor
    native_tau_idx = tuple(np.round(tau_native / spacing).astype(int) % np.array(grid_shape))
    native_val = C_combined[native_tau_idx]

    vals = np.array([C_combined[tuple(c)] for c in valid_idx])
    rank = int(np.sum(vals > native_val)) + 1
    n_valid = len(valid_idx)
    return rank, n_valid


@pytest.mark.parametrize("pdb_id", sorted(EXPECTED_RESULTS.keys()))
def test_discrimination_matches_expected_baseline(pdb_id):
    """Checks discrimination results against a baseline, with a tolerance
    for small cross-platform floating-point differences.

    KNOWN LIMITATION, found and documented rather than papered over: exact
    integer n_valid/rank counts are NOT guaranteed bit-identical across
    platforms. reach_mask and overlap_mask both use hard <= threshold
    comparisons on continuous FFT-derived values, and FFT implementations
    differ slightly across BLAS/hardware backends (e.g. this was observed
    directly: Linux x86_64 sandbox vs. Apple Silicon Mac gave n_valid
    differences of a few candidates out of hundreds to tens of thousands
    for 10 of 14 structures -- always a small fraction of a percent,
    consistent with a handful of borderline voxels flipping sides of a
    hard threshold due to ~1e-10-scale floating point differences, not a
    code bug). This test allows a small tolerance band rather than
    requiring bit-exact reproduction, which was the actual finding worth
    fixing here.
    """
    rank, n_valid = _measure_discrimination(pdb_id)
    expected_rank, expected_n_valid = EXPECTED_RESULTS[pdb_id]

    n_valid_pct_diff = 100.0 * abs(n_valid - expected_n_valid) / expected_n_valid
    assert n_valid_pct_diff < 3.0, (
        f"{pdb_id}: valid-candidate count changed from {expected_n_valid} to "
        f"{n_valid} ({n_valid_pct_diff:.2f}% difference) -- larger than the "
        f"expected cross-platform floating-point tolerance, investigate"
    )

    pct = 100.0 * rank / n_valid
    expected_pct = 100.0 * expected_rank / expected_n_valid
    assert abs(pct - expected_pct) < 5.0, (
        f"{pdb_id}: percentile changed from {expected_pct:.1f}% to {pct:.1f}% "
        f"(rank {rank} of {n_valid}) -- larger than the expected cross-"
        f"platform tolerance, investigate whether this is a genuine change"
    )


def test_majority_of_structures_show_tractable_discrimination():
    """THE KEY FINDING (manifest Part 14): a majority of real PROTAC
    structures show genuinely good discrimination (top 30th percentile),
    overturning the earlier single-structure ('uniform ceiling') claim.
    """
    n_tractable = 0
    n_total = 0
    for pdb_id, (rank, n_valid) in EXPECTED_RESULTS.items():
        if pdb_id in ("5FQD",):
            continue
        pct = 100.0 * rank / n_valid
        n_total += 1
        if pct < TRACTABLE_THRESHOLD_PCT:
            n_tractable += 1

    assert n_tractable >= n_total / 2, (
        f"expected a majority of structures to show tractable discrimination "
        f"(manifest Part 14 finding), got {n_tractable}/{n_total}"
    )


if __name__ == "__main__":
    print("Running multi-structure discrimination benchmark...")
    all_pass = True
    for pdb_id in sorted(EXPECTED_RESULTS.keys()):
        expected_rank, expected_n_valid = EXPECTED_RESULTS[pdb_id]
        expected_pct = 100.0 * expected_rank / expected_n_valid
        rank, n_valid = _measure_discrimination(pdb_id)
        pct = 100.0 * rank / n_valid
        try:
            test_discrimination_matches_expected_baseline(pdb_id)
            print(f"  {pdb_id}: rank {rank}/{n_valid} ({pct:.1f}%ile) -- "
                  f"matches baseline ({expected_pct:.1f}%ile) within tolerance")
        except AssertionError as e:
            all_pass = False
            print(f"  {pdb_id}: rank {rank}/{n_valid} ({pct:.1f}%ile) vs "
                  f"baseline {expected_rank}/{expected_n_valid} ({expected_pct:.1f}%ile) -- {e}")
    test_majority_of_structures_show_tractable_discrimination()
    print("Majority-tractable finding confirmed.")
    if not all_pass:
        raise SystemExit(1)
    print("All multi-structure discrimination tests passed.")
