"""
Regression test for triad.topology.linker_geometry, run across the full
benchmark set (excluding 6SIS, whose bond-perception issue is documented in
triad/benchmark/manifest.py and tracked separately).

Two things are checked per structure:
  1. Sanity: the mean bond length along the computed reach path must fall
     within a physically plausible range for real covalent bonds
     (1.0-1.8 A). A path that fails this is walking something other than a
     real bonded chain -- this would have caught the 6SIS-style spurious-
     cycle issue if it slipped through pharmacophore matching undetected.
  2. Regression: reach values are pinned to what was actually computed and
     manually sanity-checked against known PROTAC chemistry (e.g. the
     BCL-2/BCL-xL degraders 8FY0-2 showing dramatically longer reach than
     BRD4-targeting ones, consistent with published literature on those
     compounds' unusually extended architecture). A future change that
     shifts these values needs to be deliberate, not silent.
"""
import os

import pytest

from triad.io.pdb_parser import load_structure
from triad.io.ligand_prep import extract_ligand_mol
from triad.topology.pharmacophore import split_by_ligase_pharmacophore
from triad.topology.linker_geometry import compute_reach_from_warhead
from triad.benchmark.manifest import BENCHMARK_SET

PDB_DIR = "pdb_raw"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(PDB_DIR),
    reason=f"{PDB_DIR}/ not present -- run benchmark/fetch_benchmark_set.sh first",
)

# pdb_id -> (path_length_bonds, straight_line_distance_angstrom)
# Values as computed and manually validated against known PROTAC chemistry.
# 6SIS excluded: documented bond-perception issue, see manifest.py.
EXPECTED_REACH = {
    "5FQD": (1, 4.20),
    "5HXB": (9, 12.94),
    "5T35": (23, 10.31),
    "6BN7": (24, 17.77),
    "6BOY": (23, 10.75),
    "6HAX": (19, 15.25),
    "6HAY": (20, 15.13),
    "6HR2": (20, 16.30),
    "7KHH": (23, 9.51),
    "8BDS": (25, 5.81),
    "8BEB": (18, 5.76),
    "8FY0": (41, 33.92),
    "8FY1": (41, 40.13),
    "8FY2": (41, 29.69),
}


@pytest.mark.parametrize("pdb_id", sorted(EXPECTED_REACH.keys()))
def test_reach_metrics_match_expected(pdb_id):
    entry = BENCHMARK_SET[pdb_id]
    s = load_structure(f"{PDB_DIR}/{pdb_id}.pdb", structure_id=pdb_id)
    chain = s.chains[[c for c in s.hetero_chain_ids()
                       if s.chains[c].all_residue_names[0] == entry.ligand_code][0]]
    r = extract_ligand_mol(chain)
    split = split_by_ligase_pharmacophore(r.mol, entry.ligase)
    assert split.ligase_warhead_atoms, f"{pdb_id}: expected a ligase warhead match"

    reach = compute_reach_from_warhead(chain.all_coords, r.mol, split.ligase_warhead_atoms)

    assert 1.0 < reach.mean_bond_length_along_path < 1.8, (
        f"{pdb_id}: mean bond length along reach path is "
        f"{reach.mean_bond_length_along_path:.3f} A -- outside plausible "
        f"covalent bond range, path may be walking a bond-perception artifact"
    )

    expected_bonds, expected_dist = EXPECTED_REACH[pdb_id]
    assert reach.path_length_bonds == expected_bonds, (
        f"{pdb_id}: path length changed from {expected_bonds} to "
        f"{reach.path_length_bonds} bonds -- if this is an intentional "
        f"improvement (e.g. better bond perception), update EXPECTED_REACH"
    )
    assert abs(reach.straight_line_distance_angstrom - expected_dist) < 0.05, (
        f"{pdb_id}: reach distance changed from {expected_dist} to "
        f"{reach.straight_line_distance_angstrom:.2f} A"
    )


if __name__ == "__main__":
    if not os.path.isdir(PDB_DIR):
        print(f"{PDB_DIR}/ not found -- run benchmark/fetch_benchmark_set.sh first.")
        raise SystemExit(1)

    failures = []
    for pdb_id in sorted(EXPECTED_REACH.keys()):
        try:
            test_reach_metrics_match_expected(pdb_id)
            print(f"{pdb_id}: OK")
        except AssertionError as e:
            failures.append(pdb_id)
            print(f"{pdb_id}: FAILED -- {e}")

    if failures:
        print(f"\n{len(failures)}/{len(EXPECTED_REACH)} failed: {failures}")
        raise SystemExit(1)
    print(f"\nAll {len(EXPECTED_REACH)} reach-metric regressions passed.")
