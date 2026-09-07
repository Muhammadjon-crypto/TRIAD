"""
Regression test for triad.topology.pharmacophore: confirms the VHL and CRBN
warhead patterns still match the expected set of benchmark structures, and
that the held-out validation (structures NOT used to discover the patterns)
still generalizes.

6SIS is the one documented, expected failure (see manifest.py) -- it's
asserted to fail here, not silently skipped, so that if a future bond-
perception fix resolves it, this test will start failing and flag that the
"expected failures" list needs updating.
"""
import os

import pytest

from triad.io.pdb_parser import load_structure
from triad.io.ligand_prep import extract_ligand_mol
from triad.topology.pharmacophore import split_by_ligase_pharmacophore
from triad.benchmark.manifest import BENCHMARK_SET

PDB_DIR = "pdb_raw"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(PDB_DIR),
    reason=f"{PDB_DIR}/ not present -- run benchmark/fetch_benchmark_set.sh first",
)

# Structures NOT used in original pattern discovery (5T35/6HAX/8BDS for VHL,
# 6BN7/5HXB for CRBN) -- these prove the patterns generalize rather than
# overfit the discovery set.
HELD_OUT_VHL = ["8FY0", "8FY1", "8FY2", "6HAY", "6HR2", "7KHH", "8BEB"]
HELD_OUT_CRBN = ["5FQD", "6BOY"]

KNOWN_FAILURES = {"6SIS"}  # documented bond-perception issue, see manifest.py


def _get_mol(pdb_id):
    entry = BENCHMARK_SET[pdb_id]
    s = load_structure(f"{PDB_DIR}/{pdb_id}.pdb", structure_id=pdb_id)
    chain = s.chains[[c for c in s.hetero_chain_ids()
                       if s.chains[c].all_residue_names[0] == entry.ligand_code][0]]
    return extract_ligand_mol(chain).mol, entry


@pytest.mark.parametrize("pdb_id", HELD_OUT_VHL)
def test_vhl_pattern_generalizes_to_holdout(pdb_id):
    mol, entry = _get_mol(pdb_id)
    split = split_by_ligase_pharmacophore(mol, "VHL")
    assert len(split.ligase_warhead_atoms) == 22, (
        f"{pdb_id}: expected full 22-atom VHL warhead match, got "
        f"{len(split.ligase_warhead_atoms)}"
    )


@pytest.mark.parametrize("pdb_id", HELD_OUT_CRBN)
def test_crbn_pattern_generalizes_to_holdout(pdb_id):
    mol, entry = _get_mol(pdb_id)
    split = split_by_ligase_pharmacophore(mol, "CRBN")
    assert len(split.ligase_warhead_atoms) == 18, (
        f"{pdb_id}: expected full 18-atom CRBN warhead match, got "
        f"{len(split.ligase_warhead_atoms)}"
    )


def test_6sis_known_bond_perception_issue():
    """This structure is EXPECTED to fail pharmacophore matching due to a
    documented bond-perception artifact (see manifest.py). If this test
    starts failing (i.e. 6SIS suddenly matches), that's good news -- update
    KNOWN_FAILURES and this test, don't just delete it.
    """
    mol, entry = _get_mol("6SIS")
    split = split_by_ligase_pharmacophore(mol, "VHL")
    assert not split.ligase_warhead_atoms, (
        "6SIS unexpectedly matched the VHL pattern -- the documented bond-"
        "perception issue may have been fixed; update manifest.py's note "
        "and this test's expectations accordingly"
    )


if __name__ == "__main__":
    if not os.path.isdir(PDB_DIR):
        print(f"{PDB_DIR}/ not found -- run benchmark/fetch_benchmark_set.sh first.")
        raise SystemExit(1)

    failures = []
    for pdb_id in HELD_OUT_VHL:
        try:
            test_vhl_pattern_generalizes_to_holdout(pdb_id)
            print(f"{pdb_id} (VHL holdout): OK")
        except AssertionError as e:
            failures.append(pdb_id)
            print(f"{pdb_id}: FAILED -- {e}")
    for pdb_id in HELD_OUT_CRBN:
        try:
            test_crbn_pattern_generalizes_to_holdout(pdb_id)
            print(f"{pdb_id} (CRBN holdout): OK")
        except AssertionError as e:
            failures.append(pdb_id)
            print(f"{pdb_id}: FAILED -- {e}")
    try:
        test_6sis_known_bond_perception_issue()
        print("6SIS (known issue): OK -- still fails as documented")
    except AssertionError as e:
        failures.append("6SIS")
        print(f"6SIS: UNEXPECTED -- {e}")

    if failures:
        print(f"\n{len(failures)} failed: {failures}")
        raise SystemExit(1)
    print("\nAll pharmacophore regression tests passed.")
