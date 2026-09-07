"""
Integration test across the full 15-structure TRIAD benchmark set.

This requires the real PDB files in pdb_raw/ (fetched via
benchmark/fetch_benchmark_set.sh) — it is skipped, not failed, if that
directory doesn't exist, so the fast unit tests (test_rmsd.py,
test_pdb_parser.py) still run without the ~26MB benchmark download.

For every entry, this confirms:
  - the file parses without error
  - every ligase/target chain listed in the manifest actually has atoms
  - the ligand code declared in the manifest is present as a hetero chain
  - the resolution matches what's declared (to 2 decimal places)
"""
import os

import pytest

from triad.io.pdb_parser import load_structure
from triad.benchmark.manifest import BENCHMARK_SET

PDB_DIR = "pdb_raw"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(PDB_DIR),
    reason=f"{PDB_DIR}/ not present — run benchmark/fetch_benchmark_set.sh first",
)


@pytest.mark.parametrize("pdb_id", sorted(BENCHMARK_SET.keys()))
def test_structure_matches_manifest(pdb_id):
    entry = BENCHMARK_SET[pdb_id]
    path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
    assert os.path.isfile(path), f"missing file for {pdb_id}, expected at {path}"

    structure = load_structure(path, structure_id=pdb_id)

    for chain_id in entry.ligase_chains + entry.target_chains:
        assert chain_id in structure.chains, (
            f"{pdb_id}: manifest expects chain {chain_id!r} but parser found "
            f"only {structure.chain_ids()}"
        )
        assert structure.chains[chain_id].all_coords.shape[0] > 0, (
            f"{pdb_id}: chain {chain_id!r} has zero atoms"
        )

    het_chains = structure.hetero_chain_ids()
    ligand_present = any(
        structure.chains[c].all_residue_names[0] == entry.ligand_code
        for c in het_chains
    )
    assert ligand_present, (
        f"{pdb_id}: manifest expects ligand {entry.ligand_code!r}, "
        f"found hetero groups {[structure.chains[c].all_residue_names[0] for c in het_chains]}"
    )

    assert structure.resolution_angstrom is not None
    assert abs(structure.resolution_angstrom - entry.resolution_angstrom) < 0.01, (
        f"{pdb_id}: manifest says {entry.resolution_angstrom} A, "
        f"file header says {structure.resolution_angstrom} A"
    )


if __name__ == "__main__":
    if not os.path.isdir(PDB_DIR):
        print(f"{PDB_DIR}/ not found — run benchmark/fetch_benchmark_set.sh first.")
        raise SystemExit(1)

    failures = []
    for pdb_id in sorted(BENCHMARK_SET.keys()):
        try:
            test_structure_matches_manifest(pdb_id)
            print(f"{pdb_id}: OK")
        except AssertionError as e:
            failures.append(pdb_id)
            print(f"{pdb_id}: FAILED -- {e}")

    if failures:
        print(f"\n{len(failures)}/{len(BENCHMARK_SET)} structures failed: {failures}")
        raise SystemExit(1)
    print(f"\nAll {len(BENCHMARK_SET)} benchmark structures verified against manifest.")
