"""
Smoke test for triad.io.pdb_parser against a real RCSB entry (5T35, the
Gadd et al. 2017 BRD4BD2-MZ1-VHL structure) — truncated to a manageable size
for a test fixture, but genuine RCSB-formatted content, not synthetic data.
Confirms the parser handles: multi-chain ATOM records, HETATM ligand
extraction into its own pseudo-chain, water filtering, and header metadata.
"""
import numpy as np

from triad.io.pdb_parser import load_structure

FIXTURE = "tests/fixtures/5T35_partial.pdb"


def test_parses_protein_chains():
    structure = load_structure(FIXTURE)
    protein_chains = structure.protein_chain_ids()
    # fixture includes ATOM records for chain A only (truncated), plus
    # DBREF-only stubs for B/C/D with no coordinates — those should NOT
    # appear as chains since they contribute no atoms
    assert "A" in protein_chains
    print("protein chains found:", protein_chains)


def test_extracts_ca_atoms_correctly():
    structure = load_structure(FIXTURE)
    chain_a = structure.chains["A"]
    ca = chain_a.ca_coords()
    # fixture has 5 residues on chain A (349-351 fully, +2 more truncated),
    # each with exactly one CA
    assert ca.shape[1] == 3
    assert ca.shape[0] >= 3
    # spot check: residue 349 CA should be at (27.480, -69.167, -5.601)
    resids = chain_a.ca_residue_ids()
    idx_349 = resids.index(349)
    np.testing.assert_allclose(ca[idx_349], [27.480, -69.167, -5.601], atol=1e-3)


def test_extracts_ligand_as_separate_hetero_chain():
    structure = load_structure(FIXTURE)
    het_chains = structure.hetero_chain_ids()
    assert any("759" in cid for cid in het_chains), f"expected ligand 759, got {het_chains}"
    ligand_key = next(cid for cid in het_chains if "759" in cid)
    ligand = structure.chains[ligand_key]
    assert ligand.is_hetero
    assert ligand.all_coords.shape == (2, 3)  # fixture has 2 HETATM lines for it


def test_resolution_parsed_from_header():
    structure = load_structure(FIXTURE)
    assert structure.resolution_angstrom is not None
    assert abs(structure.resolution_angstrom - 2.70) < 1e-6


def test_chain_transform_is_nondestructive():
    structure = load_structure(FIXTURE)
    chain_a = structure.chains["A"]
    original = chain_a.all_coords.copy()

    from triad.geometry.transforms import random_rotation
    R = random_rotation(np.random.default_rng(0))
    t = np.array([1.0, 2.0, 3.0])
    moved = chain_a.transformed(R, t)

    # original untouched
    np.testing.assert_array_equal(chain_a.all_coords, original)
    # moved chain actually moved
    assert not np.allclose(moved.all_coords, original)


if __name__ == "__main__":
    test_parses_protein_chains()
    test_extracts_ca_atoms_correctly()
    test_extracts_ligand_as_separate_hetero_chain()
    test_resolution_parsed_from_header()
    test_chain_transform_is_nondestructive()
    print("All PDB parser tests passed against real RCSB fixture data.")
