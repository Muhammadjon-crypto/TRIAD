"""
Tests for triad.geometry.bond_perception, anchored to two real failure cases
found on 5T35's ligand (759/MZ1):
  1. RDKit's proximityBonding added spurious bonds >2.2 A at a fused-ring
     carbon, causing a valence exception.
  2. Name-based element guessing misread carbons named CAV/CAT/CAU as
     calcium, since ligand atom names are arbitrary and "CA" happens to
     also be calcium's symbol.

This test uses `Chain.all_elements` (the file's real, BioPython-parsed
element column) throughout, NOT `guess_element(atom_name)` — that name-based
guesser is a fallback of last resort, not the primary path, precisely
because of failure mode #2 above.
"""
import numpy as np

from triad.geometry.bond_perception import detect_bonds, bond_length_sanity_check
from triad.io.pdb_parser import load_structure

FIXTURE = "pdb_raw/5T35.pdb"  # requires the real benchmark set — see benchmark/fetch_benchmark_set.sh


def _load_759_ligand():
    structure = load_structure(FIXTURE, structure_id="5T35")
    key = next(c for c in structure.hetero_chain_ids()
               if structure.chains[c].all_residue_names[0] == "759")
    return structure.chains[key]


def test_atom5_has_exactly_three_bonds_not_six():
    """This is the specific case that broke RDKit's proximityBonding:
    atom index 5 (CCH) is an aromatic ring carbon that should have exactly
    3 heavy-atom neighbors (two ring carbons + the ring sulfur), not 6.
    """
    import os
    if not os.path.isfile(FIXTURE):
        import pytest
        pytest.skip("benchmark set not present")

    chain = _load_759_ligand()
    bonds = detect_bonds(chain.all_coords, chain.all_elements)

    atom5_bonds = [(i, j, d) for i, j, d in bonds if i == 5 or j == 5]
    assert len(atom5_bonds) == 3, f"expected 3 bonds at atom 5, got {atom5_bonds}"

    partner_names = sorted(
        chain.all_atom_names[j if i == 5 else i] for i, j, d in atom5_bonds
    )
    assert partner_names == sorted(["CBZ", "CCF", "SBR"]), partner_names


def test_all_bond_lengths_are_physically_plausible():
    import os
    if not os.path.isfile(FIXTURE):
        import pytest
        pytest.skip("benchmark set not present")

    chain = _load_759_ligand()
    bonds = detect_bonds(chain.all_coords, chain.all_elements)

    for i, j, d in bonds:
        assert 0.4 < d < 2.5, f"implausible bond length {d:.3f} A between atoms {i},{j}"

    warnings = bond_length_sanity_check(bonds)
    assert warnings == [], warnings


def test_real_file_elements_are_not_misread_as_calcium():
    """Regression test for the specific bug found during development: atoms
    named CAV, CAT, CAU in the real MZ1 ligand are carbon, but a name-based
    guesser reads their "CA" prefix as calcium. `Chain.all_elements` must
    come from the file's real element column and get this right.
    """
    import os
    if not os.path.isfile(FIXTURE):
        import pytest
        pytest.skip("benchmark set not present")

    chain = _load_759_ligand()
    for name, element in zip(chain.all_atom_names, chain.all_elements):
        if name in ("CAV", "CAT", "CAU"):
            assert element.upper() == "C", (
                f"atom {name!r} should be carbon, got element {element!r} "
                f"— this is the calcium-misread regression"
            )


if __name__ == "__main__":
    import os
    if not os.path.isfile(FIXTURE):
        print("benchmark set not present — run benchmark/fetch_benchmark_set.sh first")
        raise SystemExit(1)
    test_atom5_has_exactly_three_bonds_not_six()
    test_all_bond_lengths_are_physically_plausible()
    test_real_file_elements_are_not_misread_as_calcium()
    print("All bond perception tests passed against real MZ1 ligand data.")
