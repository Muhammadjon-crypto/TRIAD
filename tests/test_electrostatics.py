"""
Analytical validation of triad.scoring.electrostatics against known physical
behavior: same-charge repulsion, opposite-charge attraction, correct
distance decay, and linear charge-magnitude scaling.
"""
import numpy as np

from triad.scoring.electrostatics import (
    coulomb_energy, assign_formal_charges, COULOMB_CONSTANT,
)


def test_opposite_charges_are_attractive():
    coords_a = np.array([[0.0, 0.0, 0.0]])
    coords_b = np.array([[4.0, 0.0, 0.0]])
    energy = coulomb_energy(coords_a, np.array([1.0]), coords_b, np.array([-1.0]))
    assert energy < 0, f"opposite charges should give negative (attractive) energy, got {energy}"


def test_same_sign_charges_are_repulsive():
    coords_a = np.array([[0.0, 0.0, 0.0]])
    coords_b = np.array([[4.0, 0.0, 0.0]])
    energy = coulomb_energy(coords_a, np.array([1.0]), coords_b, np.array([1.0]))
    assert energy > 0, f"same-sign charges should give positive (repulsive) energy, got {energy}"


def test_energy_matches_exact_formula():
    """For a single isolated pair, energy should match k*q1*q2/r^2 exactly."""
    coords_a = np.array([[0.0, 0.0, 0.0]])
    coords_b = np.array([[5.0, 0.0, 0.0]])
    q1, q2 = 0.5, -0.5
    energy = coulomb_energy(coords_a, np.array([q1]), coords_b, np.array([q2]))
    expected = COULOMB_CONSTANT * q1 * q2 / 5.0 ** 2
    assert abs(energy - expected) < 1e-9, f"expected {expected}, got {energy}"


def test_energy_decays_with_distance():
    coords_a = np.array([[0.0, 0.0, 0.0]])
    charges_a = np.array([1.0])
    charges_b = np.array([-1.0])

    distances = [3.0, 5.0, 10.0, 20.0]
    magnitudes = []
    for d in distances:
        coords_b = np.array([[d, 0.0, 0.0]])
        e = coulomb_energy(coords_a, charges_a, coords_b, charges_b)
        magnitudes.append(abs(e))

    for i in range(len(magnitudes) - 1):
        assert magnitudes[i] > magnitudes[i + 1], (
            f"energy magnitude should strictly decrease with distance: "
            f"{magnitudes}"
        )


def test_charge_magnitude_scales_linearly():
    coords_a = np.array([[0.0, 0.0, 0.0]])
    coords_b = np.array([[5.0, 0.0, 0.0]])

    e1 = coulomb_energy(coords_a, np.array([1.0]), coords_b, np.array([-1.0]))
    e2 = coulomb_energy(coords_a, np.array([2.0]), coords_b, np.array([-1.0]))

    assert abs(e2 - 2 * e1) < 1e-9, f"doubling one charge should double energy: {e1} vs {e2}"


def test_zero_charge_atoms_contribute_nothing():
    """Neutral atoms (the vast majority of any real protein) should not
    affect the energy at all.
    """
    coords_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    charges_a = np.array([1.0, 0.0])  # second atom neutral
    coords_b = np.array([[5.0, 0.0, 0.0]])
    charges_b = np.array([-1.0])

    energy_with_neutral = coulomb_energy(coords_a, charges_a, coords_b, charges_b)
    energy_charged_only = coulomb_energy(
        coords_a[:1], charges_a[:1], coords_b, charges_b
    )
    assert abs(energy_with_neutral - energy_charged_only) < 1e-9


def test_formal_charge_assignment_on_known_residues():
    """Sanity check against real PDB-style residue/atom naming: only the
    documented ionizable atoms should get nonzero charge.
    """
    residue_names = ["ASP", "ASP", "LYS", "GLY", "ARG", "HIS"]
    atom_names = ["OD1", "CA", "NZ", "CA", "NH1", "NE2"]
    charges = assign_formal_charges(residue_names, atom_names)

    expected = np.array([-0.5, 0.0, 1.0, 0.0, 0.5, 0.0])  # HIS treated as neutral
    np.testing.assert_array_equal(charges, expected)


def test_salt_bridge_geometry_is_favorable():
    """A realistic Asp-Lys salt bridge distance (~3.0 A between carboxylate
    O and ammonium N) should give a clearly favorable (negative) energy,
    much stronger than a distant, non-interacting pair.
    """
    asp_od1 = np.array([[0.0, 0.0, 0.0]])
    lys_nz_close = np.array([[3.0, 0.0, 0.0]])   # realistic salt bridge distance
    lys_nz_far = np.array([[30.0, 0.0, 0.0]])     # far apart, no real interaction

    e_close = coulomb_energy(asp_od1, np.array([-0.5]), lys_nz_close, np.array([1.0]))
    e_far = coulomb_energy(asp_od1, np.array([-0.5]), lys_nz_far, np.array([1.0]))

    assert e_close < 0
    assert abs(e_close) > abs(e_far) * 50, (
        f"salt bridge at realistic distance should be dramatically stronger "
        f"than the same charges 10x farther apart: close={e_close}, far={e_far}"
    )


if __name__ == "__main__":
    test_opposite_charges_are_attractive()
    test_same_sign_charges_are_repulsive()
    test_energy_matches_exact_formula()
    test_energy_decays_with_distance()
    test_charge_magnitude_scales_linearly()
    test_zero_charge_atoms_contribute_nothing()
    test_formal_charge_assignment_on_known_residues()
    test_salt_bridge_geometry_is_favorable()
    print("All electrostatics analytical validation tests passed.")
