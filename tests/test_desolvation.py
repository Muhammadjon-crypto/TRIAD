"""
Validation of triad.scoring.desolvation against known physical behavior:
burying nonpolar atoms should be favorable (negative energy); burying
polar atoms should be unfavorable (positive energy). Required before this
touches any real docking score.
"""
import numpy as np

from triad.scoring.desolvation import desolvation_energy
from triad.scoring.sasa import compute_sasa


def test_burying_nonpolar_atoms_is_favorable():
    coords_a = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    elements_a = ["C", "C"]
    sasa_free_a = compute_sasa(coords_a, elements_a, n_points=200)

    coords_b_far = np.array([[500.0, 0.0, 0.0], [501.5, 0.0, 0.0]])
    coords_b_close = np.array([[3.2, 0.0, 0.0], [4.7, 0.0, 0.0]])
    elements_b = ["C", "C"]
    sasa_free_b = compute_sasa(coords_b_far, elements_b, n_points=200)

    energy_close = desolvation_energy(
        coords_a, elements_a, sasa_free_a, coords_b_close, elements_b, sasa_free_b, n_points=200,
    )
    assert energy_close < 0, f"burying nonpolar atoms should be favorable, got {energy_close:.2f}"


def test_burying_polar_atoms_is_unfavorable():
    coords_a = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    elements_a = ["O", "N"]
    sasa_free_a = compute_sasa(coords_a, elements_a, n_points=200)

    coords_b_far = np.array([[500.0, 0.0, 0.0], [501.5, 0.0, 0.0]])
    coords_b_close = np.array([[3.2, 0.0, 0.0], [4.7, 0.0, 0.0]])
    elements_b = ["O", "N"]
    sasa_free_b = compute_sasa(coords_b_far, elements_b, n_points=200)

    energy_close = desolvation_energy(
        coords_a, elements_a, sasa_free_a, coords_b_close, elements_b, sasa_free_b, n_points=200,
    )
    assert energy_close > 0, f"burying polar atoms should be unfavorable, got {energy_close:.2f}"


def test_no_contact_gives_near_zero_energy():
    coords_a = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    elements_a = ["C", "C"]
    sasa_free_a = compute_sasa(coords_a, elements_a, n_points=200)

    coords_b = np.array([[500.0, 0.0, 0.0], [501.5, 0.0, 0.0]])
    elements_b = ["C", "C"]
    sasa_free_b = compute_sasa(coords_b, elements_b, n_points=200)

    energy = desolvation_energy(coords_a, elements_a, sasa_free_a, coords_b, elements_b, sasa_free_b, n_points=200)
    assert abs(energy) < 1.0, f"expected ~0 energy for non-contacting bodies, got {energy:.2f}"


def test_deeper_burial_gives_more_favorable_nonpolar_energy():
    coords_a = np.array([[0.0, 0.0, 0.0]])
    elements_a = ["C"]
    sasa_free_a = compute_sasa(coords_a, elements_a, n_points=200)

    coords_b_glancing = np.array([[3.4, 0.0, 0.0]])
    coords_b_deep = np.array([[3.2, 0.0, 0.0], [-3.2, 0.0, 0.0], [0.0, 3.2, 0.0], [0.0, -3.2, 0.0]])

    elements_b_1 = ["C"]
    elements_b_4 = ["C"] * 4
    sasa_free_b_1 = compute_sasa(np.array([[500.0, 0, 0]]), elements_b_1, n_points=200)
    sasa_free_b_4 = compute_sasa(coords_b_deep + np.array([500.0, 0, 0]), elements_b_4, n_points=200)

    energy_glancing = desolvation_energy(
        coords_a, elements_a, sasa_free_a, coords_b_glancing, elements_b_1, sasa_free_b_1, n_points=200,
    )
    energy_deep = desolvation_energy(
        coords_a, elements_a, sasa_free_a, coords_b_deep, elements_b_4, sasa_free_b_4, n_points=200,
    )
    assert energy_deep < energy_glancing, (
        f"deeper burial should be more favorable: glancing={energy_glancing:.2f}, "
        f"deep={energy_deep:.2f}"
    )


if __name__ == "__main__":
    test_burying_nonpolar_atoms_is_favorable()
    test_burying_polar_atoms_is_unfavorable()
    test_no_contact_gives_near_zero_energy()
    test_deeper_burial_gives_more_favorable_nonpolar_energy()
    print("All desolvation validation tests passed.")
