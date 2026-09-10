"""
Validates triad.scoring.shape_complementarity against known geometric
behavior before trusting it on real data: a nested (concave-convex) fit
should score higher than a tangential/off-center (poor) fit, and
non-contacting bodies should return None.
"""
import numpy as np

from triad.scoring.shape_complementarity import shape_complementarity


def test_returns_none_for_noncontacting_bodies():
    coords_a = np.array([[0.0, 0.0, 0.0]])
    coords_b = np.array([[500.0, 0.0, 0.0]])
    sc = shape_complementarity(coords_a, ["C"], coords_b, ["C"], n_points=100, interface_cutoff=3.0)
    assert sc is None


def test_nested_fit_scores_higher_than_tangential_fit():
    rng = np.random.default_rng(0)
    n_cup_atoms = 40
    theta = np.arccos(rng.uniform(0, 1, n_cup_atoms))
    phi = rng.uniform(0, 2 * np.pi, n_cup_atoms)
    R = 4.0
    cup_coords = np.stack(
        [R * np.sin(theta) * np.cos(phi), R * np.sin(theta) * np.sin(phi), R * np.cos(theta)], axis=1
    )
    cup_elements = ["C"] * n_cup_atoms

    ball_nested = np.array([[0.0, 0.0, 0.0]])
    ball_offset = np.array([[6.0, 0.0, 3.5]])

    sc_nested = shape_complementarity(cup_coords, cup_elements, ball_nested, ["C"], n_points=200, interface_cutoff=5.0)
    sc_offset = shape_complementarity(cup_coords, cup_elements, ball_offset, ["C"], n_points=200, interface_cutoff=5.0)

    assert sc_nested is not None and sc_offset is not None
    assert sc_nested > sc_offset, (
        f"nested fit ({sc_nested:.3f}) should score higher than tangential fit ({sc_offset:.3f})"
    )


def test_prefiltering_optimization_does_not_change_result():
    """Regression test: the atom-prefiltering optimization added for
    real-data performance must not change the computed value.
    """
    rng = np.random.default_rng(0)
    n_cup_atoms = 40
    theta = np.arccos(rng.uniform(0, 1, n_cup_atoms))
    phi = rng.uniform(0, 2 * np.pi, n_cup_atoms)
    R = 4.0
    cup_coords = np.stack(
        [R * np.sin(theta) * np.cos(phi), R * np.sin(theta) * np.sin(phi), R * np.cos(theta)], axis=1
    )
    cup_elements = ["C"] * n_cup_atoms
    ball_coords = np.array([[0.0, 0.0, 0.0]])

    sc = shape_complementarity(cup_coords, cup_elements, ball_coords, ["C"], n_points=200, interface_cutoff=5.0)
    np.testing.assert_allclose(sc, 0.160, atol=0.001)


if __name__ == "__main__":
    test_returns_none_for_noncontacting_bodies()
    test_nested_fit_scores_higher_than_tangential_fit()
    test_prefiltering_optimization_does_not_change_result()
    print("All shape complementarity validation tests passed.")
