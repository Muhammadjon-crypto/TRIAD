"""
Validation of triad.sampling.local_refinement on a synthetic case with a
KNOWN correct answer: two point clouds placed with a deliberate clash and a
deliberately wrong reach distance, refined toward the unique configuration
that resolves both. If the optimizer can't solve this simple, exactly-known
case, it has no business touching real docking candidates.
"""
import numpy as np

from triad.sampling.local_refinement import local_refine, rotation_from_rotvec


def test_rotvec_identity_at_zero():
    R = rotation_from_rotvec(np.zeros(3))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-10)


def test_rotvec_is_proper_rotation():
    rng = np.random.default_rng(0)
    omega = rng.normal(size=3) * 0.5
    R = rotation_from_rotvec(omega)
    assert abs(np.linalg.det(R) - 1.0) < 1e-10
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


def test_refinement_resolves_synthetic_clash_and_reach_violation():
    """Two small point clouds: fixed at origin, mobile initially placed
    clashing into it AND at the wrong distance from a target anchor.
    Refinement should reduce clash to ~0 and reach deviation to ~0.
    """
    rng = np.random.default_rng(42)

    fixed_coords = rng.normal(size=(20, 3)) * 3.0
    fixed_elements = ["C"] * 20

    mobile_local = rng.normal(size=(20, 3)) * 3.0
    mobile_pivot_local = mobile_local.mean(axis=0)
    mobile_elements = ["C"] * 20

    initial_rotation = np.eye(3)
    initial_pivot = np.array([2.0, 0.0, 0.0])  # clashes with fixed body centered at origin

    target_anchor = np.array([20.0, 0.0, 0.0])
    reach_distance = 15.0  # initial pivot is far from the correct distance

    result = local_refine(
        initial_rotation=initial_rotation,
        initial_pivot=initial_pivot,
        mobile_coords=mobile_local,
        mobile_pivot_local=mobile_pivot_local,
        mobile_elements=mobile_elements,
        fixed_coords=fixed_coords,
        fixed_elements=fixed_elements,
        target_anchor=target_anchor,
        reach_distance=reach_distance,
        n_steps=500, learning_rate=0.05, k_reach=5.0,
    )

    assert result.final_clash_score < 1.0, (
        f"expected clash to be resolved, got {result.final_clash_score:.3f}"
    )
    assert abs(result.final_reach_deviation) < 0.5, (
        f"expected reach constraint satisfied, got deviation "
        f"{result.final_reach_deviation:.3f} A"
    )
    assert result.objective_history[-1] < result.objective_history[0] * 0.1, (
        f"objective should have decreased substantially: "
        f"{result.objective_history[0]:.2f} -> {result.objective_history[-1]:.2f}"
    )


def test_refinement_converges_from_multiple_starting_orientations():
    """Confirms the optimizer isn't accidentally dependent on a lucky
    starting rotation -- refinement from several different random starting
    orientations should all end up with low clash and satisfied reach.
    """
    rng = np.random.default_rng(1)
    fixed_coords = rng.normal(size=(15, 3)) * 3.0
    fixed_elements = ["C"] * 15
    mobile_local = rng.normal(size=(15, 3)) * 3.0
    mobile_pivot_local = mobile_local.mean(axis=0)
    mobile_elements = ["C"] * 15
    target_anchor = np.array([20.0, 0.0, 0.0])
    reach_distance = 15.0

    from triad.geometry.transforms import random_rotation

    for seed in range(3):
        r = np.random.default_rng(seed)
        initial_rotation = random_rotation(r)
        initial_pivot = np.array([5.0, 0.0, 0.0])

        result = local_refine(
            initial_rotation=initial_rotation, initial_pivot=initial_pivot,
            mobile_coords=mobile_local, mobile_pivot_local=mobile_pivot_local,
            mobile_elements=mobile_elements, fixed_coords=fixed_coords,
            fixed_elements=fixed_elements, target_anchor=target_anchor,
            reach_distance=reach_distance, n_steps=500, learning_rate=0.05,
        )
        assert result.final_clash_score < 2.0, f"seed {seed}: clash not resolved"
        assert abs(result.final_reach_deviation) < 1.0, f"seed {seed}: reach not satisfied"


if __name__ == "__main__":
    test_rotvec_identity_at_zero()
    test_rotvec_is_proper_rotation()
    test_refinement_resolves_synthetic_clash_and_reach_violation()
    test_refinement_converges_from_multiple_starting_orientations()
    print("All local refinement validation tests passed.")
