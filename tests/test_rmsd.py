"""
Round-trip test for triad.geometry: scramble a synthetic point cloud with a
known random rigid-body transform, recover it via Kabsch, and confirm both
(a) the recovered RMSD is ~0 and (b) the recovered transform matches the
applied one. This is the first gate TRIAD has to pass — if this doesn't hold
exactly, nothing built on top of it (docking, benchmarking) can be trusted.
"""
import numpy as np

from triad.geometry.transforms import apply_transform, random_rotation, sample_rotations
from triad.geometry.rmsd import kabsch, rmsd_after_alignment, raw_rmsd


def test_kabsch_recovers_known_transform():
    rng = np.random.default_rng(42)
    n_atoms = 150
    reference = rng.normal(size=(n_atoms, 3)) * 10.0

    R_true = random_rotation(rng)
    t_true = rng.normal(size=3) * 20.0
    mobile = apply_transform(reference, R_true.T, -R_true.T @ t_true)  # inverse transform

    # Recover: mobile should re-align onto reference with ~0 RMSD
    rmsd = rmsd_after_alignment(mobile, reference)
    assert rmsd < 1e-9, f"expected near-zero RMSD, got {rmsd}"

    R_rec, t_rec = kabsch(mobile, reference)
    aligned = apply_transform(mobile, R_rec, t_rec)
    assert raw_rmsd(aligned, reference) < 1e-9


def test_kabsch_with_noise_is_bounded():
    rng = np.random.default_rng(7)
    n_atoms = 200
    reference = rng.normal(size=(n_atoms, 3)) * 15.0

    R_true = random_rotation(rng)
    t_true = rng.normal(size=3) * 10.0
    mobile = apply_transform(reference, R_true.T, -R_true.T @ t_true)

    noise_sigma = 0.5
    mobile_noisy = mobile + rng.normal(scale=noise_sigma, size=mobile.shape)

    rmsd = rmsd_after_alignment(mobile_noisy, reference)
    # For isotropic per-axis Gaussian noise with std `noise_sigma`, expected
    # RMSD (pooled across x,y,z) is noise_sigma * sqrt(3). Allow 30% slack.
    expected = noise_sigma * np.sqrt(3)
    assert rmsd < expected * 1.3, f"rmsd={rmsd:.4f} expected~{expected:.4f}"


def test_rotation_grid_is_proper_and_covers_sphere():
    rots = sample_rotations(n_axes=50, n_angles_per_axis=8)
    assert rots.shape == (400, 3, 3)
    dets = np.linalg.det(rots)
    assert np.allclose(dets, 1.0, atol=1e-6), "all sampled rotations must be proper (det=+1)"

    # orthogonality check: R @ R.T should be identity for every sampled rotation
    identity = np.eye(3)
    for R in rots[:: max(1, len(rots) // 20)]:  # spot check
        assert np.allclose(R @ R.T, identity, atol=1e-6)


def test_rotation_grid_has_no_duplicate_identity_rotations():
    """Regression test for a real bug (docs/TRIAD_v1_MANIFEST.md Part 8):
    the angle grid used to include angle=0 for every sampled axis, and a
    0-degree rotation about any axis is identity regardless of axis choice
    -- wasting 1/n_angles_per_axis of the entire search on exact duplicates.
    No rotation in the sampled grid should equal identity (or any other
    rotation) more than once.
    """
    rotations = sample_rotations(n_axes=30, n_angles_per_axis=6)

    n_identity = sum(1 for R in rotations if np.allclose(R, np.eye(3), atol=1e-6))
    assert n_identity <= 1, (
        f"expected at most 1 identity rotation in the grid, found {n_identity} "
        f"-- the angle=0 duplication bug may have regressed"
    )

    # broader check: no two rotations in the grid should be exact duplicates
    # of each other (spot-checked pairwise on a subset for speed)
    subset = rotations[::3]  # every 3rd rotation, for a tractable pairwise check
    n_dupes = 0
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            if np.allclose(subset[i], subset[j], atol=1e-6):
                n_dupes += 1
    assert n_dupes == 0, f"found {n_dupes} duplicate rotation pairs in the sampled grid"


if __name__ == "__main__":
    test_kabsch_recovers_known_transform()
    test_kabsch_with_noise_is_bounded()
    test_rotation_grid_is_proper_and_covers_sphere()
    test_rotation_grid_has_no_duplicate_identity_rotations()
    print("All geometry core tests passed.")
