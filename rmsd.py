"""
triad.geometry.rmsd
====================
Optimal rigid-body superposition (Kabsch algorithm) and RMSD calculation.

This is the ground-truth metric the entire benchmark suite is built on: every
claim TRIAD makes about pose accuracy reduces to a call into this module.
It has to be correct and numerically stable, so it's kept small, dependency-light,
and covered by the round-trip test in tests/test_rmsd.py.
"""
from __future__ import annotations

import numpy as np

from triad.geometry.transforms import centroid


def kabsch(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the optimal rotation R and translation t that superpose
    `mobile` onto `reference` in the least-squares sense:

        reference ≈ mobile @ R.T + t

    Parameters
    ----------
    mobile, reference : (N, 3) arrays, same N, paired atom-to-atom
        (e.g. matching CA atoms by residue index).

    Returns
    -------
    R : (3, 3) proper rotation matrix
    t : (3,) translation vector

    Notes
    -----
    Handles the reflection case explicitly: SVD alone can return an improper
    rotation (det = -1) when the point sets are degenerate/planar or mirror
    images. We correct the sign of the last singular vector so R is always a
    proper rotation, matching the standard Kabsch fix.
    """
    mobile = np.asarray(mobile, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if mobile.shape != reference.shape:
        raise ValueError(
            f"mobile and reference must have the same shape, got "
            f"{mobile.shape} vs {reference.shape}"
        )
    if mobile.shape[0] < 3:
        raise ValueError("Kabsch superposition needs at least 3 paired points")

    mobile_c = centroid(mobile)
    ref_c = centroid(reference)
    mobile_centered = mobile - mobile_c
    ref_centered = reference - ref_c

    H = mobile_centered.T @ ref_centered
    U, S, Vt = np.linalg.svd(H)

    d = np.sign(np.linalg.det(Vt.T @ U.T))
    correction = np.diag([1.0, 1.0, d])
    R = Vt.T @ correction @ U.T

    t = ref_c - R @ mobile_c
    return R, t


def rmsd_after_alignment(mobile: np.ndarray, reference: np.ndarray) -> float:
    """Kabsch-align `mobile` onto `reference`, then return the RMSD.

    This is "RMSD after optimal superposition" — the standard structural-
    biology metric — not raw coordinate-difference RMSD, which is only
    meaningful if both structures already share a common frame.
    """
    R, t = kabsch(mobile, reference)
    aligned = mobile @ R.T + t
    diff = aligned - reference
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def raw_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD with no superposition — assumes both arrays are already in the
    same coordinate frame. Used to score a *proposed pose* against the native
    structure after a docking run, where the target chain has been held fixed
    as the reference frame and only the ligase/linker pose varies.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = a - b
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
