"""
triad.sampling.rotational_search
===================================
Generates and screens candidate rigid-body poses of the E3 ligase relative
to a fixed target, constrained by the degrader's reach distance (see
triad.topology.linker_geometry).

Scope of this first version, stated plainly: this treats "target protein +
its bound target-warhead fragment" as one fixed rigid body, and "ligase
protein + its bound ligase-warhead fragment" as one mobile rigid body, with
the flexible linker's own atoms excluded from both (its only role here is
supplying the reach-distance constraint between the two rigid bodies'
attachment points). Full linker conformational sampling is a real
simplification being deferred, not solved -- this version answers "does
constraining by reach distance plus clash-screening alone contain and favor
near-native poses at all," which needs answering before investing in linker
flexibility.

Clash screening here uses CA atoms only (not all-atom) for speed -- this is
a standard coarse-first strategy: cheap enough to screen thousands of
candidates, with the understanding that a promising candidate would get a
full-atom clash and interface re-score before being trusted (not yet built).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from triad.geometry.transforms import sample_rotations, fibonacci_sphere
from triad.scoring.clash import clash_score


@dataclass
class CandidatePose:
    rotation: np.ndarray       # (3,3) applied about mobile_pivot
    translation: np.ndarray    # (3,) -- where mobile_pivot ends up
    ca_clash_score: float


def generate_candidate_poses(
    mobile_ca_coords: np.ndarray,
    mobile_pivot: np.ndarray,
    fixed_ca_coords: np.ndarray,
    fixed_anchor: np.ndarray,
    reach_distance: float,
    n_axes: int = 30,
    n_angles_per_axis: int = 6,
    n_directions: int = 12,
) -> list[CandidatePose]:
    """Generate every (rotation, placement-direction) combination and score
    each by CA-level clash against the fixed body.

    For each candidate: the mobile body is rotated about `mobile_pivot`
    (which stays fixed under its own rotation), then translated so that
    `mobile_pivot` lands exactly `reach_distance` away from `fixed_anchor`,
    along a sampled direction. This directly encodes the reach constraint
    into every candidate generated, rather than filtering after the fact.

    Total candidates = (n_axes * n_angles_per_axis) * n_directions. Kept
    deliberately small in this first version (default ~2,160) for sandbox
    compute limits -- a production run would use a finer grid.
    """
    rotations = sample_rotations(n_axes=n_axes, n_angles_per_axis=n_angles_per_axis)
    directions = fibonacci_sphere(n_directions)

    mobile_centered = np.asarray(mobile_coords_for_pivot(mobile_ca_coords, mobile_pivot))

    candidates = []
    for R in rotations:
        rotated = mobile_centered @ R.T  # pivot stays at origin under its own rotation
        for d in directions:
            translation = fixed_anchor + reach_distance * d
            final_coords = rotated + translation
            score = clash_score(
                fixed_ca_coords, ["C"] * len(fixed_ca_coords),
                final_coords, ["C"] * len(final_coords),
            )
            candidates.append(CandidatePose(rotation=R, translation=translation, ca_clash_score=score))

    candidates.sort(key=lambda c: c.ca_clash_score)
    return candidates


def mobile_coords_for_pivot(coords: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    """Re-center coordinates so `pivot` sits at the origin -- required
    before applying a rotation matrix "about" that pivot.
    """
    return np.asarray(coords, dtype=np.float64) - pivot


def apply_pose(coords: np.ndarray, pivot: np.ndarray, pose: CandidatePose) -> np.ndarray:
    """Apply a CandidatePose's rotation+translation to an arbitrary
    coordinate set that shares the mobile body's original frame (e.g. the
    mobile body's all-atom coordinates, not just its CA atoms).
    """
    centered = mobile_coords_for_pivot(coords, pivot)
    return centered @ pose.rotation.T + pose.translation
