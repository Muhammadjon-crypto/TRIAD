"""
triad.correlation.search
===========================
The full rotation-loop driver: for each candidate rotation (from
triad.geometry.transforms.sample_rotations), builds fresh ligand shape and
electrostatic grids, FFT-correlates against precomputed receptor grids,
masks to the reach-constrained sphere, and tracks the best-scoring pose
found across all rotations.

Key design choice: the ligand is rotated about its own attachment point
(reach.ligase_warhead_centroid), and the rotated template is embedded with
that attachment point placed exactly at the target's attachment point
(fixed_anchor). This makes the reach constraint reduce to a simple
`|tau| ~= reach_distance` mask, computed once and reused across every
rotation, rather than recomputed per-rotation.

VERIFIED FINDING (docs/TRIAD_v1_MANIFEST.md Part 7): the pose-application
math here is exactly correct (confirmed to machine precision against the
true native pose). The search infrastructure itself has no known bugs.
Discrimination power (shape+electrostatics ranking the true native pose
correctly) remains the limiting factor, confirmed independently here and
in triad.correlation.validation.test_real_data_5t35 -- this module should
NOT be "fixed" by adding more rotations or finer grids without first
addressing scoring discrimination (see manifest Part 7 for the reasoning).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from triad.correlation.channels import build_ligand_shape_grid, build_charge_grid
from triad.correlation.fft_dock import fft_correlate_3d
from triad.geometry.transforms import sample_rotations


@dataclass
class SearchResult:
    best_rotation: np.ndarray
    best_translation: np.ndarray
    best_score: float
    n_rotations_tried: int


def build_reach_mask(
    grid_shape: tuple[int, int, int], spacing: float, reach_distance: float,
    tolerance: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute the physical-tau grid and the boolean reach-constraint
    mask (|tau| within `tolerance` of `reach_distance`) ONCE -- reused
    across every rotation tried, since with the attachment-point-embedding
    convention this doesn't depend on rotation at all.
    """
    nx, ny, nz = grid_shape
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    tau = np.stack([ix * spacing, iy * spacing, iz * spacing], axis=-1)
    for d, n in enumerate(grid_shape):
        half = n * spacing / 2
        tau[..., d] = np.where(tau[..., d] > half, tau[..., d] - n * spacing, tau[..., d])
    tau_mag = np.linalg.norm(tau, axis=-1)
    mask = np.abs(tau_mag - reach_distance) <= tolerance
    return tau, mask


def search_rotations(
    lig_coords: np.ndarray, lig_radii: np.ndarray, lig_charges: np.ndarray,
    mobile_anchor: np.ndarray, fixed_anchor: np.ndarray,
    shape_receptor_grid: np.ndarray, potential_receptor_grid: np.ndarray,
    grid_shape: tuple[int, int, int], origin: np.ndarray, spacing: float,
    reach_distance: float, electrostatic_weight: float = 0.1,
    n_axes: int = 30, n_angles_per_axis: int = 6,
) -> SearchResult:
    """Full rotation-loop search: try every rotation in the sample grid,
    FFT-correlate each against the fixed receptor grids, keep the best
    reach-constrained pose found overall.
    """
    tau, reach_mask = build_reach_mask(grid_shape, spacing, reach_distance)
    lig_centered = lig_coords - mobile_anchor
    rotations = sample_rotations(n_axes=n_axes, n_angles_per_axis=n_angles_per_axis)

    best = SearchResult(
        best_rotation=np.eye(3), best_translation=np.zeros(3),
        best_score=-np.inf, n_rotations_tried=0,
    )

    for R in rotations:
        rotated = lig_centered @ R.T
        embedded = rotated + fixed_anchor

        shape_l = build_ligand_shape_grid(embedded, lig_radii, grid_shape, origin, spacing)
        charge_l = build_charge_grid(embedded, lig_charges, grid_shape, origin, spacing)

        C_shape = fft_correlate_3d(shape_receptor_grid, shape_l)
        C_elec = -fft_correlate_3d(potential_receptor_grid, charge_l)
        C_combined = C_shape + electrostatic_weight * C_elec
        C_masked = np.where(reach_mask, C_combined, -np.inf)

        idx = np.unravel_index(np.argmax(C_masked), C_masked.shape)
        score = C_masked[idx]
        if score > best.best_score:
            best = SearchResult(
                best_rotation=R, best_translation=tau[idx],
                best_score=float(score), n_rotations_tried=best.n_rotations_tried,
            )
        best.n_rotations_tried += 1

    return best


def apply_search_result(
    coords: np.ndarray, mobile_anchor: np.ndarray, fixed_anchor: np.ndarray,
    result: SearchResult,
) -> np.ndarray:
    """Apply a SearchResult's rotation+translation to any coordinate set
    that shares the mobile body's original frame (e.g. CA atoms for RMSD
    scoring, or the full atom set for output).
    """
    return (coords - mobile_anchor) @ result.best_rotation.T + fixed_anchor + result.best_translation
