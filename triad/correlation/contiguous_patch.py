"""
triad.correlation.contiguous_patch
======================================
Addresses the local-patch failure mode diagnosed in
docs/TRIAD_v1_MANIFEST.md Part 22 and confirmed by the buried-surface-area
refutation in Part 24: the shape channel's raw correlation score sums
reward across every overlapping voxel regardless of whether those voxels
form one real, contiguous interface patch or several small, scattered,
coincidental touches spread across a large but spurious contact.

This computes, for a given ligand placement, the SIZE OF THE LARGEST
CONNECTED COMPONENT of overlapping (reward) voxels between receptor and
ligand shape grids, using scipy.ndimage.label. A genuine protein-protein
interface is one contiguous patch; a spurious high-shape-score pose from
an incorrect rotation can rack up a similar total overlap COUNT while that
overlap is fragmented across several small, disconnected patches that
individually look nothing like a real interface.

This is a per-candidate check (not yet an FFT-accelerated channel): it
must be applied to a shortlist of candidates already ranked by the
existing correlation score, not to the entire grid at once, since
connected-component labeling does not have an FFT-equivalent fast
formulation the way simple overlap counting does.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def largest_contiguous_overlap(
    receptor_shape_grid: np.ndarray, ligand_shape_grid: np.ndarray,
) -> int:
    """Size (in voxels) of the largest connected component of simultaneous
    receptor-reward and ligand-occupancy overlap, for a SINGLE, already-
    aligned placement (both grids in the same frame, no translation search
    -- this checks one candidate, not all of them at once).

    Reward voxels are those where receptor_shape_grid > 0 (the surface
    shell, not the interior-penalty region) AND ligand_shape_grid > 0
    (ligand present). Connectivity uses the full 26-neighbor 3D structure
    element, matching the moderate degree of atomic-scale surface
    roughness expected at a real interface.
    """
    overlap_mask = (receptor_shape_grid > 0) & (ligand_shape_grid > 0)
    if not overlap_mask.any():
        return 0
    structure = np.ones((3, 3, 3), dtype=bool)
    labeled, n_components = ndimage.label(overlap_mask, structure=structure)
    if n_components == 0:
        return 0
    component_sizes = ndimage.sum(overlap_mask, labeled, index=range(1, n_components + 1))
    return int(component_sizes.max())


def contiguous_patch_score(
    receptor_shape_grid: np.ndarray, ligand_shape_grid: np.ndarray,
    raw_shape_score: float, min_patch_fraction: float = 0.3,
) -> float:
    """Rescale a raw shape correlation score by how much of the total
    overlap belongs to its single largest contiguous patch. A pose whose
    overlap is one real, mostly-contiguous interface keeps close to its
    raw score; a pose whose overlap is fragmented across many small,
    scattered touches is penalized toward zero, since that pattern is the
    signature of the spurious local-coincidence failure mode, not a real
    interface.
    """
    total_overlap = np.sum((receptor_shape_grid > 0) & (ligand_shape_grid > 0))
    if total_overlap == 0:
        return 0.0
    largest = largest_contiguous_overlap(receptor_shape_grid, ligand_shape_grid)
    contiguous_fraction = largest / total_overlap
    if contiguous_fraction >= min_patch_fraction:
        return raw_shape_score
    # linear falloff below the threshold rather than a hard cutoff, so the
    # score degrades smoothly rather than creating a new hard discontinuity
    return raw_shape_score * (contiguous_fraction / min_patch_fraction)
