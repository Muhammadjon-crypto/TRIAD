"""
Two-sphere synthetic docking test: place a sphere at a KNOWN voxel offset
from a reference/template sphere, run FFT correlation, and confirm the
correlation peak lands exactly at the known correct translation. This is
the final gate in the Phase 2 validation sequence (docs/TRIAD_v1_MANIFEST.md
Part 2.3, step 1) before any real protein data touches this machinery.
"""
import numpy as np

from triad.correlation.grid import voxelize_sphere
from triad.correlation.fft_dock import fft_correlate_3d, find_best_translation_index


def test_peak_recovers_known_offset_simple_case():
    grid_shape = (32, 32, 32)
    radius = 4.0

    template = voxelize_sphere(grid_shape, center=(0, 0, 0), radius_voxels=radius)
    known_offset = (10, 15, 20)
    target = voxelize_sphere(grid_shape, center=known_offset, radius_voxels=radius)

    C = fft_correlate_3d(target, template)
    peak = find_best_translation_index(C)

    assert peak == known_offset, f"expected peak at {known_offset}, got {peak}"


def test_peak_recovers_known_offset_various_positions():
    """Repeat with several different offsets and radii to make sure the
    first case wasn't a lucky coincidence of the specific numbers chosen.
    """
    grid_shape = (32, 32, 32)
    test_cases = [
        ((3, 3, 3), 2.0),
        ((25, 4, 17), 3.0),
        ((0, 0, 16), 5.0),
        ((31, 31, 31), 2.5),  # near the wraparound edge -- worth stressing
        ((16, 16, 16), 6.0),
    ]
    for offset, radius in test_cases:
        template = voxelize_sphere(grid_shape, center=(0, 0, 0), radius_voxels=radius)
        target = voxelize_sphere(grid_shape, center=offset, radius_voxels=radius)
        C = fft_correlate_3d(target, template)
        peak = find_best_translation_index(C)
        assert peak == offset, f"offset={offset}, radius={radius}: expected peak {offset}, got {peak}"


def test_peak_value_equals_full_sphere_overlap():
    """At the correct offset, correlation value should equal the number of
    voxels in the sphere (perfect overlap of two identical-radius spheres) --
    a stronger, quantitative check than just "peak is in the right place."
    """
    grid_shape = (32, 32, 32)
    radius = 4.0
    template = voxelize_sphere(grid_shape, center=(0, 0, 0), radius_voxels=radius)
    sphere_volume_voxels = template.sum()

    offset = (12, 8, 5)
    target = voxelize_sphere(grid_shape, center=offset, radius_voxels=radius)

    C = fft_correlate_3d(target, template)
    peak_value = C[offset]

    np.testing.assert_allclose(peak_value, sphere_volume_voxels, atol=1e-8)


def test_non_overlapping_offset_gives_lower_correlation():
    """A translation that does NOT align the spheres should score
    meaningfully lower than the correct one.
    """
    grid_shape = (32, 32, 32)
    radius = 4.0
    template = voxelize_sphere(grid_shape, center=(0, 0, 0), radius_voxels=radius)
    correct_offset = (10, 10, 10)
    target = voxelize_sphere(grid_shape, center=correct_offset, radius_voxels=radius)

    C = fft_correlate_3d(target, template)
    correct_score = C[correct_offset]
    wrong_score = C[(0, 0, 0)]

    assert correct_score > wrong_score, (
        f"correct alignment ({correct_score}) should score higher than "
        f"misaligned ({wrong_score})"
    )


if __name__ == "__main__":
    test_peak_recovers_known_offset_simple_case()
    test_peak_recovers_known_offset_various_positions()
    test_peak_value_equals_full_sphere_overlap()
    test_non_overlapping_offset_gives_lower_correlation()
    print("All two-sphere synthetic docking tests passed.")
