"""
Analytical validation of triad.scoring.sasa against known exact answers.
This must pass before SASA is trusted for any real pose scoring -- the same
discipline applied to every other numerical module this session.
"""
import numpy as np

from triad.scoring.sasa import compute_sasa, buried_surface_area, PROBE_RADIUS
from triad.scoring.clash import VDW_RADII


def test_isolated_atom_sasa_matches_full_sphere_area():
    """A single atom with no neighbors should be 100% solvent-accessible:
    its SASA should equal the full sphere area 4*pi*(vdw+probe)^2 exactly
    (to within point-sampling error).
    """
    coords = np.array([[0.0, 0.0, 0.0]])
    elements = ["C"]
    sasa = compute_sasa(coords, elements, n_points=500)

    expected_radius = VDW_RADII["C"] + PROBE_RADIUS
    expected_area = 4 * np.pi * expected_radius ** 2

    assert abs(sasa[0] - expected_area) / expected_area < 0.02, (
        f"isolated atom SASA {sasa[0]:.3f} should match full sphere area "
        f"{expected_area:.3f} (within 2% sampling error)"
    )


def test_two_atoms_far_apart_both_fully_accessible():
    """Two atoms far enough apart to not occlude each other at all should
    each show full sphere accessibility, same as if isolated.
    """
    coords = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    elements = ["C", "C"]
    sasa = compute_sasa(coords, elements, n_points=500)

    expected_radius = VDW_RADII["C"] + PROBE_RADIUS
    expected_area = 4 * np.pi * expected_radius ** 2

    for s in sasa:
        assert abs(s - expected_area) / expected_area < 0.02


def test_two_overlapping_atoms_show_reduced_sasa():
    """Two atoms close enough to overlap should show LESS than full sphere
    area each (mutual occlusion), and the reduction should be substantial
    for significant overlap.
    """
    # place two carbons 2.0 A apart -- well within 2x(vdw+probe)=6.2A, so
    # substantial mutual occlusion is expected
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = ["C", "C"]
    sasa = compute_sasa(coords, elements, n_points=500)

    expected_radius = VDW_RADII["C"] + PROBE_RADIUS
    full_area = 4 * np.pi * expected_radius ** 2

    for s in sasa:
        assert s < full_area * 0.8, (
            f"overlapping atom SASA {s:.3f} should be well below full "
            f"sphere area {full_area:.3f} due to mutual occlusion"
        )


def test_buried_surface_area_is_zero_for_distant_bodies():
    """Two atom groups far apart should show ~zero buried surface area --
    they don't touch, so nothing is occluded between them.
    """
    coords_a = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    coords_b = np.array([[500.0, 0.0, 0.0], [501.5, 0.0, 0.0]])
    elements = ["C", "C"]

    bsa = buried_surface_area(coords_a, elements, coords_b, elements, n_points=300)
    assert abs(bsa) < 1.0, f"expected ~0 buried surface area for distant bodies, got {bsa:.3f}"


def test_buried_surface_area_is_positive_for_touching_bodies():
    """Two atom groups placed in contact should show clearly positive
    buried surface area.
    """
    coords_a = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    coords_b = np.array([[3.2, 0.0, 0.0], [4.7, 0.0, 0.0]])  # ~3.2A gap, within contact range
    elements = ["C", "C"]

    bsa = buried_surface_area(coords_a, elements, coords_b, elements, n_points=300)
    assert bsa > 5.0, f"expected clearly positive buried surface area for contacting bodies, got {bsa:.3f}"


if __name__ == "__main__":
    test_isolated_atom_sasa_matches_full_sphere_area()
    test_two_atoms_far_apart_both_fully_accessible()
    test_two_overlapping_atoms_show_reduced_sasa()
    test_buried_surface_area_is_zero_for_distant_bodies()
    test_buried_surface_area_is_positive_for_touching_bodies()
    print("All SASA analytical validation tests passed.")
