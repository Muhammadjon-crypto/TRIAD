"""
Validates triad.correlation.contiguous_patch against synthetic geometry
before trusting it on real data: a genuine contiguous interface patch
must score higher than scattered fragments with an IDENTICAL total
overlap voxel count, since raw overlap counting alone cannot distinguish
these (manifest Part 22's diagnosed failure mode).
"""
import numpy as np

from triad.correlation.contiguous_patch import largest_contiguous_overlap, contiguous_patch_score


def test_contiguous_patch_scores_higher_than_scattered_at_equal_overlap():
    grid_shape = (20, 20, 20)

    receptor_a = np.zeros(grid_shape)
    receptor_a[5:9, 5:9, 5:7] = 1.0
    ligand_a = np.zeros(grid_shape)
    ligand_a[5:9, 5:9, 5:7] = 1.0

    receptor_b = np.zeros(grid_shape)
    ligand_b = np.zeros(grid_shape)
    scattered_points = [(1,1,1),(1,1,15),(1,15,1),(15,1,1),(15,15,1),(15,1,15),(1,15,15),(15,15,15)]
    for x, y, z in scattered_points:
        receptor_b[x:x+2, y:y+2, z:z+1] = 1.0
        ligand_b[x:x+2, y:y+2, z:z+1] = 1.0

    total_a = np.sum((receptor_a > 0) & (ligand_a > 0))
    total_b = np.sum((receptor_b > 0) & (ligand_b > 0))
    assert total_a == total_b, "test setup requires identical raw overlap counts"

    raw_score = 100.0
    rescored_a = contiguous_patch_score(receptor_a, ligand_a, raw_score)
    rescored_b = contiguous_patch_score(receptor_b, ligand_b, raw_score)
    assert rescored_a > rescored_b


def test_largest_component_matches_expected_size():
    grid_shape = (10, 10, 10)
    receptor = np.zeros(grid_shape)
    ligand = np.zeros(grid_shape)
    receptor[2:5, 2:5, 2:4] = 1.0  # 3x3x2 = 18 voxels
    ligand[2:5, 2:5, 2:4] = 1.0
    assert largest_contiguous_overlap(receptor, ligand) == 18


def test_no_overlap_gives_zero():
    grid_shape = (10, 10, 10)
    receptor = np.zeros(grid_shape)
    ligand = np.zeros(grid_shape)
    receptor[0:2, 0:2, 0:2] = 1.0
    ligand[8:10, 8:10, 8:10] = 1.0
    assert largest_contiguous_overlap(receptor, ligand) == 0


if __name__ == "__main__":
    test_contiguous_patch_scores_higher_than_scattered_at_equal_overlap()
    test_largest_component_matches_expected_size()
    test_no_overlap_gives_zero()
    print("Contiguous patch validation passed.")
