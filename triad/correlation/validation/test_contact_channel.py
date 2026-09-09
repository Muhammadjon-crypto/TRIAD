"""
Validates triad.correlation.contact_channel's linearity-trick
implementation against a direct brute-force computation, before trusting
it on any real docking score.
"""
import numpy as np

from triad.correlation.fft_dock import fft_correlate_3d
from triad.correlation.contact_channel import build_residue_type_grids, contact_potential_correlation
from triad.potentials.contact_potential import ContactPotential, STANDARD_AA


def _brute_force_reference(t_grids, l_grids, potential, grid_shape):
    result = np.zeros(grid_shape)
    for aa_i in STANDARD_AA:
        for aa_j in STANDARD_AA:
            w = potential.get(aa_i, aa_j)
            if w != 0.0:
                result += w * fft_correlate_3d(t_grids[aa_i], l_grids[aa_j])
    return result


def test_linearity_trick_matches_bruteforce_small_case():
    rng = np.random.default_rng(0)
    grid_shape = (10, 10, 10)
    spacing, origin = 1.0, np.zeros(3)

    t_coords = rng.uniform(1, 9, size=(6, 3))
    t_resnames = ["LEU", "ASP", "LEU", "LYS", "GLY", "ASP"]
    t_radii = np.full(6, 1.0)
    l_coords = rng.uniform(1, 9, size=(5, 3))
    l_resnames = ["ILE", "LYS", "ASP", "LEU", "GLY"]
    l_radii = np.full(5, 1.0)

    potential = ContactPotential(
        scores={
            tuple(sorted(["LEU", "ILE"])): 2.0,
            tuple(sorted(["ASP", "LYS"])): -1.5,
            tuple(sorted(["LEU", "LEU"])): 3.0,
            tuple(sorted(["GLY", "GLY"])): 0.5,
        },
        n_contacts_observed=100, n_structures=15,
    )

    t_grids = build_residue_type_grids(t_coords, t_resnames, t_radii, grid_shape, origin, spacing)
    l_grids = build_residue_type_grids(l_coords, l_resnames, l_radii, grid_shape, origin, spacing)

    fast = contact_potential_correlation(t_grids, l_grids, potential)
    brute = _brute_force_reference(t_grids, l_grids, potential, grid_shape)

    np.testing.assert_allclose(fast, brute, atol=1e-8)


def test_linearity_trick_matches_bruteforce_dense_potential():
    """Repeat with a denser, more realistic potential (many nonzero
    entries) to make sure the trick holds generally, not just for a
    sparse special case.
    """
    rng = np.random.default_rng(1)
    grid_shape = (8, 8, 8)
    spacing, origin = 1.0, np.zeros(3)

    t_coords = rng.uniform(1, 7, size=(10, 3))
    t_resnames = list(rng.choice(STANDARD_AA, size=10))
    t_radii = np.full(10, 1.0)
    l_coords = rng.uniform(1, 7, size=(8, 3))
    l_resnames = list(rng.choice(STANDARD_AA, size=8))
    l_radii = np.full(8, 1.0)

    scores = {}
    for i, aa_i in enumerate(STANDARD_AA):
        for aa_j in STANDARD_AA[i:]:
            scores[tuple(sorted([aa_i, aa_j]))] = rng.normal()
    potential = ContactPotential(scores=scores, n_contacts_observed=1000, n_structures=15)

    t_grids = build_residue_type_grids(t_coords, t_resnames, t_radii, grid_shape, origin, spacing)
    l_grids = build_residue_type_grids(l_coords, l_resnames, l_radii, grid_shape, origin, spacing)

    fast = contact_potential_correlation(t_grids, l_grids, potential)
    brute = _brute_force_reference(t_grids, l_grids, potential, grid_shape)

    np.testing.assert_allclose(fast, brute, atol=1e-6)


if __name__ == "__main__":
    test_linearity_trick_matches_bruteforce_small_case()
    test_linearity_trick_matches_bruteforce_dense_potential()
    print("Contact potential correlation channel validated against brute force.")
