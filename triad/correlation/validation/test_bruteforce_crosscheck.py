"""
Cross-check triad.correlation.fft_dock.fft_correlate_3d against
brute_force_correlate_3d on small grids. This is the single most important
test in the Phase 2 validation sequence (per docs/TRIAD_v1_MANIFEST.md Part
2.3, step 3): a subtle indexing, conjugation, or reflection error in a
from-scratch FFT correlation implementation would NOT crash -- it would
silently produce a plausible-looking correlation grid with the peak in the
wrong place, corrupting every docking result built on top of it.
"""
import numpy as np

from triad.correlation.fft_dock import fft_correlate_3d, brute_force_correlate_3d


def test_fft_matches_bruteforce_random_grid():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 8, 8))
    B = rng.normal(size=(8, 8, 8))

    C_fft = fft_correlate_3d(A, B)
    C_brute = brute_force_correlate_3d(A, B)

    np.testing.assert_allclose(C_fft, C_brute, atol=1e-8)


def test_fft_matches_bruteforce_asymmetric_grid():
    """Non-cubic grid -- catches bugs that only manifest when axes have
    different lengths (e.g. an axis-order mixup in the shift/roll logic).
    """
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 10, 8))
    B = rng.normal(size=(6, 10, 8))

    C_fft = fft_correlate_3d(A, B)
    C_brute = brute_force_correlate_3d(A, B)

    np.testing.assert_allclose(C_fft, C_brute, atol=1e-8)


def test_fft_matches_bruteforce_sparse_binary_grid():
    """Binary occupancy-style grids (mostly zeros, a few ones) -- the actual
    kind of grid this will see in real shape-complementarity docking, as
    opposed to dense random noise.
    """
    rng = np.random.default_rng(2)
    A = np.zeros((8, 8, 8))
    B = np.zeros((8, 8, 8))
    for _ in range(5):
        A[tuple(rng.integers(0, 8, size=3))] = 1.0
        B[tuple(rng.integers(0, 8, size=3))] = 1.0

    C_fft = fft_correlate_3d(A, B)
    C_brute = brute_force_correlate_3d(A, B)

    np.testing.assert_allclose(C_fft, C_brute, atol=1e-8)


def test_self_correlation_peak_is_at_zero_translation():
    """Correlating a grid with itself: C[0] should equal sum(A^2), the
    maximum possible value (Cauchy-Schwarz), at translation (0,0,0) -- a
    shape perfectly overlapping itself with no shift is the best possible
    alignment. A physically meaningful check, not just a numerical one.
    """
    rng = np.random.default_rng(3)
    A = rng.random(size=(10, 10, 10))  # nonnegative, like an occupancy grid

    C = fft_correlate_3d(A, A)
    peak_index = np.unravel_index(np.argmax(C), C.shape)

    assert peak_index == (0, 0, 0), f"expected self-correlation peak at origin, got {peak_index}"
    np.testing.assert_allclose(C[0, 0, 0], np.sum(A ** 2), atol=1e-8)


if __name__ == "__main__":
    test_fft_matches_bruteforce_random_grid()
    test_fft_matches_bruteforce_asymmetric_grid()
    test_fft_matches_bruteforce_sparse_binary_grid()
    test_self_correlation_peak_is_at_zero_translation()
    print("All FFT-vs-brute-force cross-check tests passed.")
