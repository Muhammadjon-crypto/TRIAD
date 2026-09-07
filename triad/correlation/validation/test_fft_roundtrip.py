"""
FFT round-trip test: IFFT(FFT(x)) ~= x for random 3D grids of various
shapes, including non-cubic and odd-sized grids (some FFT bugs only show up
for non-power-of-2 or asymmetric dimensions). This is the first, simplest
gate in the Phase 2 validation sequence -- if this doesn't hold exactly
(to numerical precision), nothing built on top of FFT correlation can be
trusted.
"""
import numpy as np


def test_roundtrip_cubic_power_of_two():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 16, 16))
    recovered = np.fft.ifftn(np.fft.fftn(x)).real
    np.testing.assert_allclose(recovered, x, atol=1e-10)


def test_roundtrip_non_power_of_two():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(13, 17, 11))  # deliberately awkward, non-power-of-2 dims
    recovered = np.fft.ifftn(np.fft.fftn(x)).real
    np.testing.assert_allclose(recovered, x, atol=1e-10)


def test_roundtrip_asymmetric_cuboid():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(32, 8, 24))
    recovered = np.fft.ifftn(np.fft.fftn(x)).real
    np.testing.assert_allclose(recovered, x, atol=1e-10)


def test_roundtrip_preserves_zero_grid():
    x = np.zeros((10, 10, 10))
    recovered = np.fft.ifftn(np.fft.fftn(x)).real
    np.testing.assert_allclose(recovered, x, atol=1e-12)


def test_roundtrip_imaginary_part_is_negligible_for_real_input():
    """For a real-valued input, IFFT(FFT(x)) should have a negligible
    imaginary component -- if this fails, something is wrong with how
    frequency-domain products will be interpreted downstream in
    fft_correlate_3d.
    """
    rng = np.random.default_rng(3)
    x = rng.normal(size=(16, 16, 16))
    result = np.fft.ifftn(np.fft.fftn(x))
    assert np.max(np.abs(result.imag)) < 1e-10


if __name__ == "__main__":
    test_roundtrip_cubic_power_of_two()
    test_roundtrip_non_power_of_two()
    test_roundtrip_asymmetric_cuboid()
    test_roundtrip_preserves_zero_grid()
    test_roundtrip_imaginary_part_is_negligible_for_real_input()
    print("All FFT round-trip tests passed.")
