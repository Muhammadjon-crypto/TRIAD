"""
triad.correlation.fft_dock
=============================
Core FFT correlation primitives for rigid-body docking search.

This module implements the correlation theorem application described in
docs/TRIAD_v1_MANIFEST.md Part 2.2: for two real 3D grids A (fixed) and B
(mobile, at some fixed rotation), the cross-correlation

    C(tau) = sum_x  A(x) * B(x + tau)   [circular, i.e. indices mod grid shape]

is computed as

    C = IFFT3D( FFT3D(A) * conj(FFT3D(B)) )

Every function here is validated against brute-force direct computation in
tests/test_bruteforce_crosscheck.py before being trusted for anything real —
a from-scratch FFT correlation implementation is exactly the kind of code
where a subtle indexing or conjugation error produces a plausible-looking
but silently wrong result.

IMPORTANT: this computes CIRCULAR correlation (an artifact of the DFT's
periodic boundary assumption). For real docking use, grids must be padded
generously beyond the two molecules' combined extent, or a real translation
of interest will wrap around and corrupt the correlation. Padding strategy
is handled by grid.py, not here.
"""
from __future__ import annotations

import numpy as np


def fft_correlate_3d(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Circular cross-correlation of two same-shape real 3D arrays via FFT,
    using the docking-relevant convention:

        C[tau] = sum_x  A(x) * B(x - tau)   [indices mod grid shape]

    This specific convention (B(x - tau), not B(x + tau)) is deliberate and
    was confirmed against brute-force computation, not assumed: it matches
    the same translation convention used everywhere else in TRIAD
    (triad.geometry.transforms.apply_transform: new_coords = old_coords +
    tau implies new_shape(x) = old_shape(x - tau)). So C[tau] is exactly
    "the docking overlap score if the mobile body's shape B is translated
    by tau" -- the physically meaningful quantity for the search.

    IMPLEMENTATION HISTORY (a real mistake made and corrected during
    development, worth keeping visible): an earlier version of this
    function was "fixed" to match a brute-force reference that used the
    OPPOSITE convention (B(x + tau)) -- which is a perfectly valid
    correlation definition in the abstract, but NOT the one that means
    "translate B forward by tau," which is what docking actually needs.
    Both formulas are internally self-consistent; only one of them answers
    the question this codebase actually asks. See
    tests/test_bruteforce_crosscheck.py for the corrected brute-force
    reference using the same B(x - tau) convention.
    """
    if A.shape != B.shape:
        raise ValueError(f"grids must have the same shape, got {A.shape} vs {B.shape}")
    FA = np.fft.fftn(A)
    FB = np.fft.fftn(B)
    C = np.fft.ifftn(FA * np.conj(FB))
    max_imag = np.max(np.abs(C.imag))
    if max_imag > 1e-6 * (np.max(np.abs(C.real)) + 1e-12):
        raise RuntimeError(
            f"unexpectedly large imaginary component in correlation result "
            f"(max |imag|={max_imag:.2e}) -- inputs may not both be real, "
            f"or there's a numerical issue"
        )
    return C.real


def brute_force_correlate_3d(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Direct (non-FFT) circular cross-correlation, for validating
    fft_correlate_3d against, using the SAME docking-relevant convention:

        C[tau] = sum_x  A(x) * B(x - tau)

    np.roll(B, shift=tau)[x] = B[(x - tau) mod N] -- i.e. rolling B forward
    by tau gives exactly B(x - tau), matching the definition above. This
    was gotten wrong once during development (rolled by -tau instead of
    +tau, matching a different, less useful correlation convention) and
    caught by comparing against real docking semantics, not just internal
    self-consistency -- see fft_correlate_3d's docstring for the full story.
    """
    if A.shape != B.shape:
        raise ValueError(f"grids must have the same shape, got {A.shape} vs {B.shape}")
    nx, ny, nz = A.shape
    C = np.zeros((nx, ny, nz), dtype=np.float64)
    for tx in range(nx):
        for ty in range(ny):
            for tz in range(nz):
                B_shifted = np.roll(B, shift=(tx, ty, tz), axis=(0, 1, 2))
                C[tx, ty, tz] = np.sum(A * B_shifted)
    return C


def find_best_translation_index(correlation_grid: np.ndarray) -> tuple[int, int, int]:
    """Grid index of the maximum correlation value -- the best translation
    (in voxel units) found by the search. Converting this to a physical
    Angstrom offset requires the grid spacing and origin (handled by the
    caller, since this module only deals in voxel index space).
    """
    flat_idx = np.argmax(correlation_grid)
    return np.unravel_index(flat_idx, correlation_grid.shape)


def fft_convolve_3d(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Circular convolution of two same-shape real 3D arrays via FFT:

        (A * B)[n] = sum_m  A(m) * B(n - m)

    Distinct from fft_correlate_3d (which uses a conjugated product): plain
    convolution has no conjugation subtlety, confirmed directly against
    brute-force computation (see test_bruteforce_crosscheck.py). Used here
    to solve for an electrostatic potential field from a point-charge
    distribution: potential(x) = sum_y charge(y) * kernel(x - y), which is
    exactly this convolution with kernel = 1/r (Coulomb's law in
    real-space/free-space form).
    """
    if A.shape != B.shape:
        raise ValueError(f"grids must have the same shape, got {A.shape} vs {B.shape}")
    C = np.fft.ifftn(np.fft.fftn(A) * np.fft.fftn(B))
    max_imag = np.max(np.abs(C.imag))
    if max_imag > 1e-6 * (np.max(np.abs(C.real)) + 1e-12):
        raise RuntimeError(
            f"unexpectedly large imaginary component in convolution result "
            f"(max |imag|={max_imag:.2e})"
        )
    return C.real
