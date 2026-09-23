"""
triad.correlation.preflight
===============================
A cheap, pre-search predictor of whether TRIAD's full rotation-inclusive
search is likely to recover a near-native pose for a given target/ligase
pair, WITHOUT running the expensive search itself.

Built from the finding in docs/TRIAD_v1_MANIFEST.md Part 29: native's own
electrostatic correlation score (a single, millisecond-scale evaluation)
correlates with final search RMSD at r=-0.765, p=0.0061 (n=11) -- a
stronger and far cheaper signal than the post-hoc gap metric in Part 28,
which requires the full search to already be complete.

STATED LIMITATION: n=11 is a small sample for a correlation coefficient.
This is a refinement of an already-established mechanism (Part 27-28),
not an independently re-validated finding, and the classification
threshold below has not itself been calibrated against held-out data --
it is a reasonable first estimate from the available points, not a
tuned decision boundary. Treat this function's output as a real, useful
estimate, not a guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from triad.correlation.channels import build_receptor_potential_grid, build_charge_grid
from triad.correlation.fft_dock import fft_correlate_3d


@dataclass
class PreflightResult:
    native_score: float
    predicted_tractable: bool
    confidence: str  # "high", "medium", "low" -- qualitative, from Part 29's observed spread


# Thresholds read directly off the Part 29 data (not independently tuned):
# the two best-recovered structures scored 333.6 and 489.2; the worst two
# scored -404.3 and 107.2 but with RMSD > 130A. A native score above ~200
# has, so far, always recovered under 30A; below 0 has always recovered
# over 50A. The middle band is genuinely uncertain given current data.
HIGH_CONFIDENCE_TRACTABLE = 200.0
HIGH_CONFIDENCE_INTRACTABLE = 0.0


def preflight_check(
    t_coords: np.ndarray, t_charges: np.ndarray,
    lig_coords: np.ndarray, lig_charges: np.ndarray,
    grid_shape: tuple[int, int, int], origin: np.ndarray, spacing: float,
) -> PreflightResult:
    """Evaluate native's own electrostatic correlation score at the given
    (presumably true, or a candidate) pose, and return a qualitative
    tractability estimate. This does NOT run any search -- it is a single,
    cheap evaluation meant to be called before deciding whether to commit
    to the full rotation-inclusive search.
    """
    pot_r = build_receptor_potential_grid(t_coords, t_charges, grid_shape, origin, spacing)
    charge_l = build_charge_grid(lig_coords, lig_charges, grid_shape, origin, spacing)
    C_elec = -fft_correlate_3d(pot_r, charge_l)
    native_score = float(C_elec[0, 0, 0])

    if native_score >= HIGH_CONFIDENCE_TRACTABLE:
        return PreflightResult(native_score, True, "high")
    if native_score < HIGH_CONFIDENCE_INTRACTABLE:
        return PreflightResult(native_score, False, "high")
    return PreflightResult(native_score, native_score > 100.0, "low")
