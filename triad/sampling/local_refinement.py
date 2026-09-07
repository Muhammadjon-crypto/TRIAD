"""
triad.sampling.local_refinement
==================================
Local refinement of a coarse rigid-body candidate pose via finite-difference
numerical optimization (NOT analytic gradient descent -- the clash/BSA
objective has boolean occlusion transitions that aren't strictly
differentiable, so gradients here are numerical estimates, re-linearized at
each step, in the style of Gauss-Newton local search).

Objective minimized: clash_score(fixed, mobile) + k_reach * (reach_deviation)^2

The reach-constraint penalty term is essential, not optional: without it,
the optimizer would trivially minimize clash by translating the mobile body
infinitely far away. The penalty keeps the mobile body's anchor point
tethered to the correct reach distance from the fixed body's anchor while
letting rotation and residual translation move freely to resolve steric
overlap -- this is the "snap into place" step: coarse sampling finds the
right neighborhood, refinement finds the actual local optimum within it.

Small rotations are parameterized as rotation vectors via Rodrigues'
formula, which is exact for any rotation magnitude (not a small-angle
linear approximation), so this remains valid even if a refinement step
takes an unexpectedly large corrective turn.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from triad.scoring.clash import clash_score


def rotation_from_rotvec(omega: np.ndarray) -> np.ndarray:
    """Exact rotation matrix from a rotation vector (axis * angle) via
    Rodrigues' formula. Not a small-angle approximation -- valid for any
    rotation magnitude.
    """
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        return np.eye(3)
    axis = omega / theta
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


@dataclass
class RefinementResult:
    rotation: np.ndarray
    pivot_position: np.ndarray
    objective_history: list[float]
    final_clash_score: float
    final_reach_deviation: float


def _objective(
    delta: np.ndarray,
    current_R: np.ndarray, current_pivot: np.ndarray,
    mobile_centered: np.ndarray, mobile_elements: list[str],
    fixed_coords: np.ndarray, fixed_elements: list[str],
    target_anchor: np.ndarray, reach_distance: float, k_reach: float,
) -> float:
    domega, dt = delta[:3], delta[3:]
    dR = rotation_from_rotvec(domega)
    new_R = dR @ current_R
    new_pivot = current_pivot + dt

    transformed = mobile_centered @ new_R.T + new_pivot
    clash = clash_score(fixed_coords, fixed_elements, transformed, mobile_elements)
    reach_dev = np.linalg.norm(new_pivot - target_anchor) - reach_distance
    return clash + k_reach * reach_dev ** 2


def _numerical_gradient(f, x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (f(xp) - f(xm)) / (2 * eps)
    return grad


def local_refine(
    initial_rotation: np.ndarray,
    initial_pivot: np.ndarray,
    mobile_coords: np.ndarray,
    mobile_pivot_local: np.ndarray,
    mobile_elements: list[str],
    fixed_coords: np.ndarray,
    fixed_elements: list[str],
    target_anchor: np.ndarray,
    reach_distance: float,
    n_steps: int = 40,
    learning_rate: float = 0.02,
    k_reach: float = 5.0,
) -> RefinementResult:
    """Iteratively refine a coarse candidate pose.

    `mobile_coords` should be in the SAME original (untransformed) frame as
    `mobile_pivot_local` -- both get re-centered internally on the pivot so
    rotations are applied correctly about it.
    """
    mobile_centered = np.asarray(mobile_coords, dtype=np.float64) - mobile_pivot_local

    R_cur = initial_rotation.copy()
    pivot_cur = initial_pivot.copy()
    history = []
    step_size = learning_rate

    for step in range(n_steps):
        f = lambda d: _objective(
            d, R_cur, pivot_cur, mobile_centered, mobile_elements,
            fixed_coords, fixed_elements, target_anchor, reach_distance, k_reach,
        )
        current_score = f(np.zeros(6))
        history.append(current_score)

        grad = _numerical_gradient(f, np.zeros(6))
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-8:
            break  # converged: flat gradient, no improving direction found

        # backtracking line search (Armijo-style) on the natural (non-normalized)
        # gradient direction: halve the step multiplier until the move actually
        # improves the objective. Normalizing the gradient direction was tried
        # first and made things WORSE (broke a previously-passing case) --
        # it discards the gradient's natural magnitude information, causing
        # overly large steps on shallow slopes. Kept here as -step * grad,
        # exactly proportional descent, just with an adaptive step size.
        trial_step = step_size
        improved = False
        for _ in range(20):  # bounded backtracking attempts
            trial_delta = -trial_step * grad
            trial_score = f(trial_delta)
            if trial_score < current_score:
                improved = True
                break
            trial_step *= 0.5

        if not improved:
            break  # no improving step found even at tiny size: converged

        domega, dt = trial_delta[:3], trial_delta[3:]
        R_cur = rotation_from_rotvec(domega) @ R_cur
        pivot_cur = pivot_cur + dt
        step_size = min(trial_step * 1.5, learning_rate)  # grow back gradually next step

    final_transformed = mobile_centered @ R_cur.T + pivot_cur
    final_clash = clash_score(fixed_coords, fixed_elements, final_transformed, mobile_elements)
    final_reach_dev = np.linalg.norm(pivot_cur - target_anchor) - reach_distance

    return RefinementResult(
        rotation=R_cur, pivot_position=pivot_cur,
        objective_history=history,
        final_clash_score=final_clash,
        final_reach_deviation=final_reach_dev,
    )
