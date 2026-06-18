"""E2: coupled, covariance-dependent node transmit-power constraint on (F, R).

The hard constraint contains a covariance block K11 = F F^T + sigma^2 I and
couples the two optimisation variables:

    maximise  MI(F, R)
    s.t.      tr( R K11 R^T ) <= P_relay      (nonlinear, coupled, no projection)
              ||F||_F^2 <= Pf                  (easy: Frobenius ball, R free)

This is the flagship example's problem (gaussian-dag "future work"), promoted to
an automated regression test. The constraint is ACTIVE by construction (MI
saturates in the relay gain, so a finite cap binds).
"""

import torch
from pga_toolbox import pga_ascent_spg, project_frobenius_ball

from auglag_toolbox import augmented_lagrangian
from _helpers import logdet_pd, randmat


def test_e2_coupled_covariance_dependent_power_is_active():
    d, sigma, Pf = 2, 0.5, 4.0
    H2 = randmat(d, seed=3)
    I = torch.eye(d, dtype=torch.float64)

    def mi(F, R):
        G = H2 @ R @ F
        C = sigma**2 * (H2 @ R @ R.T @ H2.T) + sigma**2 * I
        return logdet_pd(G @ G.T + C) - logdet_pd(C)

    def easy_proj(params):  # project F only; R free
        F, R = params
        return [project_frobenius_ball(F, Pf), R]

    # Cap near an R=I operating point (x2), which binds because MI saturates.
    F = randmat(d, seed=4, scale=0.3).clone().requires_grad_(True)
    R = randmat(d, seed=5, scale=0.3).clone().requires_grad_(True)
    pga_ascent_spg(lambda: mi(F, R), [F, R], projector=easy_proj,
                   max_iter=400, forward_budget=600)
    with torch.no_grad():
        K11 = F @ F.T + sigma**2 * I
        P_relay = 2.0 * float(torch.trace(K11))

    F = randmat(d, seed=4, scale=0.3).clone().requires_grad_(True)
    R = randmat(d, seed=5, scale=0.3).clone().requires_grad_(True)

    def g_relay():
        K11 = F @ F.T + sigma**2 * I        # covariance block enters the constraint
        return torch.trace(R @ K11 @ R.T) - P_relay

    res = augmented_lagrangian(
        lambda: mi(F, R), [F, R], inequality=[g_relay], easy_projector=easy_proj,
    )

    assert res.feasibility <= 1e-6
    assert res.complementarity <= 1e-6
    assert res.stationarity <= 1e-5

    # Constraint must be ACTIVE: positive multiplier and relay power at the cap.
    assert res.multipliers_ineq[0] > 1e-6
    with torch.no_grad():
        relay_power = float(torch.trace(R @ (F @ F.T + sigma**2 * I) @ R.T))
    assert abs(relay_power - P_relay) / P_relay < 1e-3
