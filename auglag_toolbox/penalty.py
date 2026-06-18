"""Pure quadratic-penalty baseline (B1) for comparison against the ALM.

Single inner solve of

    maximise f(x) - mu * ( sum_c [g_c(x)]_+^2 + sum_k h_k(x)^2 )

with no dual variables. As ``mu`` grows the problem becomes ill-conditioned and
the objective degrades while feasibility improves -- the contrast that motivates
the augmented Lagrangian (smoke E3). Returns the same ``AugLagResult`` type.
"""

from __future__ import annotations

import torch
from pga_toolbox import pga_ascent_spg

from .kkt import kkt_residuals
from .types import AugLagResult, InnerSolver, Projector, ScalarClosure

_DEFAULT_INNER_KWARGS = {"max_iter": 300, "forward_budget": 400}


def penalty_method(
    objective: ScalarClosure,
    params: list[torch.Tensor],
    *,
    inequality: list[ScalarClosure] | None = None,
    equality: list[ScalarClosure] | None = None,
    easy_projector: Projector | None = None,
    inner: InnerSolver = pga_ascent_spg,
    inner_kwargs: dict | None = None,
    mu: float = 1.0,
    feas_tol: float = 1e-6,
    verbose: bool = False,
) -> AugLagResult:
    """Maximise ``objective`` with a single quadratic-penalty inner solve.

    ``mu`` is the (fixed) penalty weight. ``multipliers_ineq`` /
    ``multipliers_eq`` in the result are empty since the method uses no duals;
    ``complementarity`` is therefore reported as 0 and ``stationarity`` is the
    projected-gradient residual of ``objective`` alone.

    This is a single solve with no outer loop, so ``stop_reason`` is always
    ``"single_penalty_solve"``. ``converged`` reports only whether feasibility
    was reached (``feasibility <= feas_tol``) -- the penalty baseline does not
    certify a KKT point (it gates on no stationarity / complementarity).
    """
    inequality = list(inequality) if inequality else []
    equality = list(equality) if equality else []
    kw = dict(inner_kwargs) if inner_kwargs else dict(_DEFAULT_INNER_KWARGS)

    counter = {"n": 0}

    def penalised():
        counter["n"] += 1
        val = objective()
        for g in inequality:
            val = val - mu * torch.clamp(g(), min=0.0) ** 2
        for h in equality:
            val = val - mu * h() ** 2
        return val

    inner(penalised, params, projector=easy_projector, **kw)

    feas, compl, stat = kkt_residuals(
        objective, params, inequality, equality,
        [0.0] * len(inequality), [0.0] * len(equality), easy_projector,
    )
    with torch.no_grad():
        objective_value = float(objective().detach().real)

    if verbose:
        print(f"  [penalty mu={mu:g}] f={objective_value:.5f} feas={feas:.2e} "
              f"inner_evals={counter['n']}")

    return AugLagResult(
        params=params,
        objective=objective_value,
        multipliers_ineq=[],
        multipliers_eq=[],
        feasibility=feas,
        complementarity=compl,
        stationarity=stat,
        rho=mu,
        outer_iters=1,
        inner_evals_total=counter["n"],
        history=[{"feas": feas, "compl": compl, "stat": stat,
                  "mu": mu, "inner_evals": counter["n"]}],
        converged=feas <= feas_tol,
        stop_reason="single_penalty_solve",
    )
