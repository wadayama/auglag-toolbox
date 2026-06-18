"""PHR augmented-Lagrangian outer loop on top of a projected first-order solver.

Productizes the verified reference ``alm_solve`` (smoke E1-E5), extended with
equality constraints, an optional non-warm-start mode, sequential multi-start,
a combined feasibility+stationarity stopping rule, and a structured result.

Problem (maximisation form):

    maximise f(x)
    s.t.  g_c(x) <= 0   (inequality, multiplier lambda_c >= 0)
          h_k(x) = 0    (equality,   multiplier mu_k, sign-free)
          x in S_easy   (handled by closed-form projection inside the inner solve)

PHR augmented objective maximised by the inner solver at fixed (lambda, mu, rho):

    F_rho(x) = f(x)
             - sum_c (1/(2 rho)) ( [lambda_c + rho g_c(x)]_+^2 - lambda_c^2 )
             - sum_k ( mu_k h_k(x) + (rho/2) h_k(x)^2 )

Dual updates:  lambda_c <- [lambda_c + rho g_c]_+ ;  mu_k <- mu_k + rho h_k.
"""

from __future__ import annotations

import torch
from pga_toolbox import pga_ascent_spg

from .kkt import kkt_residuals
from .types import AugLagResult, InnerSolver, Projector, ScalarClosure

_DEFAULT_INNER_KWARGS = {"max_iter": 300, "forward_budget": 400}


def augmented_lagrangian(
    objective: ScalarClosure,
    params: list[torch.Tensor],
    *,
    inequality: list[ScalarClosure] | None = None,
    equality: list[ScalarClosure] | None = None,
    easy_projector: Projector | None = None,
    inner: InnerSolver = pga_ascent_spg,
    inner_kwargs: dict | None = None,
    rho0: float = 1.0,
    rho_grow: float = 3.0,
    rho_max: float = 1e6,
    feas_tol: float = 1e-6,
    compl_tol: float = 1e-6,
    stat_tol: float = 1e-5,
    outer_iters: int = 40,
    warm_start: bool = True,
    multistart: int = 1,
    seed: int | None = None,
    verbose: bool = False,
) -> AugLagResult:
    """Maximise ``objective`` subject to inequality / equality constraints.

    ``objective`` and each constraint are zero-argument closures returning a
    real scalar tensor that capture the live ``params`` (leaf tensors with
    ``requires_grad=True``). Constraints follow the sign convention
    ``g_c(x) <= 0`` and ``h_k(x) == 0``; for complex parameters the constraint
    value must be real (e.g. ``tr(F W F^H).real``).

    ``params`` is updated in place to the best point found (also returned in
    ``AugLagResult.params``). The inner solver must maximise its closure by
    updating ``params`` in place (the pga-toolbox contract); ``pga_ascent_spg``
    is the default, ``pga_ascent_armijo`` also satisfies it.

    With ``warm_start=True`` (default) the primal iterate is carried across outer
    iterations; with ``warm_start=False`` it is reset to the start point before
    each inner solve. ``multistart > 1`` runs the whole solve sequentially from
    ``multistart`` seed-derived initial points and returns the best feasible one
    (a non-convexity safeguard).

    The solve stops early once all three KKT residuals meet their (user-tunable)
    tolerances -- ``feasibility <= feas_tol``, ``complementarity <= compl_tol``,
    ``stationarity <= stat_tol`` -- in which case ``AugLagResult.converged`` is
    True and ``stop_reason == "kkt_tol"``. If the ``outer_iters`` budget is
    exhausted first, ``converged`` is False and ``stop_reason ==
    "max_outer_iters"``; the returned point is then the last iterate and should
    not be treated as a KKT point. Always check ``converged`` / the residuals.
    """
    inequality = list(inequality) if inequality else []
    equality = list(equality) if equality else []
    kw = dict(inner_kwargs) if inner_kwargs else dict(_DEFAULT_INNER_KWARGS)

    init_snapshot = [p.detach().clone() for p in params]

    best_result: AugLagResult | None = None
    best_values: list[torch.Tensor] | None = None
    best_key: tuple[int, float] | None = None

    for s in range(multistart):
        _reinitialize(params, init_snapshot, s, seed)
        result = _solve_single(
            objective, params, inequality, equality, easy_projector, inner, kw,
            rho0, rho_grow, rho_max, feas_tol, compl_tol, stat_tol, outer_iters,
            warm_start, verbose, s,
        )
        feasible = result.feasibility <= feas_tol
        # Rank: feasible beats infeasible; among feasible maximise objective;
        # among infeasible prefer the least-infeasible point.
        key = (int(feasible), result.objective if feasible else -result.feasibility)
        if best_key is None or key > best_key:
            best_result, best_key = result, key
            best_values = [p.detach().clone() for p in params]

    assert best_result is not None and best_values is not None
    with torch.no_grad():
        for p, v in zip(params, best_values):
            p.copy_(v)
    best_result.params = params
    return best_result


def augmented_lagrangian_descent(
    cost: ScalarClosure,
    params: list[torch.Tensor],
    **kwargs,
) -> AugLagResult:
    """Minimise ``cost`` subject to the same constraints, via ``max (-cost)``.

    Accepts the same keyword arguments as :func:`augmented_lagrangian`. The
    returned ``AugLagResult.objective`` is the original cost ``cost(x*)`` (the
    sign flip is undone before returning).
    """
    result = augmented_lagrangian(lambda: -cost(), params, **kwargs)
    result.objective = -result.objective
    return result


def _solve_single(
    objective, params, inequality, equality, easy_projector, inner, inner_kwargs,
    rho0, rho_grow, rho_max, feas_tol, compl_tol, stat_tol, outer_iters,
    warm_start, verbose, start_index,
) -> AugLagResult:
    """Run the PHR outer loop once from the current ``params``."""
    lam = [0.0] * len(inequality)
    mu = [0.0] * len(equality)
    rho = rho0
    prev_feas = float("inf")
    total_inner_evals = 0
    history: list[dict] = []

    start_values = [p.detach().clone() for p in params]
    feas = compl = stat = float("inf")
    converged = False
    t = 0

    for t in range(outer_iters):
        if not warm_start:
            with torch.no_grad():
                for p, v in zip(params, start_values):
                    p.copy_(v)

        counter = {"n": 0}

        def augmented(_lam=lam, _mu=mu, _rho=rho):
            counter["n"] += 1
            val = objective()
            for c, g in enumerate(inequality):
                viol = torch.clamp(_lam[c] + _rho * g(), min=0.0)
                val = val - (viol**2 - _lam[c] ** 2) / (2.0 * _rho)
            for k, h in enumerate(equality):
                hk = h()
                val = val - (_mu[k] * hk + (_rho / 2.0) * hk**2)
            return val

        inner(augmented, params, projector=easy_projector, **inner_kwargs)
        total_inner_evals += counter["n"]

        # Dual update from the new primal point.
        with torch.no_grad():
            gvals = [float(g().detach().real) for g in inequality]
            hvals = [float(h().detach().real) for h in equality]
        for c in range(len(lam)):
            lam[c] = max(0.0, lam[c] + rho * gvals[c])
        for k in range(len(mu)):
            mu[k] = mu[k] + rho * hvals[k]

        feas, compl, stat = kkt_residuals(
            objective, params, inequality, equality, lam, mu, easy_projector
        )
        history.append({
            "feas": feas, "compl": compl, "stat": stat,
            "lam": list(lam), "mu": list(mu), "rho": rho,
            "inner_evals": counter["n"],
        })

        if verbose:
            print(
                f"  [start {start_index} outer {t:2d}] feas={feas:.3e} "
                f"compl={compl:.3e} stat={stat:.3e} rho={rho:.1e} "
                f"lam={[round(x, 3) for x in lam]} "
                f"mu={[round(x, 3) for x in mu]} inner_evals={counter['n']}"
            )

        if (feas <= feas_tol and compl <= compl_tol and stat <= stat_tol
                and t >= 1):
            converged = True
            break
        if feas > 0.5 * prev_feas:  # feasibility stalled -> tighten
            rho = min(rho * rho_grow, rho_max)
        prev_feas = feas

    with torch.no_grad():
        objective_value = float(objective().detach().real)

    return AugLagResult(
        params=params,
        objective=objective_value,
        multipliers_ineq=list(lam),
        multipliers_eq=list(mu),
        feasibility=feas,
        complementarity=compl,
        stationarity=stat,
        rho=rho,
        outer_iters=t + 1,
        inner_evals_total=total_inner_evals,
        history=history,
        converged=converged,
        stop_reason="kkt_tol" if converged else "max_outer_iters",
    )


def _reinitialize(params, init_snapshot, start_index, seed):
    """Place ``params`` at the initial point for multi-start ``start_index``.

    Start 0 restores the user-provided initial point exactly. Later starts draw
    fresh Gaussian values that preserve the RMS scale of the original guess,
    using a generator seeded by ``seed + start_index`` for reproducibility.
    """
    if start_index == 0:
        with torch.no_grad():
            for p, p0 in zip(params, init_snapshot):
                p.copy_(p0)
        return

    base = 0 if seed is None else int(seed)
    gen = torch.Generator().manual_seed(base + start_index)
    with torch.no_grad():
        for p, p0 in zip(params, init_snapshot):
            rms = float(p0.detach().abs().pow(2).mean().sqrt())
            scale = rms if rms > 0 else 1.0
            shape = tuple(p0.shape)
            if p0.is_complex():
                real = torch.randn(shape, generator=gen, dtype=torch.float64)
                imag = torch.randn(shape, generator=gen, dtype=torch.float64)
                draw = torch.complex(real, imag) / (2.0**0.5)
            else:
                draw = torch.randn(shape, generator=gen, dtype=torch.float64)
            p.copy_((scale * draw).to(dtype=p0.dtype, device=p0.device))
