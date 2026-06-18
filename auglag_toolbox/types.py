"""Shared types for auglag-toolbox.

``InnerSolver`` and ``Projector`` mirror the pga-toolbox contracts so that the
default inner solver (``pga_ascent_spg``) and any custom projector compose
without adapters. ``AugLagResult`` is the common return type of
``augmented_lagrangian`` / ``augmented_lagrangian_descent`` / ``penalty_method``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

# A constraint / objective closure: zero-arg, returns a real scalar tensor.
ScalarClosure = Callable[[], torch.Tensor]

# Euclidean projector onto S_easy. Either mutates ``params`` in place and
# returns None, or returns a new sequence of tensors (functional). Same
# convention as pga_toolbox.Projector.
Projector = Callable[[list[torch.Tensor]], None | Sequence[torch.Tensor]]

# Inner solver contract: maximise ``closure()`` by updating ``params`` in place,
# optionally projecting onto S_easy. ``pga_ascent_spg`` / ``pga_ascent_armijo``
# satisfy this. The return value is ignored by the outer loop.
InnerSolver = Callable[..., object]


@dataclass
class AugLagResult:
    """Outcome of a constrained solve.

    Attributes:
        params: final parameters (the same tensors passed in, updated in place).
        objective: f(x*) at the returned point.
        multipliers_ineq: final inequality multipliers ``lambda_c >= 0``.
        multipliers_eq: final equality multipliers ``mu_k`` (sign-free).
        feasibility: ``max( max_c [g_c]_+ , max_k |h_k| )``.
        complementarity: ``max_c |lambda_c * g_c|``.
        stationarity: projected-gradient-map residual of the Lagrangian.
        rho: final penalty parameter.
        outer_iters: number of outer iterations used.
        inner_evals_total: total inner objective evaluations (effort metric).
        history: per-outer-iteration records
            ``{feas, compl, stat, lam, mu, rho, inner_evals}``.
        converged: True iff the solve stopped because the KKT tolerances
            (feasibility, complementarity, stationarity) were all met within
            ``outer_iters``. When True the returned point is an approximate KKT
            point; when False it is the last iterate (do not treat it as KKT).
        stop_reason: why the solve stopped --
            ``"kkt_tol"`` (tolerances met),
            ``"max_outer_iters"`` (iteration budget exhausted),
            ``"single_penalty_solve"`` (penalty baseline, no outer loop).
    """

    params: list[torch.Tensor]
    objective: float
    multipliers_ineq: list[float]
    multipliers_eq: list[float]
    feasibility: float
    complementarity: float
    stationarity: float
    rho: float
    outer_iters: int
    inner_evals_total: int
    history: list[dict] = field(default_factory=list)
    converged: bool = False
    stop_reason: str = "unknown"


@dataclass
class BatchedAugLagResult:
    """Outcome of a batched (parallel multi-start) constrained solve.

    Every per-element quantity is a ``(B,)`` tensor (or a list of them, one entry
    per constraint), where ``B`` is the number of parallel starts. ``winner`` is
    the index of the global incumbent: the feasible element with the best
    objective (or, if none are feasible, the least-infeasible element).

    Attributes:
        params: final parameters, each shaped ``(B, *shape)``; ``params[m][b]``
            is element ``b``'s best-seen point.
        objective: ``(B,)`` objective per element.
        multipliers_ineq: list of ``(B,)`` inequality multipliers (>= 0).
        multipliers_eq: list of ``(B,)`` equality multipliers (sign-free).
        feasibility, complementarity, stationarity: ``(B,)`` KKT residuals.
        rho: ``(B,)`` final penalty parameters.
        converged: ``(B,)`` bool; True where all three residuals met tolerance.
        winner: index of the global incumbent.
        outer_iters: number of outer iterations run.
        inner_evals_total: total batched objective evaluations.
        history: per-outer-iteration records of ``(B,)`` tensors.
    """

    params: list[torch.Tensor]
    objective: torch.Tensor
    multipliers_ineq: list[torch.Tensor]
    multipliers_eq: list[torch.Tensor]
    feasibility: torch.Tensor
    complementarity: torch.Tensor
    stationarity: torch.Tensor
    rho: torch.Tensor
    converged: torch.Tensor
    winner: int
    outer_iters: int
    inner_evals_total: int
    history: list[dict] = field(default_factory=list)

    def winner_result(self) -> AugLagResult:
        """Extract the global incumbent as a scalar :class:`AugLagResult`."""
        b = self.winner
        won = bool(self.converged[b])
        return AugLagResult(
            params=[p[b].detach().clone() for p in self.params],
            objective=float(self.objective[b]),
            multipliers_ineq=[float(lam[b]) for lam in self.multipliers_ineq],
            multipliers_eq=[float(mu[b]) for mu in self.multipliers_eq],
            feasibility=float(self.feasibility[b]),
            complementarity=float(self.complementarity[b]),
            stationarity=float(self.stationarity[b]),
            rho=float(self.rho[b]),
            outer_iters=self.outer_iters,
            inner_evals_total=self.inner_evals_total,
            history=[],
            converged=won,
            stop_reason="kkt_tol" if won else "max_outer_iters",
        )
