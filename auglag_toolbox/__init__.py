"""auglag-toolbox: augmented-Lagrangian / penalty constraint handling on top of
a first-order projected solver (pga-toolbox), for complex (Wirtinger) and real
parameters.

This library owns only the *outer* loop that enforces nonlinear, coupled,
covariance-dependent constraints. The *inner* maximisation over the easy
(closed-form-projectable) set is delegated to a pluggable first-order solver --
by default ``pga_toolbox.pga_ascent_spg``.

Problem (maximisation form):

    maximise f(x)   s.t.   g_c(x) <= 0,  h_k(x) = 0,  x in S_easy.

Public API:
    Solvers:
        augmented_lagrangian, augmented_lagrangian_descent, penalty_method
    Batched (parallel multi-start) solvers:
        augmented_lagrangian_batched, augmented_lagrangian_descent_batched
    Diagnostics:
        kkt_residuals, kkt_residuals_batched
    Types:
        AugLagResult, BatchedAugLagResult, InnerSolver, Projector

Typical usage:
    >>> from auglag_toolbox import augmented_lagrangian
    >>> from pga_toolbox import project_frobenius_ball
    >>>
    >>> def objective():
    ...     return mutual_information(F)          # real scalar tensor, maximise
    >>> def power():
    ...     return torch.trace(F @ W @ F.T) - P1  # g(x) <= 0
    >>> def projector(params):
    ...     return [project_frobenius_ball(params[0], P)]
    >>>
    >>> res = augmented_lagrangian(
    ...     objective, [F], inequality=[power], easy_projector=projector)
    >>> res.feasibility, res.objective, res.multipliers_ineq
"""

from .batched import (
    augmented_lagrangian_batched,
    augmented_lagrangian_descent_batched,
)
from .core import augmented_lagrangian, augmented_lagrangian_descent
from .kkt import kkt_residuals, kkt_residuals_batched
from .penalty import penalty_method
from .types import AugLagResult, BatchedAugLagResult, InnerSolver, Projector

__version__ = "0.2.0"

__all__ = [
    "augmented_lagrangian",
    "augmented_lagrangian_descent",
    "augmented_lagrangian_batched",
    "augmented_lagrangian_descent_batched",
    "penalty_method",
    "kkt_residuals",
    "kkt_residuals_batched",
    "AugLagResult",
    "BatchedAugLagResult",
    "InnerSolver",
    "Projector",
    "__version__",
]
