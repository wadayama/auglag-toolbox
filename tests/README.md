# tests

Promoted from the verified pga-toolbox smoke (E1-E5) plus new equality coverage.

| File | Covers |
| --- | --- |
| `test_alm_inequality.py` | E1: covariance-weighted inequality (W != I), active KKT |
| `test_alm_equality.py` | **new** equality constraint: dual `mu <- mu + rho h`, `|h| -> 0` |
| `test_alm_coupled_covariance.py` | E2: coupled, covariance-block constraint `tr(R K11 R^H) <= P_relay`, active |
| `test_penalty_vs_alm.py` | E3: penalty mu-sweep (infeasible / degraded) vs ALM |
| `test_complex_wirtinger.py` | E5: complex (Wirtinger) ALM, small KKT residual |
| `test_kkt_and_conventions.py` | sign conventions, active/inactive multipliers, `kkt_residuals` |
| `test_pluggable_inner.py` | `inner=pga_ascent_armijo` also reaches feasible KKT |
| `test_multistart_and_descent.py` | E4 multistart robustness to rho schedule; descent wrapper |
| `test_convergence_reporting.py` | `converged` / `stop_reason` flags; user-tunable tolerances |
| `test_batched_multistart.py` | batched ALM vs sequential oracle; multimodal `winner`; batched KKT |
| `test_slsqp_reference.py` | E1 vs scipy SLSQP, relative objective < 1e-3 (skipped without scipy) |

Pass tolerances: `feas <= 1e-6`, `compl <= 1e-6`, `stat <= 1e-5`; SLSQP relative `< 1e-3`.

## Running

```sh
uv sync --extra test
uv run pytest
```

During development, before `pga-toolbox` is published, inject the sibling
checkout instead of fetching `git+https`:

```sh
uv run --with-editable ../pga-toolbox --extra test pytest
```
