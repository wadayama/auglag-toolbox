# Tutorial 5 — Reproducing the node-power figure

The flagship example is fully reproducible: every printed, plotted, and saved
number comes from the actual optimiser run (no hard-coded results).

## Run it

```bash
uv run --extra examples python examples/node_power_constraint.py
```

(Development, pre-publication of `pga-toolbox`:
`uv run --with-editable ../pga-toolbox --extra examples python examples/node_power_constraint.py`.)

## What you should see

```
[B0 projection-only] MI=4.3944  relay power=28119744.72  (cap P_relay=9.0000)
  [start 0 outer  0] feas=8.613e-02 ... rho=1.0e+00 lam=[0.086] ...
  ...
[ALM] MI=3.22687  feas=2.67e-07 compl=2.34e-08 stat=1.18e-07
      lambda=0.0877  rho=1.0e+00  outer_iters=4  inner_evals=643
      relay power=9.0000 (cap 9.0000) -> constraint ACTIVE
```

The story: projection-only leaves the relay power running away to ~`2.8e7`
(MI saturates in relay gain), while the ALM solution sits exactly at the cap
`9.0` with a positive multiplier — the constraint is **active**, and the KKT
residuals are all tiny at a finite `rho = 1.0`.

## Outputs

Written next to the script:

- `node_power_constraint_results.npz` — config, final KKT residuals, multiplier,
  and per-outer-iteration history arrays (`feas_hist`, `stat_hist`, …).
- `node_power_constraint.pdf` — two panels:
  - **left**: KKT-residual convergence (feasibility, stationarity) vs outer
    iteration;
  - **right**: relay power, projection-only vs ALM, against the cap (log scale).

## Inspecting the saved data

```python
import numpy as np
d = np.load("examples/node_power_constraint_results.npz")
print(d["mi_alm"], d["relay_power_alm"], d["P_relay"], d["multiplier"])
print(d["feas_hist"])      # the real convergence trajectory used in the figure
```

Because the seeds are fixed inside the script, re-running reproduces these
numbers exactly — the basis for the regression coverage in
[`tests/`](../tests/README.md).
