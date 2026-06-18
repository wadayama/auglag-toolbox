# Mathematics of auglag-toolbox

Implementation-side account of the PHR (Powell–Hestenes–Rockafellar) augmented
Lagrangian as used here. GitHub renders LaTeX between `$...$` / `$$...$$`; this
file uses `\mathrm{...}` throughout (`\operatorname` does not render on GitHub).

## Problem

Maximisation form, with easy and hard constraints separated:

$$
\max_{x}\; f(x)\quad\text{s.t.}\quad
g_c(x)\le 0\ (c=1\ldots C),\qquad
h_k(x)=0\ (k=1\ldots K),\qquad
x\in S_{\mathrm{easy}} .
$$

- $S_{\mathrm{easy}}$ has a closed-form Euclidean projection $P_{S_{\mathrm{easy}}}$
  (a Frobenius ball $\lVert x\rVert_F^2\le P$, a total-power ball). It is handled
  *inside* the inner solver by projection.
- $g_c,h_k$ are nonlinear, possibly coupling several variables and depending on
  covariances; they have **no** projection and are handled by multipliers.

## PHR augmented Lagrangian (maximisation)

With inequality multipliers $\lambda_c\ge 0$, equality multipliers $\mu_k$, and
penalty parameter $\rho>0$, the inner solver maximises

$$
F_\rho(x,\lambda,\mu)=f(x)
-\sum_{c=1}^{C}\frac{1}{2\rho}\Big(\big[\lambda_c+\rho\,g_c(x)\big]_+^2-\lambda_c^2\Big)
-\sum_{k=1}^{K}\Big(\mu_k\,h_k(x)+\tfrac{\rho}{2}\,h_k(x)^2\Big),
$$

where $[\,\cdot\,]_+=\max(0,\cdot)$. The inequality term is the PHR
(shifted-quadratic) form: it is smooth in $x$, equals $-\lambda_c g_c$ to first
order when the constraint is active, and is flat (constant $+\lambda_c^2/2\rho$,
removed by the $-\lambda_c^2$ offset) once $\lambda_c+\rho g_c<0$, i.e. when the
constraint is comfortably satisfied.

## Outer loop

1. **Inner solve** (easy constraints kept as projection):
   $x \leftarrow \arg\max_{x\in S_{\mathrm{easy}}} F_\rho(x,\lambda,\mu)$.
2. **Dual update** from the new $x$:

$$
\lambda_c\leftarrow\big[\lambda_c+\rho\,g_c(x)\big]_+,\qquad
\mu_k\leftarrow\mu_k+\rho\,h_k(x).
$$

3. **Penalty rule.** With feasibility
   $\mathrm{feas}=\max\!\big(\max_c[g_c]_+,\ \max_k|h_k|\big)$, tighten only when
   feasibility stalls:

$$
\text{if } \mathrm{feas} > \tfrac12\,\mathrm{feas}_{\mathrm{prev}}:\quad
\rho\leftarrow\min(\gamma\rho,\ \rho_{\max}),\qquad\text{else hold }\rho.
$$

4. **Stop** when all three KKT residuals meet their (user-tunable) tolerances:
   $\mathrm{feas}\le\texttt{feas\_tol}$, $\mathrm{compl}\le\texttt{compl\_tol}$,
   and $\mathrm{stat}\le\texttt{stat\_tol}$ (then `converged = True`,
   `stop_reason = "kkt_tol"`). If the outer-iteration budget is exhausted first,
   `converged = False` and the returned point is the last iterate, **not** a KKT
   point.

Because the multipliers absorb the active-constraint forces, convergence is
reached at a **finite** $\rho$ — the PHR method does not require $\rho\to\infty$,
unlike the pure quadratic penalty (see below), and is far better conditioned.

## KKT residuals

At the current $(x,\lambda,\mu)$ with Lagrangian
$L=f-\sum_c\lambda_c g_c-\sum_k\mu_k h_k$:

- **Feasibility** $\displaystyle \mathrm{feas}=\max\Big(\max_c[g_c(x)]_+,\ \max_k|h_k(x)|\Big)$.
- **Complementarity** $\displaystyle \mathrm{compl}=\max_c|\lambda_c\,g_c(x)|$
  (equalities do not contribute).
- **Stationarity** — projected-gradient-map residual with unit step:

$$
\mathrm{stat}=\sum_{p}\big\lVert\, x_p - P_{S_{\mathrm{easy}}}\!\big(x_p+\nabla_{x_p}L\big)\,\big\rVert .
$$

For $S_{\mathrm{easy}}=\mathbb{R}^n$ (no projection) this reduces to
$\lVert\nabla_x L\rVert$, the ordinary stationarity condition.

## Pure penalty baseline

For comparison, `penalty_method` solves a single subproblem

$$
\max_x\; f(x)-\mu\Big(\sum_c[g_c(x)]_+^2+\sum_k h_k(x)^2\Big),
$$

with no duals. Feasibility improves only as $\mu\to\infty$, which makes the
landscape stiff and degrades the attainable objective — the contrast that
motivates the augmented Lagrangian.

## Complex (Wirtinger) parameters

For a complex leaf $x$ and a real objective, PyTorch's `.grad` is the natural
Wirtinger gradient — the real-Euclidean steepest-ascent direction on the
$(\mathrm{Re}\,x,\mathrm{Im}\,x)$ lift. Every formula above is unchanged: the
inner solver and the projection operate on the lift, and the stationarity
residual uses the same `.grad`. Constraints must return **real** scalars, e.g.
$g(x)=\mathrm{tr}(F W F^{H}).\mathrm{real}-P$.

## Non-convexity

In general $f$ and the $g_c$ are non-convex, so the method certifies a **KKT
(stationary) point**, not a global optimum. Mitigations: `multistart > 1`
(best feasible over several seeded starts), and constraint scaling (normalise
each $g_c$ by its budget $P_j$ so the multipliers are comparably scaled).

**Batched multi-start.** `augmented_lagrangian_batched` runs $B$ starts in
parallel: every scalar above ($\lambda_c,\mu_k,\rho$, the residuals, the
convergence flag) becomes a length-$B$ vector and the whole outer loop is
vectorised over the batch. The math is identical per element; the only
requirements are that the closures return length-$B$ values, are NaN-safe (a
diverged element returns NaN rather than raising, since a batched
Cholesky/log-det cannot be caught per element), and that the batch elements are
independent (each uses only its own slice and its own $\lambda,\mu,\rho$) — which
makes the single backward pass $\nabla(\sum_b L_b)$ yield the correct
per-element gradients. The returned `winner` is the feasible element of best
objective.
