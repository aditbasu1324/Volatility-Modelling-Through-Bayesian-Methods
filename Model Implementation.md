## Model Implementation

This explains how the theory behind MCMC/PMCMC is actually computed for this project's models.

## MCMC

See theoretical.md for why MCMC works (Bayes' theorem, Markov chains, detailed balance, Metropolis-Hastings, why the normalizing constant cancels). This section covers how that theory is computed for EGARCH.

### Step 1: Propose $\theta'$
$\theta' = (\mu_{ret}, \alpha, \beta, \gamma, \nu, \sigma_\infty)$ — all 6 sampled jointly (see NUTS below for how).

### Step 2: Recurse Through the Volatility Equation
$$\log\sigma_t^2 = \omega + \beta\log\sigma_{t-1}^2 + \gamma z_{t-1} + \alpha(|z_{t-1}|-E|z_{t-1}|), \quad z_{t-1}=\frac{r_{t-1}-\mu_{ret}}{\sigma_{t-1}}$$

- Sequential dependency: $\sigma_t$ requires $\sigma_{t-1}$, back to $\sigma_0$. Implemented via `jax.lax.scan` — compiled, differentiable equivalent of a `for` loop.
- $\log\sigma_t^2$ clipped to $[-20,2]$ to prevent divergence during poor early proposals.
- $E|z_{t-1}|$ (a function of $\nu$) precomputed on a grid and interpolated — avoids unstable autodiff through the exact Gamma-function formula.

### Step 3: Compute $p(\text{data}\mid\theta')$
$$\log p(\text{data}\mid\theta') = \sum_{t=0}^{T-1}\log p(r_t\mid\sigma_t,\theta'), \quad p(r_t\mid\sigma_t,\theta')=\frac{1}{\sigma_t}f_{t_\nu}\left(\frac{r_t-\mu_{ret}}{\sigma_t}\right)$$
Computed via `numpyro.sample("obs", dist.StudentT(nu, mu_ret, scale), obs=returns)` — evaluates and sums the Student-t log-density at every return in one call.

### Step 4: Combine with the Prior
$\log\pi(\theta') = \log p(\text{data}\mid\theta') + \log p(\theta')$ — the prior term is cheap, closed-form; Steps 2–3 (the recursion) dominate cost.

### Where NUTS Differs, Concretely for This Model

See sampling.md for the general NUTS mechanism (gradient-informed proposals, adaptive trajectory length/step size). Here, $\nabla_\theta\log\pi(\theta)$ is obtained via JAX's automatic differentiation through the entire `jax.lax.scan` recursion (Step 2) and the `numpyro.sample("obs", ...)` likelihood (Step 3) — computing how a change in $\alpha,\beta,\gamma,\nu,\mu_{ret},\sigma_\infty$ affects the final log-posterior, with no derivatives manually derived.

**Return indexing**: recursion scans over `returns[:-1]` to produce $\sigma_1,...,\sigma_{T-1}$; $\sigma_0$ set directly. Full returns (all $T$) used for the likelihood.

### Known Approximations
- **$\omega$ uses fixed $\hat\beta_{egarch}$, not sampled $\beta$, within NUTS-fitted models** (regression fit and each sequential refit) — avoids divergences during gradient-based sampling. Does not apply to the plain-NumPy filtering/PIT computations, which have no such stability constraint. Affects only the early path and each draw's precise long-run level; doesn't affect overall validity.

### Prior Implementation
**Percentile band construction**: `sigma_sq_windowed_per_path` computes each path's own window-aggregated $\sigma^2$ independently (looping over paths), before `np.nanpercentile` takes percentiles across paths at each day — see theoretical.md for why this requires a different order of operations than the mean line.

## EGARCH: Regression (Implementation)

### MLE Fit Values Used to Center the Priors

Point estimates from an MLE fit (via the `arch` package, on prior-period data) are used to center several priors' means (e.g. $\hat\alpha, \hat\gamma, \hat\nu, \hat\mu$). For $\beta$ specifically, the MLE fit gave $\hat\beta_{egarch}\approx 0.94$ — informing the choice of $\text{Beta}(47,3)$ as a prior with mean $\frac{47}{50}=0.94$ and concentrated mass near that value.

- $\sigma_t$ obtained directly from the NUTS-fitted posterior, run through the actual regression-period returns (no simulation needed — see theoretical.md for why: no future uncertainty exists for already-observed data).
- `numpyro.deterministic("sigma", sigma)` captures the full per-draw path, avoiding a separate post-hoc re-simulation.
- $\hat\sigma_t^{model}$: average $\sigma_t^2$ across posterior draws *before* the window-RMS aggregation (Jensen's inequality — see theoretical.md), via `sigma_to_hv_posterior`.
- End-of-regression state for sequential carry-forward: `sigma_posterior_egarch[:, -1]` (per draw) — no re-simulation needed, since the original per-draw path already contains it.

## EGARCH: Sequential (Implementation)

### Forecast Generation — Vectorized NumPy, Not JAX/NUTS
`forward_simulate_egarch_window` runs the recursion in plain NumPy, vectorized across all posterior draws (parameters as length-$S$ arrays). Fresh `np.random.standard_t` shock drawn per draw at each step. No autodiff needed here (no sampling of new parameters), so plain NumPy suffices.

### Refitting — NumPyro/NUTS, with `numpyro.deterministic`
Structurally identical to the regression fit, plus `numpyro.deterministic("sigma", sigma)` so `mcmc_w.get_samples()["sigma"]` returns each draw's full path for that window directly.

### MCMC Settings: Regression vs. Sequential

The regression fit uses 2000 warmup + 2000 samples across 4 parallel chains (16,000 total draws from one NUTS run). Each sequential window refit instead uses 500 warmup + 1000 samples across 2 chains (3,000 total draws), run 8 times (once per window).

This reduction is a computational-cost choice, not one motivated by an observed convergence difference between the two settings — refitting via NUTS 8 separate times (once per window) is considerably more expensive in aggregate than the single regression fit, so per-window settings were scaled down to keep total runtime manageable. <!-- TODO: check R-hat/ESS across sequential window refits vs. the regression fit, to confirm reduced settings haven't compromised convergence quality -->

### Carrying State Forward
Next window's start: `ls2_current_seq = np.log(samples_w["sigma"][:, -1]**2)` — the just-refit window's own per-draw end state (not the previous window's stale carry-in).

### Approximate Prior Updating
`fit_beta_params`/`fit_gamma_params` approximate posterior samples with a parametric distribution (method-of-moments for Gamma; MLE fit for Beta) — a cheap substitute for exact conjugate updating, which isn't available for this nonlinear recursion (see theoretical.md).