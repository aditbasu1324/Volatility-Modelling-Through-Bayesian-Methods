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

## SV: Prior Centering via Linearized (Kalman) Approximation

Since SV's likelihood is intractable (see Why PMCMC Is Required, theoretical.md), there's no direct MLE-style fit available for prior-centering the way EGARCH has. Instead, a standard linearization is used: taking $\ln r_t^2$ turns the (demeaned) observation equation into an approximately linear, approximately Gaussian state-space model, letting a standard Kalman filter (`UnobservedComponents`, fit on prior-period data) produce quick, approximate point estimates for prior-centering.

**The $+1.27$ correction**: $\ln\epsilon_t^2$ (for $\epsilon_t\sim N(0,1)$) is not itself Gaussian — it follows a log-chi-squared distribution with mean $\approx -1.27$, not 0. The Kalman filter's fitted state is corrected by adding this constant back (`kf_vol = sqrt(exp(filtered_state + 1.27))`) to remove this known bias before using the result to center priors.

**Which fitted values are actually reusable**:
- `mu_kf` (mean of the filtered state) and `sigma_eta_kf` (`sqrt(sigma2.level)`) are reasonable, reusable prior-centering values — both estimate genuine SV-model quantities, just via a linear approximation.
- `sigma_v_kf` (`sqrt(sigma2.irregular)`) is **not** reusable — under the true model, $\ln\epsilon_t^2$'s variance ($\pi^2/2$) is a known, fixed constant, not something that should be freely estimated. This parameter's fitted value mostly reflects the linearization's own approximation error rather than a genuine model quantity.

## SV: Particle Filter Implementation

### Systematic Resampling

Resampling uses **systematic resampling** — a single shared random offset (`u = np.random.uniform(0, 1/N) + np.arange(N)/N`) rather than independent multinomial draws per particle — a standard variance-reduction technique relative to plain multinomial resampling, while still resampling each particle with probability proportional to its weight.

### Numerically Stable Log-Likelihood Accumulation

The log-sum-exp trick (subtracting `max_lw` before exponentiating, adding it back after) avoids underflow when computing $\log\left(\frac{1}{N}\sum_n w_t^{(n)}\right)$ — critical since raw likelihood values can be extremely small for implausible particles.

### PMCMC Proposal Tuning

The regression fit uses an adaptive random-walk proposal, widening/narrowing `proposal_std` every 50 iterations (up to iteration 500) to target a 15-20% acceptance rate — the standard target for PMCMC specifically, reflecting that the noisy, particle-filter-estimated likelihood requires a more conservative acceptance rate than exact-likelihood MCMC (e.g. EGARCH's NUTS, which targets ~95% acceptance via `target_accept_prob`).

## PMCMC

See sampling.md for why PMCMC works (the particle filter as an unbiased likelihood estimator, and the pseudo-marginal argument for why substituting this estimate into Metropolis-Hastings still yields exact posterior samples). This section covers how that theory is computed for SV.

### Step 1: Initialize Particles from the Stationary Distribution

$$\ln\sigma_0^{2,(n)} \sim N\left(\mu_h,\ \frac{\sigma_\eta^2}{1-\phi^2}\right), \qquad n=1,...,N$$

- `init_std = sigma_eta / np.sqrt(1 - phi**2)` — the stationary variance of the AR(1) log-variance process, used so particles start already distributed according to the model's own long-run behavior, rather than an arbitrary point.
- `particles = np.random.normal(mu, init_std, N)` — draws all $N$ particles at once as a single vectorized array, not a loop.

### Step 2: Propose $\theta'$

$\theta' = (\mu_h, \phi, \sigma_\eta, \mu_r)$ — proposed jointly via a random-walk step (`theta_proposed = theta_current + np.random.normal(0, proposal_std, 4)`), unlike EGARCH's NUTS, which uses gradient-informed proposals — PMCMC here uses plain Metropolis-Hastings, since the particle filter's likelihood estimate isn't differentiable in the way JAX's autodiff requires.

### Step 3: Run the Particle Filter to Estimate $\hat{p}(\text{data}\mid\theta')$

For each time step $t=1,...,T$, three operations happen in sequence, vectorized across all $N$ particles simultaneously:

**Propagate** — advance every particle one step via the SV recursion:
```python
particles = mu + phi*(particles - mu) + np.random.normal(0, sigma_eta, N)
```
This is $\ln\sigma_t^{2,(n)} = \mu_h + \phi(\ln\sigma_{t-1}^{2,(n)}-\mu_h) + \eta_t^{(n)}$, with each particle drawing its own independent $\eta_t^{(n)}$ — a single vectorized line replaces what would otherwise be a per-particle loop.

**Weight** — score each particle by how well its implied volatility explains the actual observed return:
```python
sigma_t = np.sqrt(np.exp(particles))
log_w   = scipy_stats.norm.logpdf(returns[t], loc=mu_return, scale=sigma_t)
```
This evaluates $\log p(r_t\mid\sigma_t^{(n)})$ for every particle at once — particles whose implied $\sigma_t^{(n)}$ makes the actual return $r_t$ plausible get a higher (less negative) log-weight.

**Numerically stable normalization** (the log-sum-exp trick):
```python
max_lw  = log_w.max()
log_w  -= max_lw
weights = np.exp(log_w)
weights /= weights.sum()
```
Raw likelihood values can underflow to zero for implausible particles (especially with $N$ in the hundreds and many time steps), so the maximum log-weight is subtracted before exponentiating (keeping all values in a numerically safe range), then added back afterward when accumulating the log-likelihood:
```python
log_likelihood += max_lw + np.log(np.mean(np.exp(log_w)))
```
This computes $\log\left(\frac{1}{N}\sum_n w_t^{(n)}\right)$ — the unbiased log-likelihood contribution from this time step (see sampling.md) — without ever exponentiating a raw, potentially-underflowing value directly.

### Step 4: Resample — Systematic Resampling

```python
cumsum = np.cumsum(weights)
u = np.random.uniform(0, 1/N) + np.arange(N)/N
idx = np.searchsorted(cumsum, u)
particles = particles[idx]
```

This is the part worth walking through carefully, since it's a specific, deliberate resampling scheme rather than the most naive option.

**The naive alternative (multinomial resampling)** would draw $N$ independent uniform random numbers and find where each falls in the cumulative weight distribution — but independent draws can, purely by chance, cluster unevenly (e.g., missing a high-weight particle entirely, or oversampling it more than its weight strictly warrants), adding unnecessary extra randomness on top of the randomness already inherent in the weights themselves.

**Systematic resampling instead uses a single shared random offset**: `u = np.random.uniform(0, 1/N)` is drawn **once**, and every subsequent sample point is spaced exactly $1/N$ apart from it: `u, u+1/N, u+2/N, ..., u+(N-1)/N`. This is `np.random.uniform(0, 1/N) + np.arange(N)/N` — a single random number plus a fixed, evenly-spaced grid.

**Why this is better than independent draws**: each particle's *expected* number of resampled copies still exactly equals its weight (so it's still an unbiased resampling scheme), but the *variance* of how many times a given particle actually gets duplicated is lower than under independent multinomial draws — since the $N$ sample points are forced to be evenly spread across $[0,1]$ rather than risking random clustering. This directly reduces the resampling step's own contribution to noise in the overall algorithm.

**`np.searchsorted(cumsum, u)`** is the mechanism that actually performs the resampling: `cumsum` is the cumulative sum of normalized weights (e.g. $[0.1, 0.3, 0.45, 1.0,...]$ for particles with weights $[0.1, 0.2, 0.15, 0.55,...]$) — each particle "owns" an interval of this cumulative range proportional to its own weight. `searchsorted` finds, for each of the $N$ evenly-spaced sample points in `u`, which particle's interval it falls into. A particle with a large weight owns a wide interval, so it's more likely to contain one (or more) of the evenly-spaced sample points — meaning high-weight particles get duplicated multiple times, low-weight particles may not be selected at all, and the resulting `idx` array indexes directly into `particles` to produce the new, resampled particle set.

### Step 5: Combine with the Prior

$$\log\alpha = \left[\hat{p}(\text{data}\mid\theta') \text{ (log)} + \log p(\theta')\right] - \left[\hat{p}(\text{data}\mid\theta)\text{ (log)} + \log p(\theta)\right]$$
```python
log_alpha = (ll_proposed + log_prior(theta_proposed)) - (ll_current + log_prior(theta_current))
if np.log(np.random.uniform()) < log_alpha:
    theta_current = theta_proposed
    ...
```
Exactly the same Metropolis-Hastings acceptance structure as EGARCH's NUTS (see MCMC, above) — the only difference is that `ll_proposed`/`ll_current` here come from the particle filter's *estimated* likelihood, not an exact closed-form one, per the pseudo-marginal argument in sampling.md.

### Where This Differs, Concretely, From EGARCH's MCMC

| | EGARCH (NUTS) | SV (PMCMC) |
|---|---|---|
| Likelihood | Exact, closed-form (Step 3 in MCMC section) | Estimated via particle filter (Steps 3-4 above) |
| Proposal mechanism | Gradient-informed (leapfrog, via JAX autodiff) | Random-walk (no gradient — particle filter likelihood isn't differentiable in this implementation) |
| Target acceptance rate | ~95% (`target_accept_prob=0.95`) | ~15-20% — lower, since the noisy likelihood estimate makes standard high-acceptance-rate tuning inappropriate; adapted every 50 iterations toward this target |
| Rejection of invalid proposals | Priors truncated/bounded (handled inside the model) | Explicit checks before running the particle filter (`np.abs(theta_proposed[1]) >= 1`, `theta_proposed[2] <= 0`) — avoids wasting a full particle filter run on invalid parameters |
| Chain execution | `chain_method='parallel'` (chains run simultaneously across cores) | `chain_method='sequential'` (chains run one after another) — a practical choice, not something affecting correctness |
| Convergence check | Multi-chain R-hat/ESS (automatic via NumPyro) | Single-chain ACF-based mixing check (see interpretation.md) — no multi-chain R-hat available in this hand-written implementation |

