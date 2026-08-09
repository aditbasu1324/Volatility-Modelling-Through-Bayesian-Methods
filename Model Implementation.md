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

**Return indexing**: recursion scans over the full `returns` array (all $T$) to produce $\sigma_1,...,\sigma_T$; $\sigma_0$ set directly. The likelihood uses only $\sigma_0,...,\sigma_{T-1}$ — the final term $\sigma_T$ is exposed separately (`numpyro.deterministic("ls2_next", ...)`) as the state after the window's last return, used only for carry-forward.

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
- End-of-regression state for sequential carry-forward: `mcmc_egarch.get_samples()["ls2_next"]` (per draw) — the state after the regression period's last return, exposed directly by the model (see Step 2, above) — no re-simulation needed.

## EGARCH: Sequential (Implementation)

### Forecast Generation — Vectorized NumPy, Not JAX/NUTS
Two forecasts are generated per window, both in plain NumPy (no autodiff needed, since no new parameters are being sampled):

- **Blind (t+126)**: `forward_simulate_egarch_window` runs the recursion vectorized across all posterior draws (parameters as length-$S$ arrays), drawing a fresh `np.random.standard_t` shock per draw at each step, for the whole window in one pass. An extra $h-1$ "overhang" days are simulated past the window's real end purely so the rolling $h$-day HV computation has enough runway to produce a value for every real day, including the last $h-1$ — the overhang itself is discarded afterward.
- **Filtered (t,t+1)**: `filtered_pass_egarch_window` walks the same recursion deterministically through the window's *real* returns instead of random shocks — EGARCH's state is an exact function of $\theta$ and observed returns, so no randomness is needed here. `filtered_sigma[:, t]` is informed only by returns before $t$, never $r_t$ itself. `filtered_forecast_egarch_window` then calls `forward_simulate_egarch_window` fresh for each day $t$, forecasting $h$ days forward from that day's filtered state — this is where randomness re-enters, since the $h$-day-ahead leg is still necessarily blind.

### Refitting — NumPyro/NUTS, with `numpyro.deterministic`
Structurally identical to the regression fit, plus `numpyro.deterministic("sigma", sigma)` so `mcmc_w.get_samples()["sigma"]` returns each draw's full path for that window directly.

### MCMC Settings: Regression vs. Sequential

The regression fit uses 2000 warmup + 2000 samples across 4 parallel chains (16,000 total draws from one NUTS run). Each sequential window refit instead uses 500 warmup + 1000 samples across 2 chains (3,000 total draws), run 8 times (once per window).

This reduction is a computational-cost choice, not one motivated by an observed convergence difference between the two settings — refitting via NUTS 8 separate times (once per window) is considerably more expensive in aggregate than the single regression fit, so per-window settings were scaled down to keep total runtime manageable.

### Carrying State Forward
Next window's start: `ls2_current_seq = samples_w["ls2_next"]` — the just-refit window's own per-draw state after its last return, exposed the same way as the regression fit's `ls2_next` (see EGARCH: Regression, above).

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

### Step 1: Initialize Particles

`particle_filter_sv` supports two initializations via its optional `ls2_init` argument:

- **Stationary** (default, `ls2_init=None`) — used for the regression fit, which has no prior window to carry a state from:
$$\ln\sigma_0^{2,(n)} \sim N\left(\mu_h,\ \frac{\sigma_\eta^2}{1-\phi^2}\right), \qquad n=1,...,N$$
  `init_std = sigma_eta / np.sqrt(1 - phi**2)` (the AR(1) process's stationary variance), `particles = np.random.normal(mu, init_std, N)` — all $N$ particles drawn at once as a vectorized array, not a loop.
- **Warm-started** (`ls2_init` given) — used for every sequential refit: `particles = np.full(N, ls2_init)`, all $N$ particles seeded at a single scalar (the pooled mean across the previous window's per-draw end states — see SV: Sequential (Implementation), below). The very next propagate step (Step 3, below) immediately injects spread via $\eta_t$, so this point-mass start doesn't cause degenerate weights.

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

### `pmcmc_regression` vs. `pmcmc_sequential`: Where They Actually Differ

Both functions share the exact same core loop (propose, run the particle filter, Metropolis-Hastings accept/reject, adapt `proposal_std`) — everything in Steps 2–5 above is identical between them. The differences are narrower than the separate function names might suggest:

| | `pmcmc_regression` | `pmcmc_sequential` |
|---|---|---|
| Prior shape | Hardcoded inside `log_prior` (e.g. `scipy_stats.norm.logpdf(mu_p, mu_init, 1.0)` — std fixed at 1.0) | Passed in as arguments (`mu_prior_std`, `phi_prior_a`/`phi_prior_b`, `sigma_eta_prior_scale`, `mu_return_prior_std`) — a freshly re-centered prior every window |
| State warm-start (`ls2_init`) | Not a parameter at all — always stationary-initializes, correctly, since there's no prior window to carry from | Optional parameter, forwarded to every `particle_filter_sv` call, warm-starting from the previous window's carried state (see Carrying State Forward and Warm-Starting the Refit, below) |
| Input validation | Explicitly raises `ValueError` if `mu_init`/`sigma_eta_init`/`mu_return_init` are `None` | No equivalent check — assumes the caller (the main sequential loop) always supplies real values |
| Proposal-validity check | Two separate `if` blocks (checks $\lvert\phi'\rvert\geq 1$, then separately $\sigma_\eta'\leq 0$) | One combined `if ... or ...` — same two conditions, same outcome, just written more compactly |

The prior-shape difference is the one that actually matters, and it's not incidental: regression runs once, against a fixed, prior-period-derived prior, so a constant shape is correct there. Sequential runs eight times, and each window's prior has to be re-centered on the *previous* window's posterior (computed in the main loop, in Extract priors from previous posterior, before each refit call) — so the shape can't be a hardcoded constant, it has to be a parameter. Everything else in the table — validation, the combined-vs-separate boundary check — is a real difference in the code but not a difference in what either function actually computes. The print labels and `# NEW:` vs. `# FIX:` comment styles (not shown above) are purely cosmetic, left over from separate earlier bug-fixing passes on each function.

## SV: Sequential (Implementation)

### Forecast Generation

Mirrors EGARCH: Sequential (Implementation) structurally, both forecasts in plain NumPy:

- **Blind (t+126)**: `forward_simulate_sv_window` runs `simulate_sv_vectorized` across all posterior draws in one pass, same $h-1$ overhang-then-discard trick as EGARCH for the rolling HV window.
- **Filtered (t,t+1)**: unlike EGARCH, SV's state has its own independent noise and can't be backed out deterministically from returns alone (see theoretical.md, Why Particle MCMC Is Required), so filtering requires a genuine particle filter per posterior draw — not vectorizable across draws the way EGARCH's deterministic recursion is. `particle_filter_sv_predictive` runs one draw's filter, capturing `predictive_ls2[t]` *before* the reweight step folds in $r_t$ (mirroring the same causal boundary as EGARCH's `filtered_sigma`). `filtered_pass_sv_window` loops this over all $S$ draws (the one place SV's forecast generation can't avoid a Python-level loop over draws) to build the full `(S, T_w)` filtered state array. `filtered_forecast_sv_window` then calls `forward_simulate_sv_window` fresh per day, same as EGARCH's filtered forecast.

### The t,t+1 Causal Boundary, Precisely

`particle_filter_sv_predictive`'s loop makes the causal ordering explicit at every step:
```python
for t in range(T):
    particles = mu + phi*(particles - mu) + np.random.normal(0, sigma_eta, N)   # propagate
    predictive_ls2[t] = particles.mean()                                        # recorded BEFORE reweight
    ...
    log_w = scipy_stats.norm.logpdf(window_returns[t], loc=mu_return, scale=sigma_t)  # r_t enters HERE
    ...
    particles = particles[idx]                                                  # resampled, feeds next t's propagate
```
Going into iteration $t$, `particles` already reflects the reweight/resample from iteration $t-1$ — informed through $r_{t-1}$, nothing more. Propagating advances it one step; recording `predictive_ls2[t]` immediately afterward, but *before* reweighting on $r_t$, captures the state informed only by $r_{0..t-1}$. Only after that record does $r_t$ enter, via the reweight step, producing the resampled set that iteration $t+1$'s propagate line will advance next. The t→t+1 transition is this loop boundary itself: each iteration ends informed by that day's real return; the next iteration's first two lines carry it forward and record the result before the *next* day's return touches it.

### Why the Mean Survives Resampling's Duplication, But Variance Wouldn't

Theoretical.md (SV-Specific Sequential Mechanics) notes that at any single time step, the $N$ particles post-resample are not $N$ independent pieces of information, since resampling duplicates high-weight particles. Two things limit how much this affects `predictive_ls2` specifically:

1. **Timing**: `predictive_ls2[t]` is recorded immediately after propagation, which injects an independent $\eta_t^{(n)}$ into every particle — including ones that were exact duplicates entering the step. By the time the mean is taken, that step's duplication has already been broken by fresh independent noise.
2. **Resampling is unbiased for first moments by construction**: each particle's expected number of copies exactly equals its weight, so the resampled set's mean is, in expectation, unbiased for the true weighted mean — duplication costs effective sample size (added noise), not bias.

Variance/spread statistics don't get this same protection — exact duplicates mechanically understate spread (the classic particle-impoverishment problem), a qualitatively worse failure than the mean's added noise. This is why the design never asks the within-draw $N$ particles for spread at all: cross-sample variability (CI bands, forecast spread) comes entirely from varying across the $S$ independent posterior draws instead, each with its own separate, genuinely-independent particle-filter run — never from the $N$ particles within one draw's run.

### Carrying State Forward and Warm-Starting the Refit

Next window's start: `ls2_current_seq_sv = filtered_hist[:, -1]` — `pmcmc_sequential`'s own particle filter already reweights on every day's real return (see PMCMC, Step 3, above), so its last entry already reflects the window's full return history; no extra step is needed here the way EGARCH needed `ls2_next`.

This per-draw carried state is pooled to a single scalar, `ls2_seed_scalar_sv = float(np.mean(ls2_current_seq_sv))`, mirroring EGARCH's own `ls2_seed_scalar`, and passed as `pmcmc_sequential`'s `ls2_init` argument — forwarded to every `particle_filter_sv` call inside that window's refit (both the initial $\theta$ and every proposal), so every likelihood evaluation is scored starting from the true carried-forward state rather than that proposal's own stationary distribution (see PMCMC, Step 1, above).

### A Richer Warm-Start Is Possible With No Code Change to `pmcmc_sequential`/`particle_filter_sv`

The current warm-start pools the previous window's $S$ per-draw end states to a single scalar, so every particle in every likelihood evaluation starts from the same point — the carried-forward *spread* is discarded, only its mean survives. A richer alternative: resample $N$ particles directly from the previous window's empirical distribution of end states, `np.random.choice(ls2_current_seq_sv, size=N, replace=True)`, preserving that spread instead of collapsing it.

Verified (not just assumed): this needs no change to either function. `pmcmc_sequential` never inspects `ls2_init` beyond forwarding it to `particle_filter_sv` (it appears nowhere else in the function body). And `particle_filter_sv`'s `particles = np.full(N, ls2_init)` already broadcasts correctly whether `ls2_init` is a scalar or a length-$N$ array — confirmed directly: `np.full(5, np.array([1,2,3,4,5]))` returns the array unchanged, not an unexpected fill-value error. So the entire change would be a single line at the call site in the main sequential loop, replacing the `np.mean(...)` with `np.random.choice(...)`. Not yet implemented — flagged for later, alongside the analogous (but structurally harder — see EGARCH's `beta_egarch`/NUTS-stability discussion) question of whether EGARCH's refit-seed pooling could be improved the same way.

