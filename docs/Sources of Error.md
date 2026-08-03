### Why Isn't $\sigma_t^{model}$ Simply the Truth?

$\sigma_t$ is latent in EGARCH/SV — never directly observed, only inferred via a Bayesian posterior over an assumed data-generating process. $\sigma_t^{model}$ cannot be treated as ground truth, for three distinct reasons:

1. **Irreducible posterior uncertainty** — even under a perfectly specified model, a posterior conditioned on finite data has nonzero variance (this is exactly what credible interval width reflects).
2. **Misspecification** — Bayesian posterior consistency (concentration around the true parameter) holds only if the model's assumed dynamics genuinely match the true process, and only asymptotically. Under misspecification, the posterior instead concentrates around a *pseudo-true* value (the closest model-consistent approximation to the truth), which need not equal the true volatility path.
3. **Numerical approximation error** — finite MCMC samples (and, for PF-SV, finite particles) introduce Monte Carlo error beyond the idealized posterior.

This is why $\sigma_t^{model}$ carries its own error (genuine model error, see metrics.md Sources of Error) distinct from — but structurally analogous to — the estimation error established for $\hat\sigma_t^{empirical}$: both are estimates of an unobservable quantity, subject to their own respective sources of noise. If regression-period testing reveals poor fit (Why Regression Testing?, above), reason 2 (misspecification) is the most likely candidate, since regression is the best-case, full-information scenario.

## Sources of Error: A Consolidated Overview

This project's evaluation framework involves many distinct sources of error, introduced individually alongside the specific metric or model discussion where each first arises. This section collects them in one place for reference, organized by what they pertain to.

### Errors in the Empirical Benchmark
- Systematic estimation error (finite-sample noise, $h=21$)
- Constant-volatility-within-window assumption

### Errors in the Model's Posterior
- Irreducible posterior uncertainty (present even if the model is correctly specified)
- Pseudo-true bias (if the model is misspecified)
- Numerical/Monte Carlo approximation error

### Errors from Combining Benchmark and Model
- Inseparability of $\hat\sigma_\eta$ into its benchmark-noise and model-error components
- Weak independence assumption, likely to fail specifically during crisis periods

### Structural Limitations of Specific Metrics
- HV/QLIKE cannot detect leverage at all (sign-blind by construction)
- PIT cannot separate wrong volatility level from wrong innovation shape
- Sequential forecasts use $\mathcal{F}_{\text{window start}}$, not genuine per-day $\mathcal{F}_t$
- Jensen's-inequality gap between mean forecast and percentile band (~10% for EGARCH, quantified)

Each of these is derived in full where it's first introduced (cross-referenced above); this section exists as a navigational summary, not a replacement for those derivations.

Limitations and Sources of Error from this Project

# Full List of Limitations, by Section

## Data / Historical Volatility Benchmark
- Constant-volatility-within-window assumption — the benchmark's own validity depends on volatility being roughly constant over the 21-day window, in tension with time-varying models.
- Finite-sample estimation error — $\hat\sigma_t^{empirical}$ is a 21-day estimator, carrying residual sampling variance regardless of model correctness (LLN only guarantees convergence as $h\to\infty$).
- Terminology/construction gap vs. classical realized volatility — this project's HV uses daily returns, not intraday quadratic variation; doesn't require the classical high-frequency construction, but has lower statistical power as a result.

## Log-Normal Noise Model ($\hat\sigma_\eta$)
- Inseparability — $\hat\sigma_\eta$ is a combined estimate of benchmark noise and model error; the two cannot be separated from one computable quantity.
- Weak independence assumption — $\eta_t\perp\epsilon_t^{model}$ is only weakly justified; both plausibly rise together during crisis periods, meaning $\hat\sigma_\eta$ likely understates true combined variance exactly when it matters most.

## Metrics, General
- PIT/KS jointly tests level and shape, can't separate the two.
- HV/QLIKE structurally cannot detect leverage at all (symmetric under sign/reordering).
- Volatility persistence has no direct test (ACF tests are indirect/non-diagnostic for this specifically).
- No single test suffices — volatility is latent, so no metric (or combination) confirms the model matches the unobservable truth exactly.
- Remaining uncovered dimensions: tail-specific/VaR calibration, multi-horizon path accuracy, economic/trading value, full within-window timing accuracy (only partially covered via PIT).

## Baseline: Rolling Average
- No explicit law for how volatility evolves beyond time $t$ (forecast rule is an imposed convention, not derived).
- Structurally cannot react to a genuine regime change within its own forecast window (consequence of the constant-forecast rule).
- Not parametric, not Bayesian — no native credible intervals (addressed via the noise-only interval, itself limited to that one source of uncertainty).

## Why Isn't $\sigma_t^{model}$ Simply the Truth? (EGARCH/SV posteriors generally)
- Irreducible posterior uncertainty, even under a correctly specified model.
- Pseudo-true bias under misspecification — posterior concentrates on the closest model-consistent approximation, not necessarily the truth, even asymptotically.
- Numerical/Monte Carlo approximation error from finite samples/particles.

## Filtering vs. Smoothing (Sequential Forecasting, general)
- Sequential forecast generation conditions on $\mathcal{F}_{\text{window start}}$, not genuine per-day $\mathcal{F}_t$ — a documented simplification, not the theoretically ideal per-day re-conditioning (which would be more expensive and isn't implemented).
- Regression and sequential-refitting posteriors are both "smoothed," not genuinely filtered — parameters fit via full-sample (or full-window) likelihood necessarily incorporate information from the future relative to any individual day within that sample/window.

## EGARCH-Specific
- Approximate posterior updating (sequential refitting) — no exact conjugate update exists for this nonlinear recursion; each window's posterior is approximated via a parametric refit (Beta/Gamma/Normal), a loss of fidelity relative to carrying the true posterior forward exactly.
- Fixed-$\hat\beta_{egarch}$ stability trade-off — $\omega$ uses the MLE point estimate, not each draw's own sampled $\beta$, to avoid divergences; individual draws' implied long-run variance can deviate slightly from $\sigma_\infty^2$ as a result.
- Reduced MCMC settings for sequential refits (warmup/samples/chains) — a computational-cost choice, not one confirmed by convergence diagnostics to be safe (flagged as an open TODO).
- $\hat\sigma_\eta$ inseparability/independence limitations (see general Noise Model section) apply here too.

## SV-Specific
- Same approximate posterior updating and $\mathcal{F}_t$ limitations as EGARCH (inherited, not independently re-derived).
- Particle filter likelihood is only ever an unbiased *estimate*, never exact — more particles reduce noise but never eliminate it; too few particles can slow PMCMC's mixing even though correctness is preserved.
- Weaker sampler — random-walk Metropolis-Hastings (not gradient-informed NUTS), since the particle filter's likelihood isn't differentiable; generally slower-mixing, checked via single-chain ACF diagnostics rather than multi-chain R-hat.
- Prior-centering via a linearized (Kalman) approximation is itself approximate — the observation noise isn't truly Gaussian (log-chi-squared, corrected by the $+1.27$ constant), and one resulting parameter (`sigma_v_kf`) is confirmed not meaningful/reusable at all.
- No leverage term and Gaussian (not fat-tailed) innovations by construction — a deliberate simplification relative to EGARCH, testable via Engle-Ng/PIT but not fixable within SV's own specification.

## Comparing EGARCH vs. SV
- A pooled aggregate comparison (e.g. overall QLIKE) mixes together two independent design differences (explicit stylized-fact encoding vs. deterministic/latent volatility) — cannot, by itself, attribute a performance gap to either cause without reading alongside each model's own specific diagnostics.

## PMCMC Mechanics
- Particles at an individual time step, mid-filter, are not mutually independent (resampling duplicates/discards based on fit to nearby data) — unlike EGARCH's posterior draws, which are independent throughout. This equivalence to EGARCH only holds for the fully-completed, per-draw path, not intermediate particle states.



# Final Breakdown: Sources of Error, Limitations, and Actionable Items

Three genuinely different categories emerged from this discussion. Distinguishing them matters: they call for different kinds of response (documentation vs. an actual re-run vs. accepting a hard limit).

---

## Category A: Sources of Error (inherent to the method, documented where first introduced)

Errors that exist by the nature of the approach, correctly acknowledged in their own section, not something a re-run would eliminate.

**Empirical benchmark**
- Systematic estimation error (finite-sample, $h=21$)
- Constant-volatility-within-window assumption

**Model posterior**
- Irreducible posterior uncertainty
- Pseudo-true bias under misspecification
- Numerical/Monte Carlo approximation error

**Combining benchmark + model**
- Inseparability of $\hat\sigma_\eta$ into benchmark-noise vs. model-error
- Weak independence assumption between $\eta_t$ and $\epsilon_t^{model}$

**Structural metric blind spots**
- HV/QLIKE cannot detect leverage (sign-blind by construction)
- PIT cannot separate wrong level from wrong shape
- Sequential forecasts use $\mathcal{F}_{\text{window start}}$, not genuine per-day $\mathcal{F}_t$
- Jensen's-inequality gap between mean forecast and percentile band (~10% for EGARCH — the one item here actually *quantified*)

---

## Category B: Structural Limitations (not fixable by any test within this project)

Not a source of error to quantify — a hard boundary on what's knowable or achievable given the project's scope.

- **EGARCH vs. SV bundled design differences** — needs a 4th model to isolate; not resolvable with the current three models.
- **"No single test suffices"** — volatility is permanently latent; no metric battery, however large, confirms a match to the unobservable truth.
- **Pseudo-true bias unverifiable directly** — the truth is never observable, so convergence to it (vs. a pseudo-true value) can't be directly confirmed regardless of data/model size.
- **Constant-volatility benchmark fix requires intraday data** — theoretically fixable, but not with data currently available.

---

## Category C: Acknowledged But Never Actually Tested (the actionable list)

Each of these has a **specific, already-designed check named somewhere in this project** — but the check itself was never run against actual results. This is the list worth prioritizing if there's time to do more work.

| # | Item | The check that was named but not run |
|---|---|---|
| 1 | Noise independence assumption | Cross-model $\hat\sigma_\eta$ comparison (EGARCH vs. SV vs. baseline) |
| 2 | Numerical/MCMC approximation error | Repeated-run (different-seed) variance check |
| 3 | Window size/count arbitrariness ($h=21$; 8 windows) | Sensitivity re-run at alternative values (e.g. $h=15/30$; 4/12 windows) |
| 4 | Particle count adequacy ($N$) | Fixed-$\theta$ repeated-likelihood-variance diagnostic |
| 5 | Sequential MCMC settings adequacy | R-hat/ESS comparison vs. regression settings (already flagged as a TODO) |
| 6 | Parametric-refit quality (sequential posterior approximation) | KS test/QQ-plot: fitted parametric prior vs. actual previous-window posterior samples |

---

## Summary: What Each Category Means for the Write-Up

- **Category A** → stays exactly where it is, in each section's own derivation; this final list is just a navigational index (already drafted as "Sources of Error: A Consolidated Overview").
- **Category B** → worth one honest paragraph in a "Scope and Limitations" section, stating plainly these aren't resolvable here, and why — not framed as an oversight.
- **Category C** → the genuine to-do list. Each row is a concrete, boundable task (rerun with a different seed/parameter/comparison) rather than an open-ended limitation — worth flagging explicitly as "future work" or actually executing if time allows, since these are the only items in the whole discussion where you already know exactly what to do and just haven't done it yet.

### Additional $\mathcal{F}_t$-Related Items For Historical Volatility

- **Regression (and sequential refitting) posteriors are "smoothed," not genuinely filtered**: parameters are fit using the entire regression period (or entire window)'s likelihood at once, so even a "backward-looking" recursion's $\sigma_t$ at any point uses parameters informed by data *after* that point too. Nothing in this project produces a genuinely fully-filtered $\sigma_t$ path — only the window-to-window boundary (not looking at *other* windows) is enforced.
- **Prior predictive check conditions on $\mathcal{F}_t=\emptyset$, not real information**: distinct from the sequential forecast's $\mathcal{F}_{\text{window start}}$ approximation — this is a deliberately *empty* conditioning set, not an approximation of a nonempty one, since the whole point is checking what the prior implies before any data is seen.
- **SV particles are not mutually independent mid-filter** (resampling-induced lineage/duplication) — the equivalence to EGARCH's independent posterior draws only holds for the fully-completed, per-draw path, not intermediate particle states.

