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