## Theoretical Section

This file stores information on the theoretical parts of this project.

## Volatility

- Volatility = standard deviation of daily log returns.
- Unlike price, volatility isn't published by the market — it's an unobservable ("latent") quantity.
- Consequence: a latent volatility model can only be evaluated indirectly, via metrics designed for this (there is no single direct test)

## Stylized Facts of Returns and Volatility

Before introducing specific metrics, it's worth stating the empirical properties of returns/volatility that any good model — and by extension, any good test of that model — needs to engage with:

- **Level**: volatility is time-varying and its magnitude matters directly (position sizing, option pricing).
- **Uncertainty**: volatility is never known with certainty, even by the model itself — a legitimate model should express a distribution, not just a point estimate.
- **Shape**: return distributions are not Gaussian — they exhibit fat tails (excess kurtosis) relative to Normal.
- **Volatility clustering**: large changes tend to follow large changes (of either sign) — formally, raw returns show little autocorrelation, but squared/absolute returns show strong, persistent autocorrelation. This is the single most fundamental stylized fact motivating GARCH-family and stochastic volatility models.
- **Leverage effect**: negative returns tend to increase future volatility more than positive returns of equal magnitude.
- **Volatility persistence / long memory**: volatility shocks decay slowly, with effects lingering over many periods.

Each metric below is designed to test whether a model has correctly captured one or more of these properties — not as an arbitrary battery of statistical procedures, but as a direct check against the known empirical structure of real markets.

## Metrics Overview

The different metrics used and their purposes are listed below.

### Historical Volatility Comparison
**Point**: check how closely the model's volatility matches reality across a window.

- **QLIKE (mean loss)**: mean pointwise loss, weighted to penalize underestimating volatility more heavily (generally more costly than overestimating).
- **HV Coverage**: checks if the model's stated confidence intervals actually contain the empirical value at the expected rate.

### Return Distribution Calibration
**Point**: checks if the model's implied return distribution matches the actual return distribution — a day-by-day test, independent of the window-level comparison above and of the noise correction used in HV coverage.

- **PIT/KS**: tests whether the model's day-by-day claimed uncertainty matches what actually happens (jointly tests volatility level and assumed shape — can't separate the two)
- KS test / histogram: formal and visual versions of the same check.

### Serial Dependence Diagnostics
**Point**: checks if the model's errors are predictable over time — a model can be right on average while still failing in a systematic, exploitable pattern.

- **ACF of $z_t$**: tests for directional persistence in errors.
- **ACF of $z_t^2$**: tests for volatility-clustering persistence in errors.
- **Engle-Ng sign-bias test**: tests if errors depend on the sign of the previous return (leverage effect).

These metrics — including remaining test-coverage gaps and why no single test suffices — are covered in more detail in the metrics.md section.

### Why Regression Testing?

- Best-case scenario: parameters fit with full knowledge of the returns being evaluated.
- Diagnostic value: separates two failure modes that pure out-of-sample testing would conflate:
  - **Model misspecification** — fails even in-sample (see Why Isn't $\sigma_t^{model}$ Simply the Truth?, below, for what happens if this occurs)
  - **Generalization failure** — works in-sample, fails out-of-sample
- These call for different fixes (respecify the model vs. improve adaptation), so separating them matters.

### Why Isn't $\sigma_t^{model}$ Simply the Truth?

$\sigma_t$ is latent in EGARCH/SV — never directly observed, only inferred via a Bayesian posterior over an assumed data-generating process. $\sigma_t^{model}$ cannot be treated as ground truth, for three distinct reasons:

1. **Irreducible posterior uncertainty** — even under a perfectly specified model, a posterior conditioned on finite data has nonzero variance (this is exactly what credible interval width reflects).
2. **Misspecification** — Bayesian posterior consistency (concentration around the true parameter) holds only if the model's assumed dynamics genuinely match the true process, and only asymptotically. Under misspecification, the posterior instead concentrates around a *pseudo-true* value (the closest model-consistent approximation to the truth), which need not equal the true volatility path.
3. **Numerical approximation error** — finite MCMC samples (and, for PF-SV, finite particles) introduce Monte Carlo error beyond the idealized posterior.

This is why $\sigma_t^{model}$ carries its own error (genuine model error, see metrics.md Sources of Error) distinct from — but structurally analogous to — the estimation error established for $\hat\sigma_t^{empirical}$: both are estimates of an unobservable quantity, subject to their own respective sources of noise. If regression-period testing reveals poor fit (Why Regression Testing?, above), reason 2 (misspecification) is the most likely candidate, since regression is the best-case, full-information scenario.

### Why Sequential Updating?

- Genuinely out-of-sample provided the fitting process uses only data prior to each window (see Filtering vs. Smoothing, below).
- The most stringent test: can the model forecast using only current information?

### Filtering vs. Smoothing: What "Genuinely Out-of-Sample" Actually Requires

Sequential updating's core idea is simple: each window carries forward the previous window's posterior, and propagates the recursion forward through the current window. The subtlety is that there are two distinct things this could mean, serving two different purposes:

- **Forecasting** (before the window's data is available): propagate using *simulated* shocks, since the real returns for this window haven't happened yet from the perspective of a genuine forecast.
- **Refitting** (after the window's data is observed): propagate using the *real* returns, to update the posterior for the next window.

Using real returns to generate what's claimed to be a "forecast" would be a form of look-ahead bias — a day early in the window would be evaluated using a $\sigma_t$ that implicitly depends on later days' real returns within the same window (since the refitting recursion runs sequentially through that same window's real data). This project therefore keeps the two processes separate: forecast generation is used for *all* evaluation metrics (QLIKE, RV coverage, PIT, serial-dependence diagnostics); refitting is used only to prepare the next window's prior and produces no evaluation output itself.

This means every evaluation metric shares the same limitation: the theoretically ideal quantity is $E[\sigma_{t+i}^2\mid\mathcal{F}_t]$, freshly conditioned at each real day $t$; what is computed instead is $E[\sigma_{t+i}^2\mid\mathcal{F}_{\text{window start}}]$ for the whole window at once, from one simulation per posterior draw. Forecast quality may degrade for days further from the window's start, since they use increasingly stale conditioning information. A fully rigorous version would branch a fresh, short forward simulation from each day's real, filtered state — computationally more expensive, and not implemented here.

## Baseline: Rolling Average

$$
    \hat{\sigma}_t^{roll} = \sqrt{\frac{1}{h-1}
    \sum_{i=0}^{h-1}(r_{t-i} - \bar{r}_t)^2}
$$

where $\bar{r}_t = \frac{1}{h}\sum_{i=0}^{h-1} r_{t-i}$, $h=21$.

**Formal specification:**
$$r_t = \mu + \sigma_t^{roll}\,\varepsilon_t, \qquad \varepsilon_t\overset{iid}{\sim}N(0,1)$$
$$(\sigma_t^{roll})^2 = \frac{1}{h-1}\sum_{i=0}^{h-1}(r_{t-i}-\bar r_t)^2$$

Unlike EGARCH/SV, the "volatility equation" is a fixed, deterministic window function of past returns — no parameters to estimate, fully determined once the last $h$ returns are known. $\sigma_t^{roll}$ is genuinely time-varying as $t$ advances, but the baseline has no explicit law for how it evolves *beyond* time $t$.

**Forecast rule**: $\sigma_{t+i}^{roll,\,forecast} := \sigma_t^{roll}$ for all $i=1,...,h$ (no-change extension) — reduces the RMS/martingale comparison to the degenerate constant-volatility case (see Historical Volatility). A standard convention for naive point forecasts, not a property inherent to the estimator itself.

**Key properties:**
- Naive baseline: no model assumptions, trivially computable.
- Not parametric → no in-sample/out-of-sample distinction; regression and sequential values computed identically.
- Not Bayesian → no native credible intervals.

### Applying the Metrics to the Baseline

The following applies to both the regression and sequential settings, unless noted otherwise below.

- **QLIKE**: applies directly to $\hat\sigma_t^{roll}$ vs $\hat\sigma_t^{empirical}$ — no distributional assumption needed.
- **HV Coverage**: no native posterior, so the interval is constructed entirely from the log-normal noise model ($\hat\sigma_\eta^{roll}$, fit the same way as for the Bayesian models, substituting $\hat\sigma_t^{roll}$ for $\hat\sigma_t^{model}$). Unlike EGARCH/PF-SV, this noise-model interval is the *sole* source of interval width — there is no posterior uncertainty to combine it with.
- **PIT/KS**: requires an externally imposed distributional shape, since the baseline gives only a point estimate. $r_t \sim N(0,(\hat\sigma_t^{roll})^2)$ is assumed — matching PF-SV's Gaussian innovation assumption, so PIT/KS results between baseline and PF-SV test the same joint hypothesis (see PIT/KS caveat).
- **ACF diagnostics / Engle-Ng**: computed identically to the other models, using $z_t=\Phi^{-1}(u_t)$ from the baseline's own PIT transform above.

#### Regression Setting

$\hat\sigma_\eta^{roll}$ is fit once, globally, over the whole regression period. Results (QLIKE, RV coverage) are additionally broken down into pre-COVID, COVID, and post-COVID sub-periods, since COVID falls within this period and a pooled average can mask sub-period miscalibration.

#### Sequential Setting

Two considerations arise that don't apply to the regression setting:

**Window-by-window breakdown, not COVID sub-periods**: the sequential period (2022–2025) contains no comparable single crisis event, so results are instead broken down into eight fixed six-month windows spanning the full period, in addition to pooled values — the same rationale as the COVID breakdown, applied as a general-purpose decomposition rather than one tied to a specific known event.

**$\hat\sigma_\eta$ must be estimated sequentially, not fit once globally**: using one global $\hat\sigma_\eta$ for the entire sequential period would violate the out-of-sample principle underlying sequential updating — the noise estimate for an early window would be silently informed by residuals from windows that haven't happened yet. $\hat\sigma_\eta$ is instead computed as an expanding, past-only estimate: the first sequential window uses $\hat\sigma_\eta$ carried forward from the regression period; each subsequent window's $\hat\sigma_\eta$ is recomputed using all residuals accumulated from the regression period plus every *completed* sequential window — never including the window currently being evaluated. This expanding-window convention was chosen (over a fixed rolling window or exponential smoothing) for being parameter-free, avoiding an arbitrary window length or decay rate.

This affects HV coverage only — $\hat\sigma_\eta$ has no role in the models' own fitting/forecasting process (EGARCH's MCMC recursion, PF-SV's particle filtering, or the baseline's point estimate); it is purely a post-hoc correction applied when checking the model's output against the empirical benchmark.

QLIKE, PIT/KS, and ACF/Engle-Ng diagnostics are otherwise unchanged from the regression setting — these operate per-day (or per-window as a simple aggregation of per-day values) and involve no noise-model fitting of their own.

## Priors

Before covering EGARCH and SV individually, it's worth establishing the concept and purpose of priors more generally.

Priors are required for Bayesian models, since priors combined with the likelihood of new data are what generate posteriors.

Theoretically, priors shouldn't matter much once enough data is available — the likelihood should dominate. In practice, however, these complex models require reasonably informative priors for the numerical methods (MCMC/NUTS) to converge well.

Priors must also respect the model's **stationarity conditions**, otherwise the model isn't theoretically valid for prediction. Stationarity means the model is stable — that a well-defined long-run variance $\sigma_\infty^2$ exists.

Priors are validated through **prior predictive checking** — simulating data from the prior alone and checking it looks plausible given the actual data being modeled.

## Prior Predictive Checking

### What It Checks

Before conditioning on any real data, a prior predictive check asks: **if only the prior were true, what range of outcomes would the model expect to see?** This is done by simulating many complete synthetic datasets purely from draws of the prior distribution, then comparing the resulting simulated data against the actual observed data.

This is a check on the prior *alone* — no posterior or fitting is involved. If real, observed data falls far outside the range the prior implies as plausible, that's evidence the prior is misspecified (too tight, wrong location, or otherwise unreasonable) before any model fitting has even begun.

### Why Full Path Simulation Is Required

Since $\sigma_t$ in EGARCH/SV is recursively defined — $\sigma_{t+i}$ depends on $\sigma_{t+i-1}$, which depends on $\sigma_{t+i-2}$, and so on — and no data has been observed to pin down *any* of these values, every $\sigma_t$ in the prior predictive check is uncertain from the very first time step onward (unlike sequential forecasting, where $\sigma_t$ itself is already known and only $\sigma_{t+i}$ for $i>1$ is uncertain). This means a single deterministic path cannot represent "what the prior implies" — the prior implies an entire *distribution* over possible paths, which can only be characterized by simulating many independent paths forward, each with its own draw of parameters and shocks.

For each simulated path $n$: draw parameters from the prior, then propagate the recursion forward one step at a time, drawing a fresh shock $z_{t+i}^{(n)}$ at each step and computing $\sigma_{t+i}^{(n)}$ from $\sigma_{t+i-1}^{(n)}$ and $z_{t+i}^{(n)}$. This forward propagation, repeated over many independent paths, automatically incorporates the full recursive chain back to $t$ — no separate backward or analytical step is needed, since each path is itself a valid, self-contained realization of the recursion.

### Correct Order of Aggregation

To compare against the historical volatility benchmark (a 21-day RMS-style quantity), the simulated paths must be aggregated in the correct order:

$$\hat\sigma_t^{prior} = \sqrt{\frac{1}{h}\sum_{i=1}^h \underbrace{\frac{1}{N}\sum_{n=1}^N(\sigma_{t+i}^{(n)})^2}_{\text{average across paths, per time step}}}$$

The average across simulated paths must be taken **before** the square root and window aggregation — averaging *after* taking the square root of each individual path's own RMS instead computes $\frac{1}{N}\sum_n\sqrt{\frac{1}{h}\sum_i(\sigma_{t+i}^{(n)})^2}$, which is systematically biased downward relative to the correct quantity, by Jensen's inequality ($E[\sqrt{X}] \le \sqrt{E[X]}$, since $\sqrt{\cdot}$ is concave). This is the same "expectation before aggregation" principle established for sequential forecasting — the model estimator averages *conditional variances* across uncertainty, not raw per-path point values.

### Constructing a Plausibility Band (Not Just the Mean)

The mean line above requires averaging across paths *before* window-aggregating (per the derivation above). A percentile band, however, requires the **opposite order**: since averaging across paths first would already collapse away the path-to-path variability the band is meant to display, each path must instead be window-aggregated **independently first** — computing that single path's own $\frac{1}{h}\sum_i (\sigma_{t+i}^{(n)})^2$ in isolation — and only *then* are percentiles taken across the resulting per-path values, at each fixed day.

In short: the mean answers "what is the expected window-RMS volatility?" (average first, aggregate second); the band answers "how much do different simulated paths disagree about that window-RMS value?" (aggregate first, per path, then examine the resulting distribution across paths). These are not the same computation applied twice — they require genuinely different orders of operations, for the same reason the mean's own ordering matters (Jensen's inequality): collapsing the path dimension too early destroys the very information the other quantity needs.

### Additional Checks Alongside Path Comparison

- **Rejection rate**: paths violating stationarity ($|\beta|\ge 1$ for EGARCH, analogously for SV) or producing explosive/invalid returns are discarded; the fraction rejected indicates how tightly the prior enforces stationarity.
- **Return-level check**: raw simulated returns (no aggregation needed, since returns are never aggregated) are compared directly against actual daily returns, checking plausibility at the daily scale alongside the volatility-scale check above.

### General Approach to Developing Priors

1. Use volatility theory and stationarity conditions to decide on distribution *types*.
2. Use MLE fits (or suitable approximations, for the SV model) to determine mean parameters.
3. Test variances via prior predictive checking.

## EGARCH(1,1)-t

$$r_t = \mu + \epsilon_t, \quad \epsilon_t = \sigma_t z_t, \quad z_t \overset{\text{i.i.d.}}{\sim} t_\nu(0,1)$$
$$\ln\sigma_t^2 = \omega + \alpha\left(|z_{t-1}| - \mathbb{E}|z_{t-1}|\right) + \gamma z_{t-1} + \beta\ln\sigma_{t-1}^2$$

<!-- FIXME: confirm citation key — original EGARCH is Nelson (1991); verify the "Patton, 2011" attribution is not misapplied here (Patton's actual contribution is QLIKE robustness, not the EGARCH model itself) -->

**Why EGARCH, specifically:**

- **Leverage effect** ($\gamma$): captures the asymmetric response of volatility to positive vs. negative returns of equal magnitude — the key differentiator from symmetric GARCH-family models. This is the direct model-level mechanism whose absence was empirically confirmed for the rolling baseline via the Engle-Ng sign-bias test.
- **Fat tails** ($t_\nu$ innovations): captures excess kurtosis in returns, relative to a Gaussian assumption — directly relevant to the PIT/KS shape assumption (see PIT/KS caveat: EGARCH uses a scaled Student-t $F_t$, not Normal).
- **Sudden large price changes** ($\alpha$): captures the magnitude response of volatility to large shocks, regardless of sign.

### EGARCH Priors

| Parameter | Prior |
|---|---|
| $\mu$ | $\mathcal{N}(\hat\mu,\ 0.001^2)$ |
| $\alpha$ | $\mathcal{N}_{[0,\infty]}(\hat\alpha,\ 0.05^2)$ |
| $\gamma$ | $\mathcal{N}(\hat\gamma,\ 0.05^2)$ |
| $\beta$ | $\text{Beta}(47, 3)$ |
| $\nu - 2.1$ | $\text{Gamma}\left(\dfrac{\hat\nu - 2.1}{2},\ 0.5\right)$ |
| $\sigma_\infty$ | $\mathcal{N}_{[0,0.02]}(\sigma_{prior},\ 0.003^2)$ |

where $\omega = (1-\beta)\log\sigma_\infty^2$ is derived by taking expectations of the log-variance recursion as $t\to\infty$.

**Why these specific distribution families:**
- $\beta \sim \text{Beta}(47,3)$: Beta's support is naturally bounded on $[0,1]$, matching the stationarity requirement $\beta<1$ — no separate truncation needed.
- $\alpha \sim \mathcal{N}_{[0,\infty]}(\hat\alpha, 0.05^2)$: truncated at zero, since $\alpha$ (the magnitude-response coefficient) must be non-negative.
- $\nu - 2.1 \sim \text{Gamma}(\cdot)$: shifted by 2.1 rather than 2, since Student-t requires $\nu>2$ for finite variance — the shift keeps the Gamma-distributed quantity strictly positive while ensuring the resulting $\nu$ always exceeds the theoretical minimum with a small safety margin.
- $\sigma_\infty \sim \mathcal{N}_{[0,0.02]}(\sigma_{prior}, 0.003^2)$: truncated to be strictly positive, since it represents a standard deviation.

### Applying the Metrics to EGARCH

The following applies to both the regression and sequential settings, unless noted otherwise below.

- **QLIKE**: applies to $\hat\sigma_t^{model}$ (aggregated per the correct-order-of-aggregation principle above) vs. $\hat\sigma_t^{empirical}$ — no additional distributional assumption needed beyond the aggregation itself.
- **HV Coverage**: unlike the baseline, EGARCH has a genuine posterior — the credible interval combines **both** the posterior's own spread (uncertainty in $\sigma_t$ given the model) **and** the log-normal noise correction ($\hat\sigma_\eta$, correcting for the empirical benchmark's own estimation noise). These are two distinct, additive sources of interval width, unlike the baseline where the noise correction alone determines the interval.
- **PIT/KS**: uses a scaled Student-t $F_t$ (matching the model's own $t_\nu$ innovation assumption), not Normal — so a PIT/KS failure for EGARCH is more diagnostically informative than for the Gaussian-based models, since EGARCH already has flexibility to absorb ordinary fat-tail behavior via $\nu$ (see PIT/KS caveat).
- **ACF diagnostics / Engle-Ng**: computed identically to the other models. Since $\gamma$ directly targets the leverage effect, EGARCH's Engle-Ng result is the direct empirical test of whether that mechanism actually works — contrast against the baseline's confirmed failure on this same test.

#### Regression Setting

$\sigma_t$ for every day is obtained directly from the posterior fit via the recursion run through the actual, fully-observed regression-period returns (see MCMC / mcmc.md) — genuinely in-sample, with no forward simulation involved, since no future uncertainty exists relative to any day within an already-observed period. $\hat\sigma_t^{model}$ is then constructed by averaging $\sigma_t^2$ across posterior draws *before* the window-RMS aggregation (see Correct Order of Aggregation), analogous to the prior predictive check but conditioning on the posterior rather than the prior.

#### Sequential Setting

Two distinct processes run in parallel each window (see Filtering vs. Smoothing, above, for the underlying distinction):

- **Forecast generation** (for QLIKE/RV coverage): forward-simulated from the incoming posterior only, using fresh simulated shocks — no real data from the current window is used. This produces $\hat\sigma_t^{model}$ for the whole window from a single block simulation, per the documented filtering-vs-smoothing limitation.
- **Filtering** (for PIT/serial-dependence): the recursion is run through the window's real, realized returns one step at a time, producing genuine day-by-day $\sigma_t$ estimates conditioned only on data up to each respective day.

Both the window's forecast noise correction ($\hat\sigma_\eta$, expanding as described for the baseline) and the posterior itself are updated only using information available up to the start of each window — never the window's own data — preserving the same out-of-sample discipline established for the baseline, applied here to a genuinely fitted model rather than a fixed formula.

### EGARCH-Specific Sequential Mechanics

Carrying the posterior forward window-to-window requires two separate things to be propagated, for different purposes:

- **Volatility state** ($\sigma_T$ at the end of the previous window): carried forward **per posterior draw**, not as a single point estimate — each draw's own end-of-window state (obtained from that same draw's own parameters and own realized path) seeds that draw's own forward simulation into the next window. This preserves genuine posterior uncertainty in the starting condition, rather than collapsing to one "average" starting point.
- **Parameter posterior**: the previous window's posterior samples are approximated by fitting them to the same parametric families used for the original priors (Beta for $\beta$, Gamma for $\nu_{shift}$, Normal/TruncatedNormal for the rest, via method-of-moments or maximum-likelihood fits to the retained samples). This approximate distribution becomes the new window's prior, and NUTS is re-run on that window's data alone to produce an updated posterior.

This is an **approximate**, not exact, form of sequential Bayesian updating: exact conjugate updating would require the prior and likelihood to combine analytically into a closed-form posterior of the same family, which does not hold for EGARCH's nonlinear recursion. Approximating the previous posterior by a parametric fit, then treating that as a fresh prior for a new NUTS run, is a standard practical substitute — it loses some fidelity relative to carrying the full previous posterior forward exactly, but avoids needing to refit the entire history from scratch at every window.

## MCMC

Posterior samples for EGARCH and SV are obtained via MCMC (specifically NUTS). See sampling.md for the full theoretical justification (why MCMC produces valid posterior samples, Markov chains, detailed balance, the Metropolis-Hastings construction, and how NUTS extends this).

### Stationarity and the COVID Period

EGARCH and SV assume $\sigma_t$ fluctuates around a stable long-run level $\sigma_\infty^2$, not that $\sigma_t$ is constant. This is compatible with volatility clustering and mean-reversion, but assumes a single, fixed $\sigma_\infty^2$ holds throughout the estimation period.

COVID poses two potential challenges to this: (1) if COVID represents a genuine, lasting shift in the market's volatility regime rather than a temporary deviation, the single $\sigma_\infty^2$ assumed (elicited from the calmer 2013–2017 prior period) may not be the correct long-run target for the regression period as a whole; (2) even if stationarity holds in principle, the models' persistence parameters ($\beta$, $\phi$) determine how quickly volatility reverts, and COVID's unusually rapid spike may exceed the speed these parameters were calibrated to handle. Both are testable via the existing COVID sub-period evaluation (QLIKE, RV coverage) rather than assumed away.

## Stochastic (SV) Model

### Stochastic Volatility Priors

| Parameter | Prior |
|---|---|
| $\mu_h$ | $\mathcal{N}(\hat\mu_h,\ 1^2)$ |
| $\phi$ | $\text{Beta}(7, 1)$ |
| $\sigma_\eta$ | $\mathcal{N}^{+}(0,\ 0.5^2)$ |
| $\mu_r$ | $\mathcal{N}(\hat\mu_r,\ 0.001^2)$ |

where $\mu_h = \log\sigma_\infty^2$ is derived by taking expectations of the latent volatility recursion as $t\to\infty$.