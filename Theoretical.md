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
- **RV Coverage**: checks if the model's stated confidence intervals actually contain the empirical value at the expected rate.

### Return Distribution Calibration
**Point**: checks if the model's implied return distribution matches the actual return distribution — a day-by-day test, independent of the window-level comparison above and of the noise correction used in RV coverage.

- **PIT/KS**: tests whether the model's day-by-day claimed uncertainty matches what actually happens (jointly tests volatility level and assumed shape — can't separate the two)
- KS test / histogram: formal and visual versions of the same check.

### Serial Dependence Diagnostics
**Point**: checks if the model's errors are predictable over time — a model can be right on average while still failing in a systematic, exploitable pattern.

- **ACF of $z_t$**: tests for directional persistence in errors.
- **ACF of $z_t^2$**: tests for volatility-clustering persistence in errors.
- **Engle-Ng sign-bias test**: tests if errors depend on the sign of the previous return (leverage effect).

These metrics are covered in more detail in the metrics.md section

### Why Regression Testing?

- Best-case scenario: parameters fit with full knowledge of the returns being evaluated.
- Diagnostic value: separates two failure modes that pure out-of-sample testing would conflate:
  - **Model misspecification** — fails even in-sample (see Bayesian posterior consistency discussion under model-specific limitations, for what happens if this occurs)
  - **Generalization failure** — works in-sample, fails out-of-sample
- These call for different fixes (respecify the model vs. improve adaptation), so separating them matters.

### Why Sequential Updating?

- Genuinely out-of-sample **provided the fitting process uses only data prior to each window** (filtering, not smoothing — see Model Estimator / forward simulation discussion for how this is verified in practice). 
- The most stringent test: can the model forecast using only current information?

(fitting process only uses data prior to each window)

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
- **RV Coverage**: no native posterior, so the interval is constructed entirely from the log-normal noise model ($\hat\sigma_\eta^{roll}$, fit the same way as for the Bayesian models, substituting $\hat\sigma_t^{roll}$ for $\hat\sigma_t^{model}$). Unlike EGARCH/PF-SV, this noise-model interval is the *sole* source of interval width — there is no posterior uncertainty to combine it with.
- **PIT/KS**: requires an externally imposed distributional shape, since the baseline gives only a point estimate. $r_t \sim N(0,(\hat\sigma_t^{roll})^2)$ is assumed — matching PF-SV's Gaussian innovation assumption, so PIT/KS results between baseline and PF-SV test the same joint hypothesis (see PIT/KS caveat).
- **ACF diagnostics / Engle-Ng**: computed identically to the other models, using $z_t=\Phi^{-1}(u_t)$ from the baseline's own PIT transform above.

#### Regression Setting

$\hat\sigma_\eta^{roll}$ is fit once, globally, over the whole regression period. Results (QLIKE, RV coverage) are additionally broken down into pre-COVID, COVID, and post-COVID sub-periods, since COVID falls within this period and a pooled average can mask sub-period miscalibration.

#### Sequential Setting

Two considerations arise that don't apply to the regression setting:

**Window-by-window breakdown, not COVID sub-periods**: the sequential period (2022–2025) contains no comparable single crisis event, so results are instead broken down into eight fixed six-month windows spanning the full period, in addition to pooled values — the same rationale as the COVID breakdown, applied as a general-purpose decomposition rather than one tied to a specific known event.

**$\hat\sigma_\eta$ must be estimated sequentially, not fit once globally**: using one global $\hat\sigma_\eta$ for the entire sequential period would violate the out-of-sample principle underlying sequential updating — the noise estimate for an early window would be silently informed by residuals from windows that haven't happened yet. $\hat\sigma_\eta$ is instead computed as an expanding, past-only estimate: the first sequential window uses $\hat\sigma_\eta$ carried forward from the regression period; each subsequent window's $\hat\sigma_\eta$ is recomputed using all residuals accumulated from the regression period plus every *completed* sequential window — never including the window currently being evaluated. This expanding-window convention was chosen (over a fixed rolling window or exponential smoothing) for being parameter-free, avoiding an arbitrary window length or decay rate.

This affects RV coverage only — $\hat\sigma_\eta$ has no role in the models' own fitting/forecasting process (EGARCH's MCMC recursion, PF-SV's particle filtering, or the baseline's point estimate); it is purely a post-hoc correction applied when checking the model's output against the empirical benchmark.

QLIKE, PIT/KS, and ACF/Engle-Ng diagnostics are otherwise unchanged from the regression setting — these operate per-day (or per-window as a simple aggregation of per-day values) and involve no noise-model fitting of their own.

## Priors

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

(Patton, 2011 — check citation key; original EGARCH: Nelson, 1991)

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

Quick explanation beta < 1 for stationarity and 0.94 was the fit for it
alpha is always positive
nu-2.1 is needed since the value of nu never goes below 2 anyway so it makes sense to take 2.1 
sigma infinity is always positive

**Bayesian implementation**: 

Priors Through Prior Predictive Checking in 2013-2017 period

Generate posterior samples via MCMC (NUTS, through NumPyro - a gradient based sampler for efficient posterior exploration). 

### Stationarity and the COVID Period

EGARCH and SV assume $\sigma_t$ fluctuates around a stable long-run level $\sigma_\infty^2$, not that $\sigma_t$ is constant. This is compatible with volatility clustering and mean-reversion, but assumes a single, fixed $\sigma_\infty^2$ holds throughout the estimation period.

COVID poses two potential challenges to this: (1) if COVID represents a genuine, lasting shift in the market's volatility regime rather than a temporary deviation, the single $\sigma_\infty^2$ assumed (elicited from the calmer 2013–2017 prior period) may not be the correct long-run target for the regression period as a whole; (2) even if stationarity holds in principle, the models' persistence parameters ($\beta$, $\phi$) determine how quickly volatility reverts, and COVID's unusually rapid spike may exceed the speed these parameters were calibrated to handle. Both are testable via the existing COVID sub-period evaluation (QLIKE, RV coverage) rather than assumed away.


### Implementation using Jax and NumPyro

Using JAX and NumPyro, the MCMC method can be implemented. 

The idea is to sample things via NumPyro and then create a gird with EGARCH steps



## Stochastic (SV) Model

### Stochastic Volatility Priors

| Parameter | Prior |
|---|---|
| $\mu_h$ | $\mathcal{N}(\hat\mu_h,\ 1^2)$ |
| $\phi$ | $\text{Beta}(7, 1)$ |
| $\sigma_\eta$ | $\mathcal{N}^{+}(0,\ 0.5^2)$ |
| $\mu_r$ | $\mathcal{N}(\hat\mu_r,\ 0.001^2)$ |

where $\mu_h = \log\sigma_\infty^2$ is derived by taking expectations of the latent volatility recursion as $t\to\infty$.