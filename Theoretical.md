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

## Historical Volatility (Benchmark)

### The Goal

Under the assumption that volatility is constant within the window, conditional on $\mathcal{F}_t$ ($\sigma_{t+i} = \sigma$ for all $i=1,...,h$, given information up to time $t$), the target quantity is:
$$\text{Var}(r_{t+1}, ..., r_{t+h} \mid \mathcal{F}_t) = \sigma^2$$

Two different estimators of this same quantity are constructed — one from the raw returns directly (empirical side), one from the model's own outputs (model side) — and compared.

### Empirical Estimator (via LLN)

$$
\hat{\sigma}_t^{empirical} = \sqrt{\frac{1}{h-1}\sum_{i=1}^{h}(r_{t+i} - \bar{r}_t)^2}
$$

where $r_{t}=\ln\left(\frac{P_{t}}{P_{t-1}}\right)$, $\bar{r}_t = \frac{1}{h}\sum_{i=1}^{h} r_{t+i}$, $h=21$.

Conditional on $\mathcal{F}_t$, each $r_{t+i}$ is treated as an independent draw from a common distribution with variance $\sigma^2$ (and mean $\mu$).

By the Law of Large Numbers, the sample variance converges to $\sigma^2$ as $h\to\infty$. The $h-1$ denominator (Bessel's correction) corrects for the fact that the mean $\mu$ is itself estimated by $\bar r_t$ from the same $h$ observations.

### Model Estimator (via Martingale-Difference Sum)

All the models specifiy $r_{t+i} = \mu + \sigma_{t+i}\varepsilon_{t+i}$, with $\varepsilon_{t+i}$ iid mean-zero, and $\sigma_{t+i}=\text{Var}(r_{t+i}\mid \mathcal{F}_{t+i-1})$. Only $\sigma_{t+1}$ is $\mathcal{F}_t$-measurable (known at time $t$); for $i>1$, $\sigma_{t+i}$ is itself a random variable from the perspective of $\mathcal{F}_t$, since it depends on interim returns not yet realized at time $t$.

**Goal**: derive $\text{Var}\left(\sum_{i=1}^h r_{t+i}\,\middle|\,\mathcal{F}_t\right)$.

**Cross-terms vanish**

For $i<j$, conditioning throughout on $\mathcal{F}_t$ and using the tower property:
$$E[r_{t+i}\,r_{t+j}\mid\mathcal{F}_t] = E\Big[r_{t+i}\cdot E[r_{t+j}\mid\mathcal{F}_{t+j-1}]\,\Big|\,\mathcal{F}_t\Big] = E[r_{t+i}\cdot\mu\mid\mathcal{F}_t] = \mu\,E[r_{t+i}\mid\mathcal{F}_t]$$
Since $E[r_{t+i}\mid\mathcal{F}_t]\cdot E[r_{t+j}\mid\mathcal{F}_t] = E[r_{t+i}\mid\mathcal{F}_t]\cdot\mu$ as well (as $E[r_{t+j}\mid\mathcal{F}_t]=\mu$ by tower), the covariance is exactly zero:
$$\text{Cov}(r_{t+i},r_{t+j}\mid\mathcal{F}_t)=0$$

**Diagonal terms**

By the law of total variance, conditioning on the intermediate filtration $\mathcal{F}_{t+i-1}$:
$$\text{Var}(r_{t+i}\mid\mathcal{F}_t) = \underbrace{E\big[\text{Var}(r_{t+i}\mid\mathcal{F}_{t+i-1})\mid\mathcal{F}_t\big]}_{E[\sigma_{t+i}^2\mid\mathcal{F}_t]} + \underbrace{\text{Var}\big(E[r_{t+i}\mid\mathcal{F}_{t+i-1}]\mid\mathcal{F}_t\big)}_{\text{Var}(\mu\mid\mathcal{F}_t)\,=\,0}$$

So:
$$\text{Var}(r_{t+i}\mid\mathcal{F}_t) = E[\sigma_{t+i}^2\mid\mathcal{F}_t]$$

**Putting it together**

$$\text{Var}\left(\sum_{i=1}^h r_{t+i}\,\middle|\,\mathcal{F}_t\right) = \sum_{i=1}^h E[\sigma_{t+i}^2\mid\mathcal{F}_t]$$

### Applying Constant-Volatility Assumption 

If volatility is assumed constant across the window ($\sigma_{t+i}=\sigma$ for all $i$, given $\mathcal{F}_t$), then trivially:
$$E[\sigma_{t+i}^2\mid\mathcal{F}_t] = E[\sigma^2\mid\mathcal{F}_t] = \sigma^2$$

Substituting into the general result above:
$$\text{Var}\left(\sum_{i=1}^h r_{t+i}\,\middle|\,\mathcal{F}_t\right) = \sum_{i=1}^h \sigma^2 = h\sigma^2$$

Rearranging:
$$\sigma^2 = \frac{1}{h}\sum_{i=1}^h E[\sigma_{t+i}^2\mid\mathcal{F}_t]$$ (make this the expectation formula instead)

Note that $\sigma_{t+i}$ is different from $E[\sigma_{t+i}^2\mid\mathcal{F}_t]$ which will be addressed during the implementation

### Sources of Error

Even under a correctly specified model, $\hat\sigma_t^{empirical}$ and $\hat\sigma_t^{model}$ are not expected to match exactly:

1. **Systematic estimation error** — $\hat\sigma_t^{empirical}$ is a finite-sample estimator (LLN gives convergence only as $h\to\infty$; at $h=21$ it carries residual sampling variance regardless of model correctness).
2. **Genuine model error** — the model's own misspecification or approximation error.

A fair comparison requires factoring source 1 into the model side too, via a log-normal noise model.

**Note**
Patton's original paper compared metrics through daily forecasts scored against daily proxy, whereas this project applies the QLIKE to the window-aggregated quantities on both side. This extension isn't explicity proven but makes sense through the linearity of expectation and is widely used.

## Historical Volatility Coverage: Log-Normal Measurement Error

(needs to be checked for later)
### The Noise Model

$$\hat\sigma_t^{empirical} = \sigma_t^{true} \cdot \exp(\eta_t), \quad \eta_t \sim N(0,\sigma_\eta^2)$$

$$\hat\sigma_\eta = \text{std}\left(\log\hat\sigma_t^{empirical} - \log\hat\sigma_t^{model}\right)$$

### Why Log-Normal Noise?

- **Empirically valid**: log-historically-volatility is widely documented to be approximately normally distributed through empirical testing across several studies (paper's talking about the same "Volatility Is Log-Normal, But Not for the Reason You Think")
- **Positivity**: multiplicative noise ($\times\exp(\eta_t)$) keeps $\hat\sigma_t^{empirical}$ strictly positive, unlike additive Gaussian noise which could push it negative.
- **Tractability**: reduces to a single scalar parameter ($\sigma_\eta$), directly estimable and easy to combine with the model's own posterior on the log scale.
- This is a **pragmatic, standard convention**, not a theoretically mandated choice — more flexible alternatives (gamma, non-central chi-squared) exist but add complexity without strong evidence they're needed here.

### What $\hat\sigma_\eta$ Is Meant to Represent

$\eta_t$ is meant to isolate the estimation noise in $\hat\sigma_t^{empirical}$ alone — how far the empirical historical-volatility estimate deviates from the true (unobservable) $\sigma_t^{true}$, purely due to using a finite 21-day sample (see Sources of Error, historical volatility section above).

### Why the Formula Uses $\hat\sigma_t^{model}$ Instead of $\sigma_t^{true}$

$\sigma_t^{true}$ is never observable, so there is no way to compute $\eta_t$ directly. $\hat\sigma_t^{model}$ is substituted in as the best available stand-in — this is a forced practical necessity, not an independently justified choice.

### What $\hat\sigma_\eta$ Actually Estimates (Given This Substitution)

Writing $\hat\sigma_t^{model} = \sigma_t^{true}\cdot\exp(\epsilon_t^{model})$ for the model's own (not necessarily Gaussian) error:

$$\log\hat\sigma_t^{empirical} - \log\hat\sigma_t^{model} = \eta_t - \epsilon_t^{model}$$

($\log\sigma_t^{true}$ cancels exactly.) Assuming $\eta_t$ and $\epsilon_t^{model}$ are independent:

$$\hat\sigma_\eta^2 \approx \sigma_\eta^2 + \text{Var}(\epsilon_t^{model})$$

So the computed $\hat\sigma_\eta$ is an estimate of the **combined** spread of benchmark noise and model error — not benchmark noise alone. There is no way, from this single computable quantity, to separate the two contributions.

### Sources of Error This Is (and Isn't) Correcting For

- **Corrects for**: the systematic estimation error inherent in $\hat\sigma_t^{empirical}$ (Sources of Error, point 1) — widening the model's credible interval by $\hat\sigma_\eta$ prevents this benchmark noise from being mistaken for model miscalibration.
- **Does not cleanly separate from**: genuine model error (Sources of Error, point 2) — since $\hat\sigma_\eta$ is contaminated by $\text{Var}(\epsilon_t^{model})$, the correction is only accurate to the extent that model error is small relative to true benchmark noise.
- **Assumes**: $\eta_t \perp \epsilon_t^{model}$ (independence) — plausible but unverified; if both errors grow together during volatile periods (a reasonable concern), this assumption breaks down and $\hat\sigma_\eta$ may over- or under-estimate the combined variance depending on their correlation's sign.

(Probably a way to check that assumption)

### Does Adding more Data help?

Yes — $\sigma_\eta$ reflects how imprecise $\hat\sigma_t^{empirical}$ is, which is a direct consequence of the 21-day historical volatility construction relying on LLN over only 21 daily observations. A benchmark built from intraday data (day-level RV via quadratic variation, aggregating many high-frequency returns per day — see terminology note above) would be substantially more precise, shrinking $\sigma_\eta$ and making a larger share of $\hat\sigma_\eta$ attributable to genuine model error rather than benchmark imprecision — sharpening the coverage test's diagnostic power. This is a concrete motivation for the intraday data extension discussed in extensions.md.

### Is This Circular?

- **Regression setting**: $\hat\sigma_\eta$ is fit once, globally, over the whole regression period — not tuned per-window to that window's own outcome, so coverage can't be trivially gamed this way.
- **Sequential setting**: no ambiguity — $\hat\sigma_\eta$ is computed only from data prior to each window being evaluated, genuinely out-of-sample.

## QLIKE Loss

$$
    \text{QLIKE}(\hat{\sigma}_t, RV_t) = \frac{RV_t^2}{\hat{\sigma}_t^2} 
    - \log\!\left(\frac{RV_t^2}{\hat{\sigma}_t^2}\right) - 1
$$

- Compares pointwise model volatility estimates against historical volatility.
- Lower mean QLIKE = better fit.

**Why QLIKE?**

- **Asymmetric penalty for underprediction**: for *any* proportional miss, underpredicting volatility by a factor $k$ incurs strictly more loss than overpredicting by the same factor $k$ (verified directly from the loss function: $\text{Loss}_{\text{under}}(k) - \text{Loss}_{\text{over}}(k) = k^2 - 1/k^2 - 4\log k > 0$ for all $k>1$).
  - Underestimating volatility is generally more costly in practice (risk management, option pricing) — this asymmetry reflects that.
- **Robust to a noisy proxy**: true volatility is unobservable, so evaluation must use a noisy proxy ($RV_t$) instead. QLIKE remains valid under this — model rankings match what they'd be under the true (unobservable) volatility (Patton, 2011).
  - This is the main reason QLIKE (not MSE) is standard here: it directly addresses the systematic estimation error in $RV_t$ (see Sources of Error, above) rather than being distorted by it.

## Realized Volatility Coverage

$$\hat\sigma_t^{empirical} = \sigma_t^{true} \cdot \exp(\eta_t), \quad \eta_t \sim N(0,\sigma_\eta^2)$$

$$\hat\sigma_\eta = \text{std}\left(\log \hat\sigma_t^{empirical} - \log \hat\sigma_t^{model}\right)$$

- Coverage = % of $\hat\sigma_t^{empirical}$ values falling within the model's credible interval for $\hat\sigma_t^{model}$, widened using $\hat\sigma_\eta$.

## Return Calibration Test (PIT / KS)

$$u_t = F_t(r_t)$$

where $F_t$ is the model's estimated CDF of returns at time $t$ (derived from $\hat\sigma_t^{model}$), and $r_t$ is the actual observed return. $u_t$ = the fraction of the model's estimated return distribution lying below the actual return.

**Why does this work? (PIT theorem)**
- If $r_t$ truly follows $F_t$, then $u_t=F_t(r_t) \sim \text{Uniform}(0,1)$ — a general, distribution-free probability result.
- Intuition: $F_t$ maps outcomes to their own percentile. If the model is exactly correct, the actual return is equally likely to land at any percentile — 10% of returns fall in the bottom decile, 5% in the top 5%, etc. This is what "well-calibrated" means.
- Testable implication: compute $u_t$ across all $t$, check whether they look uniform. Too many $u_t$ near 0/1 → intervals too narrow (miscalibrated, underestimating tail risk).

**Testing uniformity: KS test**
- Compares empirical CDF of $\{u_t\}$ against Uniform(0,1) (Massey, 1951). $p<0.05$ → not well calibrated.

## Serial Dependence Diagnostics

KS checks whether the *pooled* $\{u_t\}$ values collectively resemble a Uniform(0,1) sample — but says nothing about whether consecutive $u_t$ are independent of each other. A model could have $u_t$ clustered near 1 during genuine high-volatility stretches (underestimating $\sigma_t$) and clustered near 0.5 during calm stretches (overestimating $\sigma_t$) — a real, time-varying miscalibration pattern — while these regimes balance out such that the pooled distribution still looks uniform overall.

To catch this, $u_t$ is transformed via $z_t = \Phi^{-1}(u_t)$, which should be iid $N(0,1)$ under correct calibration — this puts the calibration errors on a scale where standard time-series diagnostics apply directly. Three tests are run on $z_t$ to check whether the model has correctly captured volatility clustering, volatility persistence, and the leverage effect.

### ACF of $z_t$ (Ljung-Box) — directional persistence

The sample autocorrelation of $z_t$ at lag $j$ is:
$$\hat\rho_j = \frac{\sum_{t=j+1}^{n}(z_t-\bar z)(z_{t-j}-\bar z)}{\sum_{t=1}^n(z_t-\bar z)^2}$$

The Ljung-Box statistic jointly tests whether $\rho_1,...,\rho_k$ are all zero:
$$Q(k) = n(n+2)\sum_{j=1}^{k}\frac{\hat\rho_j^2}{n-j} \ \sim\ \chi^2_k \text{ under the null}$$

**Null hypothesis**: $\rho_1=...=\rho_k=0$ (no directional serial dependence in calibration errors). A **high p-value** means the null is not rejected — consecutive $z_t$'s look independent, i.e. no evidence of directional persistence (errors don't systematically "run" in the same direction from one day to the next). A **low p-value** indicates the model's errors are directionally predictable — e.g. a day of underestimation tends to be followed by more underestimation.

### ACF of $z_t^2$ (Ljung-Box, ARCH-LM style) — volatility clustering

Identical test, applied to $z_t^2$ instead of $z_t$ — squaring removes sign, so this tests whether the *magnitude* of calibration errors clusters over time, regardless of direction.

**Null hypothesis**: no autocorrelation in $z_t^2$ up to lag $k$. A **high p-value** indicates the model has successfully captured the true volatility clustering present in the data — its errors show no leftover clustering, meaning the model adapts to regime changes fast enough that its mistakes don't bunch up. A **low p-value** indicates leftover clustering: large errors follow large errors, meaning the model isn't adapting quickly enough to genuine shifts in volatility (a persistence failure).

### Engle-Ng Sign-Bias Test — leverage effect

Regress squared calibration error on the sign of the previous day's return:
$$z_t^2 = \beta_0 + \beta_1 \cdot \text{sign}(r_{t-1}) + \epsilon_t$$

Since $\text{sign}(r_{t-1})\in\{-1,+1\}$: predicted squared error after a down day is $\beta_0-\beta_1$; after an up day, $\beta_0+\beta_1$.

**Null hypothesis**: $\beta_1=0$ — squared calibration error does not depend on the direction of the prior return. A **significant, nonzero $\beta_1$** indicates the model's errors are asymmetric with respect to return direction — most commonly, larger errors following negative returns ($\beta_1<0$), consistent with the model failing to capture the leverage effect (negative returns should raise subsequent volatility more than positive returns of equal size; a model blind to this shows larger miscalibration specifically after down days).

(Extended versions of this test — negative size bias and positive size bias, using $r_{t-1}\cdot\mathbb{1}[r_{t-1}<0]$ and $r_{t-1}\cdot\mathbb{1}[r_{t-1}>0]$ as additional regressors — test whether the *magnitude*, not just the sign, of the prior return matters.)

A significant result on any of these three tests indicates systematic, time-varying miscalibration that the pooled KS test cannot detect.

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

- **QLIKE**: applies directly to $\hat\sigma_t^{roll}$ vs $\hat\sigma_t^{empirical}$ — no distributional assumption needed.
- **RV Coverage**: no native posterior, so the interval is constructed entirely from the log-normal noise model ($\hat\sigma_\eta^{roll}$, fit the same way as for the Bayesian models, substituting $\hat\sigma_t^{roll}$ for $\hat\sigma_t^{model}$). Unlike EGARCH/PF-SV, this noise-model interval is the *sole* source of interval width — there is no posterior uncertainty to combine it with.
- **PIT/KS**: requires an externally imposed distributional shape, since the baseline gives only a point estimate. $r_t \sim N(0,(\hat\sigma_t^{roll})^2)$ is assumed — matching PF-SV's Gaussian innovation assumption, so PIT/KS results between baseline and PF-SV test the same joint hypothesis (see PIT/KS caveat).
- **ACF diagnostics / Engle-Ng**: computed identically to the other models, using $z_t=\Phi^{-1}(u_t)$ from the baseline's own PIT transform above.


