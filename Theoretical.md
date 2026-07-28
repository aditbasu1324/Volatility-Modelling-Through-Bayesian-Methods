## Theoretical Section

This file stores information on the theoretical parts of this project.

## Volatility

- Volatility = standard deviation of daily log returns.
- Unlike price, volatility isn't published by the market — it's an unobservable ("latent") quantity.
- Consequence: a latent volatility model can only be evaluated indirectly, via metrics designed for this (below).

### Why Regression Testing?

- Best-case scenario: parameters fit with full knowledge of the returns being evaluated.
- Diagnostic value: separates two failure modes that pure out-of-sample testing would conflate:
  - **Model misspecification** — fails even in-sample
  - **Generalization failure** — works in-sample, fails out-of-sample
- These call for different fixes (respecify the model vs. improve adaptation), so separating them matters.

### Why Sequential Updating?

- Genuinely out-of-sample: forecast and measurement noise for each window use only data prior to that window.
- The most stringent test: can the model forecast using only current information, without hindsight (unlike regression testing)?

## Baseline: Rolling Average

$$
    \hat{\sigma}_t^{roll} = \sqrt{\frac{1}{h-1}
    \sum_{i=0}^{h-1}(r_{t-i} - \bar{r}_t)^2}
$$

where $\bar{r}_t = \frac{1}{h}\sum_{i=0}^{h-1} r_{t-i}$, $h=21$.

- Naive baseline: no model assumptions, trivially computable.
- Not parametric → no in-sample/out-of-sample distinction; both computed immediately.
- Not Bayesian → no credible intervals.

## Historical Volatility (Benchmark)

$$
    \hat{\sigma}_t^{empirical} = \sqrt{\frac{1}{h-1}
    \sum_{i=1}^{h}(r_{t+i} - \bar{r}_t)^2}
$$

where $r_{t}=\ln\left(\frac{P_{t}}{P_{t-1}}\right)$, $\bar{r}_t = \frac{1}{h}\sum_{i=1}^{h} r_{t+i}$, $h=21$.

- Represents average volatility over the window, assuming volatility is constant within it.
- Compared against the model side via RMS:

$$
    \hat{\sigma}_t^{model} =\sqrt{\frac{1}{h-1}\sum_{i=1}^{h} \sigma_{t+i}^{2}}
$$

where $\sigma_t$ = model's instantaneous volatility estimate.

**Why does this estimator work at all?**
- LLN: sample variance → true $\sigma_t^2$ as $h \to \infty$, if returns are drawn from a fixed-variance distribution.

**Why $h=21$?**
- Standard error shrinks $\propto \frac{1}{\sqrt{h}}$.
- $h=1$: mostly idiosyncratic noise. $h=21$: noise partially averages out.
- Same logic applies to RMS-aggregating the model side over the same window.

**Why RMS for the models?**
- Empirical side: one SD over the whole window (assumes constant volatility).
- Model side: a distinct daily estimate per day → needs aggregating.
- RMS (not simple average) works because variance is additive under conditional independence: $\text{Var}\left(\sum_i r_{t+i}\right) = \sum_i \sigma_{t+i}^2$, no cross-terms.
- All models tested assume conditionally independent daily variances, so this holds.

**Sources of error** (even for a correctly specified model):
1. **Systematic estimation error** — $\hat\sigma_t^{empirical}$ is a finite-sample estimator (21 returns), carrying its own sampling variance regardless of model quality.
2. **Genuine model error** — actual misspecification/approximation error.

- Fair comparison requires factoring #1 into the model side too — via a log-normal noise model (strictly positive, analytically tractable).

## QLIKE Loss

$$
    \text{QLIKE}(\hat{\sigma}_t, RV_t) = \frac{RV_t^2}{\hat{\sigma}_t^2} 
    - \log\!\left(\frac{RV_t^2}{\hat{\sigma}_t^2}\right) - 1
$$

- Compares pointwise model volatility estimates against realized/historical volatility.
- Lower mean QLIKE = better fit.

**Why QLIKE?**

- **Why QLIKE?**

- **Asymmetric penalty for underprediction**: for *any* proportional miss, underpredicting volatility by a factor $k$ incurs strictly more loss than overpredicting by the same factor $k$ (verified directly from the loss function: $\text{Loss}_{\text{under}}(k) - \text{Loss}_{\text{over}}(k) = k^2 - 1/k^2 - 4\log k > 0$ for all $k>1$).
  - Underestimating volatility is generally more costly in practice (risk management, option pricing) — this asymmetry reflects that.
- **Robust to a noisy proxy**: true volatility is unobservable, so evaluation must use a noisy proxy ($RV_t$) instead. QLIKE remains valid under this — model rankings match what they'd be under the true (unobservable) volatility (Patton, 2011).
  - This is the main reason QLIKE (not MSE) is standard here: it directly addresses the systematic estimation error in $RV_t$ (see Sources of Error, above) rather than being distorted by it.

## Realized Volatility Coverage

$$\hat\sigma_t^{empirical} = \sigma_t^{true} \cdot \exp(\eta_t), \quad \eta_t \sim N(0,\sigma_\eta^2)$$

$$\hat\sigma_\eta = \text{std}\left(\log \hat\sigma_t^{empirical} - \log \hat\sigma_t^{model}\right)$$

- Coverage = % of $\hat\sigma_t^{empirical}$ values falling within the model's credible interval for $\hat\sigma_t^{model}$, widened using $\hat\sigma_\eta$.

**Why is this adjustment needed, even for EGARCH/SV's own credible intervals?**

- A model's credible interval reflects its posterior belief about the *true* latent $\sigma_t^{true}$ — it says nothing about the noise in $\hat\sigma_t^{empirical}$ itself, since that's a property of the benchmark's construction (finite 21-return sample), not of the model.
- Without adjustment, checking coverage directly against the model's raw interval implicitly assumes $\hat\sigma_t^{empirical} = \sigma_t^{true}$ exactly — which isn't true, per Sources of Error above.
- This would show artificially poor coverage (too many "misses") purely due to the benchmark's own noise, misattributing it to model miscalibration — a generic issue independent of which model is used.
- Widening the interval by the estimated $\hat\sigma_\eta$ correctly separates "is the model's belief about the truth reasonable" from "is my benchmark itself noisy" — isolating the thing actually being tested (model calibration).

**Is this circular?**
- Regression: $\hat\sigma_\eta$ is fit once, globally over the whole regression period — not tuned per-window to that window's outcome, so it can't trivially inflate coverage to look artificially good.
- Sequential: no ambiguity — $\hat\sigma_\eta$ uses only data prior to each window, genuinely out-of-sample.

**Why log-normal noise specifically?**
- **Positivity**: multiplicative noise ($\times\exp(\eta_t)$) keeps $\hat\sigma_t^{empirical}$ strictly positive, unlike additive Gaussian noise which could push it negative.
- **Empirical regularity**: log-volatility is commonly observed to be approximately normally distributed, even when volatility itself is skewed — motivating the log transform specifically, not an arbitrary choice.
- **Tractability**: reduces to a single scalar noise parameter ($\sigma_\eta$), directly estimable and easy to combine with the model's own posterior on the log scale.
- More flexible alternatives (e.g. gamma or non-central chi-squared noise models) exist but add complexity without strong evidence they're needed here — log-normal is the simplest choice consistent with both positivity and the known empirical shape of volatility noise.

Maybe add this justification near the start

## Return Calibration Test (PIT / KS)

$$u_t = F_t(r_t)$$

where $F_t$ is the model's estimated CDF of returns at time $t$ (derived from $\hat\sigma_t^{model}$, e.g. $r_t \sim N(0, (\hat\sigma_t^{model})^2)$), and $r_t$ is the actual observed return.

- $u_t$ = the fraction of the model's estimated return distribution lying below the actual return at time $t$.

### Why Does This Work? (The PIT Theorem)

If $r_t$ is truly drawn from the distribution $F_t$ that the model assumes, then the transformed value $u_t = F_t(r_t)$ is **uniformly distributed on $[0,1]$** — this is a general probability result (the Probability Integral Transform), true for *any* continuous distribution, not specific to returns or volatility.

Intuition: $F_t$ maps outcomes to their own percentile under the model. If the model's distributional assumption is exactly correct, then by definition, the actual return is equally likely to fall at any percentile of that distribution — landing in the bottom 10% happens 10% of the time, the top 5% happens 5% of the time, and so on. This is what "well-calibrated" means: the model's stated uncertainty matches reality.

This gives a testable implication: compute $u_t$ across all $t$, and check whether the resulting values look uniform. If the model is miscalibrated (e.g. consistently too narrow, too wide, or systematically biased), the empirical distribution of $u_t$ will deviate from uniform in a detectable way — e.g., too many $u_t$ near 0 and 1 means the model's intervals are too narrow (true returns fall in the tails more often than the model expects).

### Testing Uniformity: Kolmogorov-Smirnov

Uniformity of $\{u_t\}$ is tested via the KS test (Massey, 1951), which compares the empirical CDF of $\{u_t\}$ against the CDF of a true Uniform(0,1) distribution. $p < 0.05$ indicates the model is not well calibrated.

### Return Coverage

Analogous to RV coverage: the percentage of actual returns falling within the model's credible interval (for various interval widths), directly from the $u_t$ values (e.g. a 90% interval is violated if $u_t < 0.05$ or $u_t > 0.95$).

### Why This Test, Given RV Coverage Already Exists?

- **Independent of the noise correction** used for RV coverage ($\hat\sigma_\eta$) — this test operates purely on daily returns and the model's own distributional assumption, with no separate proxy-noise model needed.
- **Different sensitivity**: RV coverage checks calibration of the volatility *magnitude* over a 21-day window; PIT/KS checks calibration of the full *return distribution*, day by day. A model could get the window-average magnitude right (passing RV coverage) while still being miscalibrated day-to-day (failing PIT/KS) — or vice versa.
- Because it says nothing directly about whether the volatility *level* itself is systematically too high or low (only whether the *shape* of the day-to-day distribution is well-calibrated), this is treated as a supporting metric rather than a primary one.

**A caveat: PIT/KS jointly tests two things, and the assumed shape differs by model**

Computing $u_t = F_t(r_t)$ requires committing to a specific shape for $F_t$:
- **Rolling average, PF-SV**: $r_t \sim N(0, (\hat\sigma_t^{model})^2)$
- **EGARCH(1,1)-t**: $r_t \sim t_\nu(0, (\hat\sigma_t^{model})^2)$ (scaled Student-t, with $\nu$ fit as part of the model)

The PIT theorem itself is distribution-free, but *this application* of it jointly tests whether (a) $\hat\sigma_t^{model}$ is the correct volatility level, and (b) the assumed innovation shape is correct. A PIT/KS failure doesn't distinguish between these.

Because EGARCH already assumes fatter tails (Student-t) than the Gaussian-based models, a PIT/KS failure for EGARCH is more diagnostically informative — it suggests the volatility level itself may be miscalibrated, since the model already has flexibility to absorb ordinary fat-tail behavior via its $t_\nu$ shape. A Gaussian model's PIT/KS failure is less conclusive on its own, since it could simply reflect the well-known fact that returns are fatter-tailed than Gaussian, independent of whether the volatility level is right.

### EGARCH

The EGARCH(1,1)-t model \citep{08a920ce-a812-35fd-8f86-d1bf6accbe7f} is parameterized by:

\begin{align}
r_t &= \mu + \epsilon_t, \quad \epsilon_t = \sigma_t z_t, \quad z_t \overset{\text{i.i.d.}}{\sim} t_\nu(0,1) \\
\ln\sigma_t^2 &= \omega + \alpha\left(|z_{t-1}| - \mathbb{E}|z_{t-1}|\right) + \gamma z_{t-1} + \beta\ln\sigma_{t-1}^2
\end{align}

Its key differentiator over other time series models is its ability to capture the leverage effect (the principle that volatility reacts asymmetrically to equal changes in positive and negative returns) through $\gamma$.

It also captures other key properties of volatility (the t-distribution captures fat-tailedness of returns and $\alpha$ captures the effect of sudden large changes in price).