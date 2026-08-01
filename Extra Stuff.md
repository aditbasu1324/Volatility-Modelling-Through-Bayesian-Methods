Extra Stuff

LLN argument and the quadratic variation argument

Key idea is mesh size of quadratic variation.
This concept doesn't hold in LLN since the day's stuff only matters i.e can't decrease mesh size





### A Note on Terminology

In the literature (e.g. Andersen-Bollerslev-Diebold-Labys), "realized volatility" typically refers to aggregating squared *intraday* (high-frequency) returns to estimate a single day's variance, with consistency guaranteed via quadratic variation as sampling frequency increases — regardless of how volatility moves within the period. The estimator used here — the sample standard deviation of 21 *daily* returns — is a related but distinct construction, more accurately termed **historical volatility**. Unlike the classical high-frequency estimator, this construction's validity depends on assuming volatility is constant across the window (see below), directly in tension with the time-varying volatility assumed by the EGARCH and PF-SV models being evaluated. This is a standard simplification in practice, but worth naming explicitly rather than treating this benchmark as unambiguous ground truth.


Need to add this later
## Historical Volatility (Benchmark)

### The Goal

Under the assumption that volatility is constant within the window ($\sigma_{t+i} = \sigma$ for all $i=1,...,h$), the target quantity is:
$$\text{Var}(r_{t+1}, ..., r_{t+h}) = \sigma^2$$

Two different estimators of this same quantity are constructed — one from the raw returns directly (empirical side), one from the model's own outputs (model side) — and compared.

### Empirical Estimator (via LLN)

$$
\hat{\sigma}_t^{empirical} = \sqrt{\frac{1}{h-1}\sum_{i=1}^{h}(r_{t+i} - \bar{r}_t)^2}
$$

where $r_{t}=\ln\left(\frac{P_{t}}{P_{t-1}}\right)$, $\bar{r}_t = \frac{1}{h}\sum_{i=1}^{h} r_{t+i}$, $h=21$.

This treats each $r_{t+i}$ as an independent draw from a common distribution with variance $\sigma^2$ (and mean $\mu$, generally nonzero for returns). By the Law of Large Numbers, the sample variance converges to $\sigma^2$ as $h\to\infty$. The $h-1$ denominator (Bessel's correction) corrects for the fact that the mean $\mu$ is itself estimated by $\bar r_t$ from the same $h$ observations, rather than being known exactly.

### Model Estimator (via Martingale-Difference Sum)

$$
\hat{\sigma}_t^{model} =\sqrt{\frac{1}{h}\sum_{i=1}^{h} \sigma_{t+i}^{2}}
$$

where $\sigma_{t+i}$ is the model's daily instantaneous volatility estimate.

This is derived differently: the model specifies $r_{t+i} = \mu + \sigma_{t+i}\varepsilon_{t+i}$, with $\varepsilon_{t+i}$ iid mean-zero (constant $\mu$ doesn't affect the argument, since it cancels out identically to how $\bar r_t$ centers the empirical side — the proof below goes through unchanged whether $\mu=0$ or constant). $\sigma_{t+i}$ is determined by information up to $t+i-1$, so returns form a martingale difference sequence around $\mu$, with conditional variance $\sigma_{t+i}^2$.

Expanding:
$$\text{Var}\left(\sum_{i=1}^h r_{t+i}\right) = \sum_i \text{Var}(r_{t+i}) + 2\sum_{i<j}\text{Cov}(r_{t+i}, r_{t+j})$$

For $i<j$, conditioning on information up to $t+j-1$ (tower property) and using that $r_{t+i}$ is already known by then:
$$\text{Cov}(r_{t+i}, r_{t+j}) = E\big[(r_{t+i}-\mu)(r_{t+j}-\mu)\big] = E\Big[(r_{t+i}-\mu)\cdot E[(r_{t+j}-\mu)\mid \mathcal{F}_{t+j-1}]\Big] = E\big[(r_{t+i}-\mu)\cdot 0\big] = 0$$

since $E[r_{t+j}-\mu\mid\mathcal{F}_{t+j-1}] = \sigma_{t+j}\cdot E[\varepsilon_{t+j}] = 0$ regardless of $\mu$. So all cross-terms vanish, leaving:
$$\text{Var}\left(\sum_i r_{t+i}\right) = \sum_i \sigma_{t+i}^2 \quad\Rightarrow\quad \sigma^2 = \frac{1}{h}\sum_i\sigma_{t+i}^2$$

No Bessel correction is needed here: unlike the empirical side, nothing is being estimated from the sample that requires bias correction — $\sigma_{t+i}$ is the model's given (or posterior-sampled) output, not a quantity computed by first estimating a nuisance parameter from the same data.

### Do the Two Estimators Agree?

Both target the same $\sigma^2$ under the constant-volatility assumption, but via genuinely different statistical arguments: the empirical side relies on the LLN across individual sample observations (needs $h\to\infty$ for consistency, has finite-sample bias corrected via $h-1$); the model side relies on martingale-difference cancellation of cross-terms in a sum (exact for any $h$, given the model specification is correct, no bias correction needed). They estimate the same target, but they are not the same estimator, and their differing denominators ($h-1$ vs. $h$) reflect this: it is not a stylistic inconsistency, but a direct consequence of the different derivations underlying each.

### Sources of Error

Even under a correctly specified model, $\hat\sigma_t^{empirical}$ and $\hat\sigma_t^{model}$ are not expected to match exactly:

1. **Systematic estimation error** — $\hat\sigma_t^{empirical}$ is a finite-sample estimator (LLN gives convergence only as $h\to\infty$; at $h=21$ it carries residual sampling variance regardless of model correctness).
2. **Genuine model error** — the model's own misspecification or approximation error.

A fair comparison requires factoring source 1 into the model side too, via a log-normal noise model (chosen for strict positivity and empirical log-normality of volatility, while remaining analytically tractable).


**Extension beyond Patton's original setting**

Patton's robustness result and empirical demonstration operate at daily granularity — a daily forecast scored against a daily proxy. This project instead applies QLIKE to window-aggregated (21-day RMS) quantities on both sides. This extension is plausible (conditional unbiasedness of the daily proxy should propagate to the window-averaged proxy, since expectation is linear), but it is an assumption carried over from the daily-level theory, not something explicitly proven for the aggregated case in the cited paper.

## Realized Volatility Coverage: Log-Normal Measurement Error

### The Noise Model

$$\hat\sigma_t^{empirical} = \sigma_t^{true} \cdot \exp(\eta_t), \quad \eta_t \sim N(0,\sigma_\eta^2)$$

$$\hat\sigma_\eta = \text{std}\left(\log\hat\sigma_t^{empirical} - \log\hat\sigma_t^{model}\right)$$

### Why Log-Normal Noise?

- **Positivity**: multiplicative noise ($\times\exp(\eta_t)$) keeps $\hat\sigma_t^{empirical}$ strictly positive, unlike additive Gaussian noise which could push it negative.
- **Empirical regularity**: log-realized-volatility is widely documented to be approximately normally distributed, even though volatility itself is skewed (Andersen-Bollerslev-style realized volatility literature; the mechanism behind this regularity is itself debated — see e.g. "Volatility Is Log-Normal, But Not for the Reason You Think" — but the empirical pattern itself is well established across many studies).
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

### Does This Noise Shrink With Better Data?

Yes — $\sigma_\eta$ reflects how imprecise $\hat\sigma_t^{empirical}$ is, which is a direct consequence of the 21-day historical volatility construction relying on LLN over only 21 daily observations. A benchmark built from intraday data (day-level RV via quadratic variation, aggregating many high-frequency returns per day — see terminology note above) would be substantially more precise, shrinking $\sigma_\eta$ and making a larger share of $\hat\sigma_\eta$ attributable to genuine model error rather than benchmark imprecision — sharpening the coverage test's diagnostic power. This is a concrete motivation for the intraday data extension discussed in extensions.md.

### Is This Circular?

- **Regression setting**: $\hat\sigma_\eta$ is fit once, globally, over the whole regression period — not tuned per-window to that window's own outcome, so coverage can't be trivially gamed this way.
- **Sequential setting**: no ambiguity — $\hat\sigma_\eta$ is computed only from data prior to each window being evaluated, genuinely out-of-sample.



## Return Calibration Test (PIT / KS)

$$u_t = F_t(r_t)$$

where $F_t$ is the model's estimated CDF of returns at time $t$ (derived from $\hat\sigma_t^{model}$), and $r_t$ is the actual observed return. $u_t$ = the fraction of the model's estimated return distribution lying below the actual return.

- **Rolling average, PF-SV**: $r_t \sim N(0, (\hat\sigma_t^{model})^2)$
- **EGARCH(1,1)-t**: $r_t \sim t_\nu(0, (\hat\sigma_t^{model})^2)$ (scaled Student-t, $\nu$ fit as part of the model)

**Why does this work? (PIT theorem)**
- If $r_t$ truly follows $F_t$, then $u_t=F_t(r_t) \sim \text{Uniform}(0,1)$ — a general, distribution-free probability result.
- Intuition: $F_t$ maps outcomes to their own percentile. If the model is exactly correct, the actual return is equally likely to land at any percentile — 10% of returns fall in the bottom decile, 5% in the top 5%, etc. This is what "well-calibrated" means.
- Testable implication: compute $u_t$ across all $t$, check whether they look uniform. Too many $u_t$ near 0/1 → intervals too narrow (miscalibrated, underestimating tail risk).

**Testing uniformity: KS test**
- Compares empirical CDF of $\{u_t\}$ against Uniform(0,1) (Massey, 1951). $p<0.05$ → not well calibrated.

**Return coverage**
- Analogous to RV coverage: % of returns within the model's credible interval, read directly off $u_t$ (e.g. 90% interval violated if $u_t<0.05$ or $u_t>0.95$).

**Why this test, given RV coverage already exists?**
- Independent of the RV-side noise correction ($\hat\sigma_\eta$) — operates purely on daily returns and the model's own distributional assumption, no separate proxy-noise model needed.
- Different sensitivity: RV coverage checks window-level magnitude; PIT/KS checks day-by-day distributional calibration. A model could pass one and fail the other.
- Doesn't directly test volatility *level* miscalibration in isolation (only distributional shape/timing) → supporting metric, not primary.

**Caveat: PIT/KS jointly tests two things**
- Computing $u_t$ requires committing to a specific shape for $F_t$ — this is a separate assumption from whether $\hat\sigma_t^{model}$ itself is correct.
- A PIT/KS failure can't distinguish: (a) wrong volatility level, vs. (b) wrong innovation shape. Both produce the same symptom (non-uniform $u_t$).
- Because EGARCH already assumes fatter tails ($t_\nu$), an EGARCH PIT/KS failure is more diagnostically informative — the model already has flexibility to absorb ordinary fat-tail behavior, so a failure more likely points to the volatility level itself. A Gaussian model's (rolling average, PF-SV) failure is less conclusive — it could simply reflect the well-known fact that returns are fatter-tailed than Gaussian, regardless of whether the volatility level is right.
- ⚠ **To confirm**: verify PF-SV's actual innovation assumption in the implementation (Gaussian assumed above, matching the rolling average's imposed shape) — if it differs, the comparison logic here needs revisiting.

**Further limitation: KS tests marginal uniformity only, not independence over time**

KS checks whether the *pooled* $\{u_t\}$ values collectively resemble a Uniform(0,1) sample — but doesn't check whether consecutive $u_t$ are independent of each other. A model could have $u_t$ systematically clustered near 1 during genuine high-volatility stretches (underestimating $\sigma_t$) and clustered near 0.5 during calm stretches (overestimating $\sigma_t$) — a real, time-varying miscalibration pattern — while these two regimes balance out such that the *pooled* distribution still looks uniform overall, passing KS despite the model having a clear, systematic timing problem. This kind of volatility-clustering mismatch is a plausible failure mode for both EGARCH and PF-SV specifically, and isn't detected by KS alone.

**How to test for this directly**

Transform $u_t$ via $z_t = \Phi^{-1}(u_t)$ (should be iid $N(0,1)$ under correct calibration). Then:
- Test autocorrelation of $z_t$ at short lags (e.g. Ljung-Box test) — checks for directional serial dependence.
- Test autocorrelation of $z_t^2$ at short lags (analogous to Engle's ARCH-LM test, applied to the calibration residual rather than raw returns) — checks for leftover volatility clustering the model failed to capture.

A significant result on either test indicates the model has systematic, time-varying miscalibration that the pooled KS test cannot detect.

## Why No Single Test Suffices

Unlike return forecasting, where the true outcome is directly observable, volatility is fundamentally latent — even with perfect data, $\sigma_t^{true}$ is never directly observed, only proxied. Because of this, there is no single "final" test analogous to comparing a return forecast against the realized return.

Each metric in this evaluation framework (QLIKE, RV coverage, PIT/KS) is instead an indirect, partial lens onto model quality, each with its own blind spots (see Sources of Error and the leverage/clustering discussion above). Disagreement between metrics is itself informative — a model passing QLIKE but failing PIT/KS, for example, indicates a specific kind of miscalibration (distributional shape or timing) rather than a magnitude problem — even though no single number can ever confirm the model matches the unobservable truth exactly.





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



This recovers the familiar plug-in formula, but reveals why it's only valid in this special case: **constant volatility assumes away exactly the forward-uncertainty problem that makes the general (time-varying) case require forward simulation.** Under constant volatility, there's no distinction between "the model's estimate at time $t$" and "the true expected future variance," since nothing changes going forward. For genuinely time-varying models like EGARCH/PF-SV, this shortcut is not valid — $E[\sigma_{t+i}^2\mid\mathcal{F}_t]$ must instead be estimated via Monte Carlo forward simulation (propagating posterior parameter draws and simulated future shocks through the model's recursion, then averaging $\sigma_{t+i}^2$ across simulated paths), not by directly plugging in a single $\sigma_{t+i}$ value obtained from fitting/smoothing over data beyond time $t$.

### Do the Two Estimators Agree?

Both target the same $\sigma^2$ under the constant-volatility assumption, but via genuinely different statistical arguments: the empirical side relies on the LLN across individual sample observations (needs $h\to\infty$ for consistency, has finite-sample bias corrected via $h-1$); the model side relies on martingale-difference cancellation of cross-terms in a sum (exact for any $h$, given the model specification is correct, no bias correction needed). They estimate the same target, but they are not the same estimator, and their differing denominators ($h-1$ vs. $h$) reflect this: it is not a stylistic inconsistency, but a direct consequence of the different derivations underlying each.


Patton's robustness result and empirical demonstration operate at daily granularity — a daily forecast scored against a daily proxy. This project instead applies QLIKE to window-aggregated (21-day RMS) quantities on both sides. This extension is plausible (conditional unbiasedness of the daily proxy should propagate to the window-averaged proxy, since expectation is linear), but it is an assumption carried over from the daily-level theory, not something explicitly proven for the aggregated case in the cited paper.

- **Empirically vaid**: log-realized-volatility is widely documented to be approximately normally distributed, even though volatility itself is skewed (Andersen-Bollerslev-style realized volatility literature; the mechanism behind this regularity is itself debated — see e.g. "Volatility Is Log-Normal, But Not for the Reason You Think" — but the empirical pattern itself is well established across many studies).

- **Rolling average, PF-SV**: $r_t \sim N(0, (\hat\sigma_t^{model})^2)$
- **EGARCH(1,1)-t**: $r_t \sim t_\nu(0, (\hat\sigma_t^{model})^2)$ (scaled Student-t, $\nu$ fit as part of the model)


**Why this test, given RV coverage already exists?**
- Independent of the RV-side noise correction ($\hat\sigma_\eta$) — operates purely on daily returns and the model's own distributional assumption, no separate proxy-noise model needed.
- Different sensitivity: RV coverage checks window-level magnitude; PIT/KS checks day-by-day distributional calibration. A model could pass one and fail the other.
- Doesn't directly test volatility *level* miscalibration in isolation (only distributional shape/timing) → supporting metric, not primary.

**Return coverage**
- Analogous to RV coverage: % of returns within the model's credible interval, read directly off $u_t$ (e.g. 90% interval violated if $u_t<0.05$ or $u_t>0.95$).

**Caveat: PIT/KS jointly tests two things**
- Computing $u_t$ requires committing to a specific shape for $F_t$ — this is a separate assumption from whether $\hat\sigma_t^{model}$ itself is correct.
- A PIT/KS failure can't distinguish: (a) wrong volatility level, vs. (b) wrong innovation shape. Both produce the same symptom (non-uniform $u_t$).
- Because EGARCH already assumes fatter tails ($t_\nu$), an EGARCH PIT/KS failure is more diagnostically informative — the model already has flexibility to absorb ordinary fat-tail behavior, so a failure more likely points to the volatility level itself. A Gaussian model's (rolling average, PF-SV) failure is less conclusive — it could simply reflect the well-known fact that returns are fatter-tailed than Gaussian, regardless of whether the volatility level is right.
- ⚠ **To confirm**: verify PF-SV's actual innovation assumption in the implementation (Gaussian assumed above, matching the rolling average's imposed shape) — if it differs, the comparison logic here needs revisiting.





## Metrics Overview

Since true volatility is latent (never directly observed, even after the fact), no single test can confirm a model is "correct." Each metric below targets a different practical question a volatility model needs to answer well — collectively, they build up a picture of model quality that no individual test could give alone.

### Historical Volatility Comparison
**Point of this category**: does the model's volatility *magnitude* track reality at all? Before asking anything more sophisticated, this checks the most basic requirement — a model that gets the overall size of volatility wrong (e.g. consistently predicting calm markets during a crisis) fails at the most fundamental task volatility estimation exists for for.

- **QLIKE (mean loss)**: gives a single number to rank models by magnitude fit, weighted to specifically penalize the costlier kind of mistake — underestimating volatility (the direction of error that actually gets traders hurt, per the project's own motivation).
- **RV Coverage**: magnitude fit alone isn't enough — a model's *stated confidence* about that magnitude matters too, since traders and risk systems act on the width of the uncertainty, not just the central estimate. This checks whether the model is honest about how sure it should be.

### Return Distribution Calibration
**Point of this category**: volatility isn't just used as a single number — it's used to build entire risk/return distributions (VaR, option pricing, hedge ratios). A model can have the right *average* volatility while still giving badly wrong day-to-day risk assessments. This category checks whether the model's implied return distribution — the actual object used for real decisions — is trustworthy on a daily basis, not just on average.

- **PIT/KS**: the direct test of whether the model's day-by-day claimed uncertainty matches what actually happens — this is the calibration property that any practical use of the model (sizing a position, pricing a tail-risk hedge) implicitly relies on.
- KS test / histogram: the formal and visual versions of this same check.

### Serial Dependence Diagnostics (ACF-based)
**Point of this category**: a model can look well-calibrated *on average* while still having systematic, exploitable patterns in *when* it fails — and these patterns matter economically, since they concentrate model risk exactly when it's most costly (crises, trending markets) rather than spreading it out harmlessly. This category asks not "is the model right on average" but "are the model's mistakes predictable" — a model with predictable mistakes is arguably more dangerous than one with larger but random ones, since predictable errors can be systematically exploited or can compound at exactly the wrong moment.

- **ACF of $z_t$**: are miscalibration errors predictable in *direction* over time?
- **ACF of $z_t^2$**: are miscalibration errors predictable in *magnitude/clustering* over time — i.e., does the model fail to adapt fast enough to genuine regime changes?
- **Engle-Ng sign-bias test**: does the model's error depend on the *direction* of the previous day's return — directly testing whether it captures the leverage effect that's central to why EGARCH exists in the first place, and central to the "hedge based on volatility direction" use case from the project's own motivation.

## Metrics Overview

### Historical Volatility Comparison
Checks similarity between model and empirical volatility estimates across a window.

- **QLIKE (mean loss)**: single aggregate number summarizing model-vs-empirical fit, with built-in asymmetric penalty for underprediction.
- **RV Coverage**: directly checks whether empirical coverage matches expected model coverage at various credible-interval levels (50%, 90%, 95%, etc.), after adjusting for the benchmark's own known estimation noise (log-normal correction).

### Return Distribution Calibration
Checks whether the model's *day-by-day* distributional assumption is well-calibrated.

- **PIT/KS**: tests whether $u_t=F_t(r_t)$ is uniformly distributed — jointly checks volatility level and assumed innovation shape (can't separate the two from this test alone).
  - KS test: formal test statistic for uniformity of $\{u_t\}$.
  - Histogram/empirical CDF: visual diagnostic for the same.

### Serial Dependence Diagnostics (ACF-based)
Checks whether calibration errors are independent over time — a dimension pooled PIT/KS cannot detect on its own. All computed on $z_t=\Phi^{-1}(u_t)$.

- **ACF of $z_t$ (level, Ljung-Box)**: tests for *directional* persistence in calibration errors (e.g. systematic runs of under/overestimation).
- **ACF of $z_t^2$ (squared, ARCH-LM style)**: tests for leftover *volatility clustering* the model failed to capture.
- **Engle-Ng sign-bias test**: regress $z_t^2$ on $\text{sign}(r_{t-1})$ (and interaction terms with $r_{t-1}$ itself) — directly tests for *leverage/asymmetry* miscalibration, i.e. whether calibration errors depend systematically on the sign of the previous return. This is the concrete implementation of the leverage-testing gap discussed earlier, and follows the same regression-based logic as the ACF checks above.
- **Cross-correlation of $z_t^2$ with lagged $|r_{t-1}|$ or $r_{t-1}^2$**: checks whether error magnitude relates to the *size* (not just sign) of recent shocks — a complementary check to the ARCH-LM-style test above, at a finer level of detail.

### Method

1. Sample parameters from their priors (as specified in the Priors section above).
2. Propagate the EGARCH recursion forward through the regression period's returns, generating the implied $\sigma_t$ path for that specific parameter draw.
3. Use this $\sigma_t$ path to evaluate the likelihood of the full regression-period returns under a Student-t observation model, and let NUTS use this (and its gradient) to inform the next proposal.
4. Repeat across many draws to build up the posterior.




### NUTS: Extending Metropolis-Hastings with Gradient Information

**Setup**: introduce auxiliary momentum $p\in\mathbb{R}^d$ (same dimension as $\theta$, artificial — discarded each iteration), $p\sim N(0,M)$. Define potential energy $U(\theta)=-\log\pi(\theta)$ and joint density $\pi(\theta,p)\propto\exp(-U(\theta)-\frac12 p^TM^{-1}p)$. Marginalizing out $p$ recovers $\pi(\theta)$ exactly.

**Leapfrog integrator** (one step, size $\epsilon$) — this is where $\nabla\log\pi(\theta)$ enters:
$$p_{1/2} = p + \frac{\epsilon}{2}\nabla\log\pi(\theta), \qquad \theta' = \theta + \epsilon M^{-1}p_{1/2}, \qquad p' = p_{1/2} + \frac{\epsilon}{2}\nabla\log\pi(\theta')$$
Repeating this $L$ times traces a trajectory $(\theta,p)\to\cdots\to(\theta_L,p_L)$, pushed toward higher posterior density at every step by the gradient.

**Basic HMC acceptance** (still Metropolis-Hastings):
$$\alpha = \min\big(1,\ \exp(-U(\theta_L)-K(p_L)+U(\theta_0)+K(p_0))\big), \quad K(p)=\tfrac12 p^TM^{-1}p$$
This is the same MH acceptance rule as before, applied to the joint $(\theta,p)$ system — detailed balance still holds, since leapfrog is reversible and volume-preserving.

**NUTS's addition — adaptive trajectory length**: rather than fixing $L$, NUTS doubles the trajectory in a randomly chosen direction at each iteration, stopping via the **No-U-Turn criterion**: for trajectory endpoints $\theta^-,\theta^+$ with momenta $p^-,p^+$,
$$(\theta^+-\theta^-)\cdot p^- \ge 0 \quad\text{and}\quad (\theta^+-\theta^-)\cdot p^+ \ge 0$$
Doubling continues while both hold; the moment either goes negative, the trajectory has curved back on itself and doubling stops. The next sample is drawn uniformly from the valid (slice-sampling-qualified) points generated across the whole trajectory — not just its endpoint — which preserves detailed balance exactly despite the trajectory's length varying dynamically.

**Step size $\epsilon$**: tuned automatically during warmup via dual averaging, targeting a specified acceptance rate (e.g. 0.95).

**Summary**: Steps 1–4 (evaluating $\log\pi(\theta)$) are unchanged for any sampler. NUTS replaces only the proposal mechanism — blind random-walk $\to$ gradient-driven leapfrog trajectory with adaptive length — while the correctness guarantee (detailed balance, ergodicity, convergence) established for basic Metropolis-Hastings continues to hold unchanged.























QLIKE involves two distinct averaging operations, over two different axes:
1. Across posterior draws, at each fixed day: $\hat\sigma_t^{model} = \sqrt{\frac{1}{S}\sum_s(\sigma_t^{(s)})^2}$ — required by the theoretical derivation ($E[\sigma_{t+i}^2\mid\mathcal{F}_t]$), producing one point estimate per day before any comparison happens.
2. Across time, after computing daily QLIKE values: $\overline{\text{QLIKE}} = \frac{1}{T}\sum_t \text{QLIKE}(\hat\sigma_t^{model}, RV_t)$ — the standard "mean QLIKE" summary statistic used for reporting/ranking models.
These are not interchangeable or reorderable — the per-day posterior collapse must happen first, since QLIKE requires one scalar $\hat\sigma_t$ per day to even be computed.

Theoretical.md

### How QLIKE Fits Into This

QLIKE compares the model's **mean historical volatility** ($\hat\sigma_t^{model}$) against the empirical historical volatility ($\hat\sigma_t^{empirical}$) — and the mean historical volatility is, by construction, always computed via Method 2 (average across posterior draws *before* applying any further formula), never Method 1.

This isn't an arbitrary implementation choice specific to QLIKE — it follows directly from how $\hat\sigma_t^{model}$ itself is defined and derived (see Historical Volatility, above): the martingale-difference/RMS derivation produces $E[\sigma_{t+i}^2\mid\mathcal{F}_t]$ as the correct per-day target, which requires averaging across the posterior (or simulated paths) *before* the window-RMS aggregation, for the same Jensen's-inequality reason established earlier. $\hat\sigma_t^{model}$ is Method 2 by definition — there is no alternative "Method 1 version" of the mean historical volatility itself, since the quantity being estimated is an expectation, not a collection of separately-scored draws.

QLIKE, in turn, simply takes this already-computed $\hat\sigma_t^{model}$ (Method 2's output) and compares it once per day against $\hat\sigma_t^{empirical}$:
$$\text{QLIKE}_t = \text{QLIKE}\left(\hat\sigma_t^{model}, \hat\sigma_t^{empirical}\right)$$
QLIKE itself never touches the posterior draws directly — it only ever sees the single, already-averaged $\hat\sigma_t^{model}$ series. The subsequent "mean QLIKE" (averaging $\text{QLIKE}_t$ across days $t$, for reporting) is a separate averaging step over a different axis (time, not posterior draws) — see the note above.

This is why QLIKE and PIT differ in which method they use: QLIKE inherits Method 2 automatically, simply by using $\hat\sigma_t^{model}$ (which is always Method 2 by construction) as its input. PIT, by contrast, is not built from $\hat\sigma_t^{model}$ at all — it evaluates a distributional CDF at the actual return, which is why it independently uses Method 1 (averaging the CDF evaluation across draws), rather than inheriting Method 2 from the HV comparison machinery.


## Historical Volatility Coverage for Posterior-Based Models

Unlike the baseline (a single point forecast, where the log-normal noise quantile has a closed-form solution), EGARCH and PF-SV produce $S$ posterior draws — requiring the coverage interval to account for two distinct, independent sources of uncertainty simultaneously: the model's own posterior spread, and the empirical benchmark's measurement noise ($\hat\sigma_\eta$).

### Step 0: What Is Already Available

$$\text{hv\_per\_draw}[s,t] = \sqrt{\frac{1}{h}\sum_{i=1}^h \left(\sigma_{t+i}^{(s)}\right)^2}, \qquad s=1,...,S$$

One window-aggregated historical volatility value per posterior draw, at every day $t$ — this reflects the model's own posterior uncertainty about the true HV at day $t$, before any benchmark noise is considered.

### Step 1: Recall the Log-Normal Noise Model

$$\hat\sigma_t^{empirical} = \sigma_t^{true}\cdot\exp(\eta_t), \qquad \eta_t \sim N(0,\hat\sigma_\eta^2)$$

$\eta_t$ represents the empirical benchmark's own measurement noise — a source of uncertainty entirely separate from, and independent of, the model's posterior spread.

### Step 2: Simulate a Noisy Realization for Each Draw

For each posterior draw $s$ at each day $t$, draw an independent noise realization and apply it multiplicatively to that draw's own HV value:

$$\text{noisy\_hv}[s,t] = \text{hv\_per\_draw}[s,t] \cdot \exp\left(\eta_t^{(s)}\right), \qquad \eta_t^{(s)} \overset{iid}{\sim} N(0,\hat\sigma_\eta^2)$$

Each cell of the resulting $(S,T)$ array represents one plausible realization of "what a noisy empirical measurement might show, if posterior draw $s$ described the true state" — jointly simulating both sources of uncertainty together, rather than combining them analytically.

### Step 3: Take Percentiles Across the Combined Ensemble

$$\text{lower}_t = \text{Percentile}_5\Big(\{\text{noisy\_hv}[1,t],...,\text{noisy\_hv}[S,t]\}\Big), \qquad \text{upper}_t = \text{Percentile}_{95}\Big(\{\text{noisy\_hv}[1,t],...,\text{noisy\_hv}[S,t]\}\Big)$$

### Why Simulation Is Required (Not an Analytic Shortcut)

For the baseline ($S=1$), the coverage interval reduces to a single log-normal distribution's quantile, computable analytically: $\hat\sigma_t^{roll}\cdot\exp(\pm z\cdot\hat\sigma_\eta)$.

For $S>1$, the combined distribution is a **mixture** of $S$ log-normal distributions (one centered at each draw's own HV value) — mixtures of log-normals have no closed-form quantile in general, so simulation (Steps 2–3 above) is the standard, correct way to obtain percentiles. This is the same underlying question in both cases — "what range of empirical values is plausible, given the model's estimate(s) and known benchmark noise?" — the baseline is simply the degenerate $S=1$ special case that happens to admit an exact analytic shortcut.

### Coverage Check

Exactly as for the baseline, coverage is checked against the interval constructed above:
$$\text{coverage}_t = \mathbb{1}\left[\text{lower}_t \le \hat\sigma_t^{empirical} \le \text{upper}_t\right]$$

### Data

Historical volatility (HV) is inherently retrospective: computing $\hat\sigma_t^{empirical}$ for any day requires the following 21 days of returns, which are never available in real time. This applies uniformly throughout the dataset (including at period and window boundaries) and reflects HV's role purely as a benchmark for post-hoc evaluation, not as a live/real-time forecasting quantity.



Rolling/historical volatility are computed on the full continuous return series, then sliced by period. Values near any boundary (period transitions, or model-internal window transitions) reflect returns from the adjacent period/window, since both are inherently backward/forward-looking by construction — this is a universal property of these estimators, not specific to any one boundary, and does not affect the underlying models' own out-of-sample forecast validity (see theoretical.md).



### Should $\hat\sigma_\eta$ Vary by Posterior Draw?

$\hat\sigma_\eta$ is computed once, globally, using the **mean** forecast series ($\hat\sigma_t^{model}$, the Method 2 posterior average) against the actual empirical HV — not per posterior draw. Each simulated noise realization $\eta_t^{(s)}$ is drawn independently for every $(s,t)$ cell, but all from the same shared $N(0,\hat\sigma_\eta^2)$ distribution.

**Why not compute a separate $\hat\sigma_\eta^{(s)}$ per draw?**

In principle, each draw's own HV series could be compared against the actual individually:
$$\hat\sigma_\eta^{(s)} = \text{std}_t\left(\log\hat\sigma_t^{empirical} - \log\hat\sigma_t^{(s)}\right)$$
giving each draw its own noise-correction magnitude, rather than sharing one global value.

This was considered and rejected, for two reasons:

1. **Conceptual**: $\sigma_\eta$ is meant to represent the empirical benchmark's own measurement noise (Sources of Error, point 1) — a property of $\hat\sigma_t^{empirical}$ alone, which has no reason to vary depending on which posterior draw it's being compared against. A shared, draw-independent $\hat\sigma_\eta$ is more consistent with this interpretation.
2. **Circularity risk**: computing $\hat\sigma_\eta^{(s)}$ from a specific draw's own deviation from the actual would widen that draw's own interval specifically *because* it deviates — potentially masking genuine miscalibration for poorly-fitting draws by construction, since the correction is derived from the very discrepancy being tested. This is a more severe version of the circularity concern already flagged for the global, regression-period $\hat\sigma_\eta$.

The known trade-off (already documented under "What $\hat\sigma_\eta$ Actually Estimates") remains: the shared $\hat\sigma_\eta$ is contaminated by *some* average model error across all draws combined ($\hat\sigma_\eta^2 \approx \sigma_\eta^2 + \text{Var}(\epsilon_t^{model})$), but this is judged preferable to a per-draw version that risks self-fulfilling interval widening.

### Note: The Mean Line May Occasionally Sit Outside Its Own Percentile Band

The mean HV forecast (Method 2: average across draws before the square root) and the percentile band (built from each draw's own square-rooted HV, plus noise) are computed via different, individually-correct orderings of the same quantities. By Jensen's inequality, the average of per-draw square roots is always ≤ the square root of the average — empirically, this gap is approximately 10% of the typical HV level for EGARCH's sequential forecasts, reflecting genuine posterior spread in $\sigma_t$ across draws. In windows where the combined posterior-spread-plus-noise band is comparably narrow, the mean line can sit above the band's upper percentile — a structural consequence of using the theoretically correct aggregation order for each quantity separately, not a plotting or alignment error.

### ACF as a Single-Chain Mixing Diagnostic (PMCMC)

Since PMCMC here runs a single chain (no multi-chain R-hat comparison available, unlike NUTS's parallel chains), the primary convergence/efficiency diagnostic is the autocorrelation function (ACF) of each post-burn-in parameter trace. Slow decay (autocorrelation remaining significant over many lags) directly corresponds to low effective sample size (see ESS, above) — the raw sample count overstates how much independent information the chain actually contains. Fast decay indicates efficient mixing, consistent with a well-tuned proposal and acceptance rate.

DM Test 
50 day return series plot