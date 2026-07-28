Extra Stuff

LLN argument and the quadratic variation argument

Key idea is mesh size of quadratic variation.
This concept doesn't hold in LLN since the day's stuff only matters





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