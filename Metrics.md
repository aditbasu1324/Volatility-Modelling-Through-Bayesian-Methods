# Volatility Metrics

## Historical Volatility (Benchmark)

### The Goal
Under constant volatility within the window, conditional on $\mathcal{F}_t$: $\text{Var}(r_{t+1},...,r_{t+h}\mid\mathcal{F}_t) = \sigma^2$. Two estimators of this target are constructed — empirical (from raw returns) and model-based — and compared.

### Empirical Estimator (via LLN)
$$\hat{\sigma}_t^{empirical} = \sqrt{\frac{1}{h-1}\sum_{i=1}^{h}(r_{t+i} - \bar{r}_t)^2}, \quad r_t=\ln(P_t/P_{t-1}),\ \bar r_t=\frac1h\sum_i r_{t+i},\ h=21$$
Sample variance converges to $\sigma^2$ by LLN as $h\to\infty$. $h-1$ (Bessel's correction) accounts for $\mu$ being estimated by $\bar r_t$ from the same sample.

### Model Estimator (via Martingale-Difference Sum)
All models specify $r_{t+i}=\mu+\sigma_{t+i}\varepsilon_{t+i}$, $\varepsilon_{t+i}$ iid mean-zero, $\sigma_{t+i}=\text{Var}(r_{t+i}\mid\mathcal{F}_{t+i-1})$. Only $\sigma_{t+1}$ is $\mathcal{F}_t$-measurable; for $i>1$, $\sigma_{t+i}$ is itself random given $\mathcal{F}_t$.

**Cross-terms vanish** (tower property, $i<j$):
$$E[r_{t+i}r_{t+j}\mid\mathcal{F}_t] = \mu\,E[r_{t+i}\mid\mathcal{F}_t] = E[r_{t+i}\mid\mathcal{F}_t]\cdot E[r_{t+j}\mid\mathcal{F}_t] \ \Rightarrow\ \text{Cov}(r_{t+i},r_{t+j}\mid\mathcal{F}_t)=0$$

**Diagonal terms** (law of total variance):
$$\text{Var}(r_{t+i}\mid\mathcal{F}_t) = \underbrace{E[\text{Var}(r_{t+i}\mid\mathcal{F}_{t+i-1})\mid\mathcal{F}_t]}_{E[\sigma_{t+i}^2\mid\mathcal{F}_t]} + \underbrace{\text{Var}(\mu\mid\mathcal{F}_t)}_{=0} = E[\sigma_{t+i}^2\mid\mathcal{F}_t]$$

**Result**:
$$\text{Var}\left(\sum_{i=1}^h r_{t+i}\,\middle|\,\mathcal{F}_t\right) = \sum_{i=1}^h E[\sigma_{t+i}^2\mid\mathcal{F}_t]$$

**Consistency check (constant volatility)**: substituting $\sigma_{t+i}=\sigma$ gives $\sum_i E[\sigma_{t+i}^2\mid\mathcal{F}_t]=h\sigma^2$, confirming the model-side formula recovers the same $\sigma^2$ as the empirical (LLN) estimator:
$$\hat\sigma_t^{model} = \sqrt{\frac{1}{h}\sum_{i=1}^h E[\sigma_{t+i}^2\mid\mathcal{F}_t]}$$
Note: $\sigma_{t+i} \ne E[\sigma_{t+i}^2\mid\mathcal{F}_t]$ in general — addressed in implementation.md (forward simulation).

### A Note on Terminology and a Key Assumption

In the literature (e.g. Andersen-Bollerslev-Diebold-Labys), "realized volatility" typically refers to aggregating squared *intraday* returns to estimate a single day's variance, with consistency guaranteed via quadratic variation as sampling frequency increases — regardless of how volatility moves within the period. The construction used here — sample standard deviation of 21 *daily* returns — is a related but distinct estimator, more accurately termed **historical volatility**.

This construction's validity depends on assuming volatility is constant across the window, directly in tension with the time-varying volatility EGARCH/SV are designed to capture. This is a limitation of the current daily-frequency data rather than a conceptual one: intraday-derived daily RV would not require this assumption (see extensions.md), and this project's PIT/return calibration test (evaluated day-by-day) already provides partial sensitivity to within-window timing errors that HV/QLIKE cannot see.

### Why Historical Volatility (and Model RMS) Can't Detect Leverage Directly

Both $\hat\sigma_t^{empirical}$ (sample variance) and $\hat\sigma_t^{model}$ (RMS of $\sigma_{t+i}^2$) are symmetric under reordering and sign-flipping of the underlying returns — squaring removes sign, summation discards sequencing. Neither can represent a conditional (sign-dependent) relationship like leverage; both only ever produce a single scalar summarizing average variance over the window.

This means leverage miscalibration cannot be detected via the aggregate window-level comparison directly — it is only exposed indirectly, via the Engle-Ng sign-bias test (below, which operates on daily returns rather than the window-level HV comparison) or by comparing sub-period breakdowns (e.g. sustained one-directional crisis windows, where sign-dependent errors have no opposing-sign counterpart to cancel against).

### Sources of Error
Even under a correctly specified model, $\hat\sigma_t^{empirical}$ and $\hat\sigma_t^{model}$ won't match exactly:
1. **Systematic estimation error** — $\hat\sigma_t^{empirical}$ is a finite-sample ($h=21$) estimator, carrying residual variance regardless of model correctness.
2. **Genuine model error** — misspecification/approximation error.

A fair comparison requires factoring error 1 into the model side too, via a log-normal noise model (below).

**Note**: Patton's original robustness result compares daily forecasts against a daily proxy; this project extends it to window-aggregated quantities on both sides. Not explicitly proven for the aggregated case, but plausible via linearity of expectation, and standard practice. This extension does not depend on the classical high-frequency (quadratic-variation) construction specifically — only conditional unbiasedness of the proxy is required, which historical volatility satisfies under the constant-volatility assumption above. What is affected is not validity but **statistical power**: a proxy built from 21 daily returns is noisier than a high-frequency one, giving these tests less power to distinguish between models.

## Historical Volatility Coverage: Log-Normal Measurement Error

<!-- FIXME: revisit this section for full correctness -->

### The Noise Model
$$\hat\sigma_t^{empirical} = \sigma_t^{true}\cdot\exp(\eta_t), \quad \eta_t\sim N(0,\sigma_\eta^2), \qquad \hat\sigma_\eta = \text{std}(\log\hat\sigma_t^{empirical}-\log\hat\sigma_t^{model})$$

**Why log-normal?**
- **Empirically supported**: log-historical-volatility is widely documented as approximately Normal across several studies.
- **Positivity**: multiplicative noise keeps $\hat\sigma_t^{empirical}>0$; additive Gaussian noise could push it negative.
- **Tractability**: one scalar parameter, easy to combine with the model's posterior on the log scale.
- A pragmatic convention, not theoretically mandated — more flexible alternatives (gamma, non-central $\chi^2$) exist but aren't clearly needed.

**What $\hat\sigma_\eta$ is meant to represent**: the estimation noise in $\hat\sigma_t^{empirical}$ alone, relative to the unobservable $\sigma_t^{true}$.

**Why $\hat\sigma_t^{model}$ substitutes for $\sigma_t^{true}$**: $\sigma_t^{true}$ is never observable — $\hat\sigma_t^{model}$ is the best available stand-in, a forced necessity rather than an independently justified choice.

**What $\hat\sigma_\eta$ actually estimates**: writing $\hat\sigma_t^{model}=\sigma_t^{true}\cdot\exp(\epsilon_t^{model})$,
$$\log\hat\sigma_t^{empirical}-\log\hat\sigma_t^{model} = \eta_t - \epsilon_t^{model} \quad(\log\sigma_t^{true}\text{ cancels})$$
$$\text{Var}(\eta_t-\epsilon_t^{model}) = \sigma_\eta^2 + \text{Var}(\epsilon_t^{model}) - 2\,\text{Cov}(\eta_t,\epsilon_t^{model})$$
Assuming $\eta_t\perp\epsilon_t^{model}$ (see caveat below), the covariance term drops and $\hat\sigma_\eta^2\approx\sigma_\eta^2+\text{Var}(\epsilon_t^{model})$ — a **combined** estimate of benchmark noise and model error, not benchmark noise alone; the two cannot be separated from this single computable quantity.

**Is independence actually justified?** Weakly, at best — a convenience assumption, not a well-supported one. $\eta_t$'s source (finite-sample noise from a 21-day window) and $\epsilon_t^{model}$'s source (model misspecification) are mechanically distinct, giving informal grounds for low average correlation. But both plausibly **rise together during volatile/crisis periods** — a regime shift can simultaneously violate the empirical estimator's constant-volatility assumption and stress-test the model's functional form. If so, dropping the covariance term understates the true combined variance exactly when it matters most. <!-- TODO: cross-model sigma_eta comparison as an indirect check -->

**Sources of error this corrects for (and doesn't)**:
- **Corrects for**: systematic estimation error in $\hat\sigma_t^{empirical}$ — widening the credible interval by $\hat\sigma_\eta$ prevents this from being mistaken for model miscalibration.
- **Does not cleanly separate from**: genuine model error — the correction is only accurate insofar as model error is small relative to true benchmark noise.

**Does more data help?** Yes — $\sigma_\eta$ reflects $\hat\sigma_t^{empirical}$'s imprecision, itself a consequence of LLN over only 21 daily observations. Intraday-derived RV (quadratic variation over many high-frequency returns per day) would shrink $\sigma_\eta$, attributing a larger share of $\hat\sigma_\eta$ to genuine model error — sharpening the test's diagnostic power (see extensions.md).

**Is this circular?**
- **Regression**: $\hat\sigma_\eta$ fit once, globally — not tuned per-window, so coverage can't be trivially gamed.
- **Sequential**: no ambiguity — computed only from data prior to each window, genuinely out-of-sample.

## QLIKE Loss
$$\text{QLIKE}(\hat\sigma_t,RV_t) = \frac{RV_t^2}{\hat\sigma_t^2} - \log\left(\frac{RV_t^2}{\hat\sigma_t^2}\right) - 1$$
Lower mean QLIKE = better fit.

**Why QLIKE?**
- **Asymmetric underprediction penalty**: for any proportional miss by factor $k$, underpredicting incurs strictly more loss than overpredicting ($\text{Loss}_{under}(k)-\text{Loss}_{over}(k) = k^2-1/k^2-4\log k > 0$ for all $k>1$) — reflecting that underestimating volatility is typically costlier in practice.
- **Robust to a noisy proxy**: model rankings under QLIKE match what they'd be under the true (unobservable) volatility (Patton, 2011) — the main reason QLIKE, not MSE, is standard here, since it directly addresses $RV_t$'s own estimation error rather than being distorted by it.

## Historical Volatility Coverage
Coverage = % of $\hat\sigma_t^{empirical}$ values falling within the model's credible interval for $\hat\sigma_t^{model}$, widened using $\hat\sigma_\eta$.

## Return Calibration Test (PIT / KS)
$$u_t = F_t(r_t)$$
$F_t$ = model's estimated return CDF at $t$ (from $\hat\sigma_t^{model}$); $u_t$ = fraction of $F_t$ lying below the actual return.

**PIT theorem**: if $r_t\sim F_t$ truly, $u_t\sim\text{Uniform}(0,1)$ — general, distribution-free. Intuition: under correct calibration, the actual return is equally likely to land at any percentile. Too many $u_t$ near 0/1 → intervals too narrow.

**KS test**: compares empirical CDF of $\{u_t\}$ against Uniform(0,1) (Massey, 1951); $p<0.05$ → not well calibrated.

## Serial Dependence Diagnostics
KS checks *pooled* uniformity only — not whether consecutive $u_t$ are independent. A model could have $u_t$ near 1 in high-vol stretches and near 0.5 in calm ones — a real miscalibration pattern — while the pooled distribution still looks uniform.

$u_t$ is transformed via $z_t=\Phi^{-1}(u_t)$ (iid $N(0,1)$ under correct calibration), then tested for clustering, persistence, and leverage.

### ACF of $z_t$ (Ljung-Box) — directional persistence
$$\hat\rho_j = \frac{\sum_{t=j+1}^n(z_t-\bar z)(z_{t-j}-\bar z)}{\sum_{t=1}^n(z_t-\bar z)^2}, \qquad Q(k)=n(n+2)\sum_{j=1}^k\frac{\hat\rho_j^2}{n-j}\sim\chi_k^2 \text{ under } H_0$$
$H_0$: $\rho_1=...=\rho_k=0$. High $p$ → no directional persistence. Low $p$ → errors run in the same direction over time.

### ACF of $z_t^2$ (Ljung-Box, ARCH-LM style) — volatility clustering
Same test on $z_t^2$ (sign removed). High $p$ → model captures true clustering, errors don't bunch up. Low $p$ → leftover clustering, model adapts too slowly to regime shifts.

### Engle-Ng Sign-Bias Test — leverage effect
$$z_t^2 = \beta_0 + \beta_1\cdot\text{sign}(r_{t-1}) + \epsilon_t$$
Predicted squared error: $\beta_0-\beta_1$ after a down day, $\beta_0+\beta_1$ after an up day. $H_0$: $\beta_1=0$. Significant $\beta_1<0$ → larger errors after negative returns — model fails to capture leverage.

(Extended sign/size-bias tests add $r_{t-1}\mathbb{1}[r_{t-1}<0]$ and $r_{t-1}\mathbb{1}[r_{t-1}>0]$ as regressors, testing magnitude as well as sign.)

A significant result on any of these three tests indicates systematic, time-varying miscalibration the pooled KS test cannot detect.

### Remaining Test Coverage Gaps

| Dimension | Covered by | Gap |
|---|---|---|
| Window-average magnitude | QLIKE, HV comparison | — |
| Window-average uncertainty | HV Coverage | — |
| Day-level distributional calibration | PIT/KS | Can't separate level vs. shape error |
| Directional/clustering error persistence | ACF of $z_t$, $z_t^2$ | — |
| Leverage/asymmetry response | Engle-Ng | Structurally invisible to HV/QLIKE directly (see above) |
| Tail-specific calibration (VaR) | *(not covered)* | Needs Kupiec/Christoffersen tests |
| Multi-horizon path accuracy | *(not covered)* | Needs term-structure evaluation |
| Economic/trading value | *(not covered)* | Needs backtested strategy |
| Within-window timing accuracy | *(partial, via PIT)* | Needs intraday-derived daily RV for full resolution |

### Why No Single Test Suffices

Unlike return forecasting, where the true outcome is directly observable, volatility is fundamentally latent — even with perfect data, $\sigma_t^{true}$ is never directly observed, only proxied. There is no single "final" test analogous to comparing a return forecast against the realized return.

Each metric above is instead an indirect, partial lens onto model quality, each with its own blind spots. Disagreement between metrics is itself informative — a model passing QLIKE but failing PIT/KS indicates a specific kind of miscalibration (distributional shape or timing) rather than a magnitude problem — even though no single number can confirm the model matches the unobservable truth exactly.