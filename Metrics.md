# Volatility Metrics

### Why No Single Test Suffices

Unlike return forecasting, where the true outcome is directly observable, volatility is fundamentally latent — even with perfect data, $\sigma_t^{true}$ is never directly observed, only proxied. There is no single "final" test analogous to comparing a return forecast against the realized return.

Each metric below is an indirect method of assessing model quality, each with its own blind spots. Overall, these metrics build up a composite picture of model quality that no single test could give alone — disagreement between metrics is itself informative (e.g. passing QLIKE but failing PIT/KS points to a shape issue rather than a magnitude one).

## Historical Volatility 

### The Goal
Under constant volatility within the window, conditional on $\mathcal{F}_t$: $\text{Var}(r_{t+1},...,r_{t+h}\mid\mathcal{F}_t) = \sigma^2$. Two estimators of this target are constructed — empirical (from raw returns) and model-based — and compared.

### Empirical Estimator (via LLN)
$$\hat{\sigma}_t^{empirical} = \sqrt{\frac{1}{h-1}\sum_{i=1}^{h}(r_{t+i} - \bar{r}_t)^2}, \quad r_t=\ln(P_t/P_{t-1}),\ \bar r_t=\frac1h\sum_i r_{t+i},\ h=21$$
Sample variance converges to $\sigma^2$ by LLN as $h\to\infty$. $h-1$ (Bessel's correction) accounts for $\mu$ being estimated by $\bar r_t$ from the same sample.

### Model Estimator (via Martingale-Difference Sum)
All models specify $r_{t+i}=\mu+\sigma_{t+i}\varepsilon_{t+i}$, $\varepsilon_{t+i}$ iid mean-zero, $\sigma_{t+i}=\text{Var}(r_{t+i}\mid\mathcal{F}_{t+i-1})$. Only $\sigma_{t+1}$ is $\mathcal{F}_t$-measurable; for $i>1$, $\sigma_{t+i}$ is itself random given $\mathcal{F}_t$.

**Cross-terms vanish** (tower property, $i<j$):
$$E[r_{t+i}r_{t+j}\mid\mathcal{F}_t] = E\big[r_{t+i}\cdot E[r_{t+j}\mid\mathcal{F}_{t+j-1}]\mid\mathcal{F}_t\big] = \mu\,E[r_{t+i}\mid\mathcal{F}_t] = E[r_{t+i}\mid\mathcal{F}_t]\cdot E[r_{t+j}\mid\mathcal{F}_t] \ \Rightarrow\ \text{Cov}(r_{t+i},r_{t+j}\mid\mathcal{F}_t)=0$$

**Diagonal terms** (law of total variance):
$$\text{Var}(r_{t+i}\mid\mathcal{F}_t) = \underbrace{E[\text{Var}(r_{t+i}\mid\mathcal{F}_{t+i-1})\mid\mathcal{F}_t]}_{E[\sigma_{t+i}^2\mid\mathcal{F}_t]} + \underbrace{\text{Var}(\mu\mid\mathcal{F}_t)}_{=0} = E[\sigma_{t+i}^2\mid\mathcal{F}_t]$$

**Result**:
$$\text{Var}\left(\sum_{i=1}^h r_{t+i}\,\middle|\,\mathcal{F}_t\right) = \sum_{i=1}^h E[\sigma_{t+i}^2\mid\mathcal{F}_t]$$

**Consistency check (constant volatility)**: substituting $\sigma_{t+i}=\sigma$ gives $\sum_i E[\sigma_{t+i}^2\mid\mathcal{F}_t]=h\sigma^2$, confirming the model-side formula recovers the same $\sigma^2$ as the empirical (LLN) estimator:
$$\hat\sigma_t^{model} = \sqrt{\frac{1}{h}\sum_{i=1}^h E[\sigma_{t+i}^2\mid\mathcal{F}_t]}$$
Note: $\sigma_{t+i} \ne E[\sigma_{t+i}^2\mid\mathcal{F}_t]$ For simplicity, the $\mathcal{F}_t$ condition isn't considered so far (it wouldn't make sense for regression because all the data is looked at, for sequential it would require forward simulations)

A fair comparison requires factoring error 1 into the model side too, via a log-normal noise model (below).

## Historical Volatility Coverage: Log-Normal Measurement Error

### The Noise Model
$$\hat\sigma_t^{empirical} = \sigma_t^{true}\cdot\exp(\eta_t), \quad \eta_t\sim N(0,\sigma_\eta^2), \qquad \hat\sigma_\eta = \text{std}(\log\hat\sigma_t^{empirical}-\log\hat\sigma_t^{model})$$

**Why log-normal?**
- **Empirically supported**: log-historical-volatility is widely documented as approximately normal across several studies.
- **Positivity**: multiplicative noise keeps $\hat\sigma_t^{empirical}>0$; additive Gaussian noise could push it negative.
- **Tractability**: one scalar parameter, easy to combine with the model's posterior on the log scale.
- A practical convention (more flexible alternatives (gamma, non-central $\chi^2$) exist).

**What $\hat\sigma_\eta$ is meant to represent**: the estimation noise in $\hat\sigma_t^{empirical}$ alone, relative to the unobservable $\sigma_t^{true}$.

$\hat\sigma_t^{model}$ substitutes for $\sigma_t^{true}$** since $\sigma_t^{true}$ is never observable

**What $\hat\sigma_\eta$ actually estimates**: writing $\hat\sigma_t^{model}=\sigma_t^{true}\cdot\exp(\epsilon_t^{model})$

$$\log\hat\sigma_t^{empirical}-\log\hat\sigma_t^{model} = \eta_t - \epsilon_t^{model} \quad(\log\sigma_t^{true}\text{ cancels})$$
$$\text{Var}(\eta_t-\epsilon_t^{model}) = \sigma_\eta^2 + \text{Var}(\epsilon_t^{model}) - 2\,\text{Cov}(\eta_t,\epsilon_t^{model})$$

### Combining Benchmark Noise with Model Error

The computed $\hat\sigma_\eta$ (assuming $\eta_t\perp\epsilon_t^{model}$) satisfies $\hat\sigma_\eta^2\approx\sigma_\eta^2+\text{Var}(\epsilon_t^{model})$ — but this involves three distinct points worth separating:

1. **Inseparability** — this is a *combined* estimate of benchmark noise and model error, not benchmark noise alone; the two cannot be separated from this single computable quantity.
2. **Weak independence justification** — $\eta_t$ (finite-sample noise) and $\epsilon_t^{model}$ (model misspecification) arise from mechanically distinct sources, giving informal grounds for low average correlation, but this isn't strongly established.
3. **Likely directional failure during crises** — both plausibly **rise together during crisis periods** (a regime shift can stress both the constant-volatility assumption and the model's functional form at once), meaning $\hat\sigma_\eta$ likely *understates* the true combined variance.

### Does More Data Help?

- $\sigma_\eta$ reflects $\hat\sigma_t^{empirical}$'s imprecision — a consequence of the LLN convergence rate over only 21 daily observations.
- Intraday-derived RV (quadratic variation over many high-frequency returns per day) would shrink $\sigma_\eta$, attributing a larger share of $\hat\sigma_\eta$ to genuine model error and sharpening the test's diagnostic power (see extensions.md).

**Is this circular?**
- **Regression**: $\hat\sigma_\eta$ fit once, globally — not tuned per-window, so coverage can't be trivially gamed.
- **Sequential**: no ambiguity — computed only from data prior to each window, genuinely out-of-sample.

## QLIKE Loss
$$\text{QLIKE}(\hat\sigma_t,RV_t) = \frac{RV_t^2}{\hat\sigma_t^2} - \log\left(\frac{RV_t^2}{\hat\sigma_t^2}\right) - 1$$
Lower mean QLIKE = better fit.

**Why QLIKE?**
- **Asymmetric underprediction penalty**: for any proportional miss by factor $k$, underpredicting incurs strictly more loss than overpredicting ($\text{Loss}_{under}(k)-\text{Loss}_{over}(k) = k^2-1/k^2-4\log k > 0$ for all $k>1$) — reflecting that underestimating volatility is typically costlier in practice.
- **Robust to a noisy proxy**: Patton (2011) shows that, under a conditionally unbiased proxy, model comparisons made using historical volatility are guaranteed to translate to the same result if instantaneous volatility could be used instead i.e QLIKE reflects which models would work well on the instantaneous volatility

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

A significant result on any of these three tests indicates systematic, time-varying miscalibration the pooled KS test cannot detect.


