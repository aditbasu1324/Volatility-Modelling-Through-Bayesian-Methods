# Interpreting Model Outputs

This file explains how to read the raw numerical outputs this project produces. See metrics.md for what each test *measures* and why, and results.md for what this project's *specific* numbers mean.

## Reading Historical Volatility Comparison Output

- **QLIKE (mean)**: lower is better. There's no fixed "good" threshold in isolation — QLIKE is primarily useful for *ranking* models against each other (via mean comparison, or formally via a Diebold-Mariano test) rather than as an absolute pass/fail number.
- **HV comparison plot (forecast vs. actual line + band)**: if the forecast line and actual HV track closely with no persistent gap, the model is capturing the volatility level well. A persistent gap in one direction (forecast consistently above or below actual) indicates systematic bias, not just noise — worth checking whether it's concentrated in specific sub-periods (see per-window/COVID breakdowns) rather than uniform throughout.
- **RV Coverage (%)**: compare directly against the stated CI level — e.g. a "90% CI" should contain the actual value ~90% of the time. Coverage well below target (e.g. 90% CI achieving only 60%) means intervals are too narrow (model overconfident); coverage well above target means intervals are too wide (model underconfident, less useful for practical risk sizing even though technically "safe").

## Reading a `statsmodels` OLS Summary

Applies to the Engle-Ng sign-bias regression (and any other OLS output in this project):

| Column | Meaning |
|---|---|
| `coef` | Estimated coefficient value |
| `std err` | Standard error of that estimate |
| `t` | $t$-statistic = coef / std err |
| `P>\|t\|` | Two-sided p-value for testing whether this coefficient is zero |
| `[0.025, 0.975]` | 95% confidence interval for the coefficient |
| `R-squared` | Proportion of variance in the dependent variable explained by the regressors |
| `Prob (F-statistic)` | p-value for the joint test that *all* regressors are zero (with one regressor, equivalent to the single coefficient's own $t$-test) |

## p-value Conventions Used Throughout This Project

Every test in this project follows the same convention: a **low p-value (< 0.05)** is evidence *against* the null hypothesis being tested — never proof the null is false, just evidence inconsistent with it. A **high p-value** means the data doesn't provide evidence against the null — never proof the null is true, just an absence of detected evidence otherwise.

| Test | Null hypothesis | Low p-value means |
|---|---|---|
| KS test (PIT) | $\{u_t\}$ is Uniform(0,1) | Model is not well-calibrated |
| Ljung-Box ($z_t$) | No autocorrelation in calibration errors | Directional persistence in errors |
| Ljung-Box ($z_t^2$) | No autocorrelation in squared errors | Volatility-clustering left in errors |
| Engle-Ng sign-bias | $\beta_1=0$ | Errors depend on sign of prior return (leverage miscalibration) |

## MCMC Convergence Diagnostics

These apply identically to any NUTS fit in this project — the regression-period fit, or any of the 8 sequential window refits — each should be checked independently.

### R-hat ($\hat R$)
Compares between-chain variance to within-chain variance across the multiple chains run in parallel. $\hat R \approx 1.0$ indicates the chains have converged to the same distribution (they agree with each other); $\hat R$ noticeably above 1 (commonly flagged above ~1.01–1.05, sometimes 1.1 depending on convention) indicates the chains haven't mixed — different chains are still exploring different regions, meaning the posterior samples shouldn't be trusted yet.

### Effective Sample Size (ESS)
MCMC draws are autocorrelated (each depends on the last), so $N$ raw draws contain less independent information than $N$ truly independent samples would. ESS estimates how many *independent* samples the draws are equivalent to. Low ESS relative to the raw sample count (a common rule of thumb: ESS below ~400, or a small fraction of total draws) means posterior summaries (means, credible intervals) computed from the chain will be noisier than the raw draw count suggests.

### Divergences
A divergence occurs when the leapfrog integrator's discrete steps fail to accurately follow the true continuous trajectory — usually in regions of sharply changing curvature in the posterior. Even a small number of divergences can indicate bias in the resulting posterior samples in the affected region; they shouldn't simply be ignored if numerous.

## A Note on Regression vs. Sequential

These diagnostics apply identically regardless of which fit produced them. A good R-hat in the regression fit says nothing about whether a later sequential window's refit also converged well — each of the 8 sequential refits should be checked independently against the same criteria.