## Procedure Section

This file details the procedure used for the project

The project splits SPY data into three periods, each serving a distinct methodological role:

| Period | Date Range | Purpose |
|---|---|---|
| Prior | 2013–2017 | Developing model priors |
| Regression | 2018–2021 | In-sample testing |
| Sequential | 2022–2025 | Out-of-sample testing via sequential updating |

3 models will be tested, a baseline rolling average, a time series model and a stochastic model.

Note that each model will be developed and then tested on both regression and sequential data. At the end of this notebook, the models will be compared to each other.

The model's effectiveness is tested using different metrics that are listed below:
Bullet point lists of just the label of the metrics

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

## Regression Conditions

Priors developed on 2013–2017 data are updated using 2018–2021 data to obtain posterior samples for model parameters.

This is an in-sample test: the fitted parameters are propagated forward through the regression-period data itself to obtain implied volatility paths.

**COVID breakdown**: since the regression period (2018–2021) contains the COVID-19 volatility shock, all metrics are additionally reported split into pre-COVID, COVID, and post-COVID sub-periods (2020-01-30 to 2020-04-30), in addition to their pooled values. This checks how quickly and how well each model adapted to a genuine, rapid regime shift — a pooled, whole-period average can mask sub-period miscalibration (see theoretical.md).

## Sequential Updating

Starting with posterior samples from the regression period, models are sequentially updated over windows spanning 2022–2025.

In each window:
1. Volatility is forecast using the previous window's posterior samples
2. The posterior is then updated by treating the previous window's posterior as the new prior (via fitting posterior samples to the original prior distributions), conditioned on the current window's data

**Window breakdown**: since the sequential period contains no single comparable crisis event, results are instead broken down into 8 fixed six-month windows spanning 2022–2025, in addition to pooled values — the same rationale as the COVID breakdown above, applied as a general-purpose decomposition (see theoretical.md).

## Baseline: Rolling Average

The rolling average is used as a naive baseline for comparison against the Bayesian models. It has no parameters and is not itself Bayesian — see theoretical.md for its definition and the implications of this.

## Time Series Model: EGARCH(1,1)-t

A Bayesian EGARCH(1,1) model with Student-t innovations, capturing the leverage effect and fat-tailed returns. See theoretical.md for the full parameterization and justification.

This model was chosen after AIC/BIC comparison with other time series models (the prior data was fit to the model)

Then the priors were developed using prior predictive checking and information obtained from fitting the prior data to the model.

Then the posterior samples are obtained via NUTS/MCMC.
