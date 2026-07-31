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

The model's effectiveness is tested using metrics covering historical volatility comparison, return distribution calibration, and serial dependence — see theoretical.md (Metrics Overview) and metrics.md for details.

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

## EGARCH: Regression

1. Sample parameters from their priors (see Priors section).
2. Propagate the EGARCH recursion forward through the regression period's *actual observed* returns, generating the implied $\sigma_t$ path for that specific parameter draw.
3. Use this $\sigma_t$ path to evaluate the likelihood of the full regression-period returns under a Student-t observation model; NUTS uses this (and its gradient) to inform the next proposal.
4. Repeat across many draws to build up the posterior.

Since EGARCH volatility is estimated pointwise (one $\sigma_t$ per day per posterior draw), it is aggregated (see theoretical.md) before comparison with historical volatility.

## EGARCH: Sequential

The volatility state is carried forward from the end of the regression period (per-draw, not a point estimate).

Each window involves two processes:

1. **Forecast generation**: the recursion is propagated forward from the incoming posterior using *simulated* shocks (both a simulated return and the resulting $\sigma_t$ are generated together at each step, though only $\sigma_t$ is used further). This forecast feeds every evaluation metric (QLIKE, HV coverage, PIT, serial-dependence diagnostics).
2. **Refitting**: the window is then refit on its actual returns — mechanically similar to the regression fit, but using priors derived from the previous window's posterior (fit to a parametric family: Beta for $\beta$, Gamma for $\nu_{shift}$, Normal/TruncatedNormal for the rest). NUTS produces an updated posterior; this step has no evaluation role of its own.

The updated posterior and its end-of-window state seed the next window's forecast, and the process repeats. Both the volatility estimate and the noise correction ($\hat\sigma_\eta$) are computed with the same expanding, out-of-sample discipline — see theoretical.md.


