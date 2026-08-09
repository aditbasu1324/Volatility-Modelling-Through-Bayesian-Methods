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

**COVID breakdown**: since the regression period (2018–2021) contains the COVID-19 volatility shock (2020-01-30 to 2020-04-30), all four metric families — historical-volatility comparison (QLIKE, RV coverage), return calibration (PIT/KS), and serial dependence (Ljung-Box, Engle-Ng) — are additionally reported split into pre-COVID, COVID, and post-COVID sub-periods, alongside their pooled values.

This checks how quickly and how well each model adapted to a genuine, rapid regime shift, since a pooled, whole-period average can mask sub-period miscalibration (see Why Break Down by Sub-Period?, theoretical.md).

A related but distinct concern applies to PIT/KS specifically — pooled uniformity testing doesn't check whether consecutive values are independent over time, addressed separately via the Serial Dependence Diagnostics (theoretical.md/metrics.md).

## Sequential Updating

Starting with posterior samples from the regression period, models are sequentially updated over windows spanning 2022–2025.

In each window:
1. Volatility is forecast using the previous window's posterior samples
2. The posterior is then updated by treating the previous window's posterior as the new prior (via fitting posterior samples to the original prior distributions), conditioned on the current window's data

**Window breakdown**: since the sequential period contains no single comparable crisis event, results are instead broken down into 8 fixed six-month windows spanning 2022–2025, in addition to pooled values — the same rationale as the COVID breakdown above, applied as a general-purpose decomposition (see theoretical.md).

## Regression, Matched Period (2022–2025)

A second, one-shot regression fit, using the same mechanism as Regression Conditions above but fit on the 2022–2025 span instead of 2018–2021 — carried forward from the 2018–2021 regression's posterior exactly as each Sequential window already is (same two channels: the prior over $\theta$, and the state seed).

This exists to separate two effects that would otherwise be confounded: comparing it against Sequential (same 2022–2025 period, refit every 6 months) isolates the effect of refit frequency; comparing it against the original 2018–2021 Regression (same one-shot method, different period) isolates the effect of the COVID shock being in-sample or not.

## Baseline: Rolling Average

The rolling average is used as a naive baseline for comparison against the Bayesian models. It has no parameters and is not itself Bayesian — see theoretical.md for its definition and the implications of this.

Applied identically across regression and sequential settings: the rolling-window formula is computed directly from observed returns (no fitting step), then extended into a constant forecast across the evaluation window (the forecast rule — see theoretical.md). All metrics (QLIKE, HV Coverage, PIT/KS, ACF/Engle-Ng) are computed as described in theoretical.md ("Applying the Metrics to the Baseline").

- **Regression**: metrics reported pooled and split into pre-COVID/COVID/post-COVID sub-periods.
- **Sequential**: metrics reported pooled and split by the 8 six-month windows; $\hat\sigma_\eta$ is re-estimated as an expanding, out-of-sample quantity per window (see Expanding-Window Noise Estimation, theoretical.md).

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

The volatility state is carried forward from the end of the regression period (per-draw, not a point estimate) — see Historical Volatility (metrics.md) for how this state is obtained.

Each window involves two forecasts plus one refit:

1. **Blind forecast (t+126)**: the recursion is propagated forward from the window-start state using *simulated* shocks for the entire window in one pass (both a simulated return and the resulting $\sigma_t$ are generated together at each step, though only $\sigma_t$ is used further) — the state is never updated within the window, so it grows less informed by reality the further into the window it runs.
2. **Filtered forecast (t,t+1)**: each day's state is instead obtained by causally filtering the recursion through that day's real return before forecasting forward, so $\sigma_{t+1}$ is genuinely $\mathcal F_t$-measurable (see theoretical.md, Why Sequential Updating?). The resulting $h$-day-ahead HV forecast is still generated by blind forward simulation from that state, since future returns remain unknown regardless of how accurately $\sigma_{t+1}$ is filtered. This is the forecast used for the models' primary reported results; the blind forecast is retained alongside it for comparison.
3. **Refitting**: the window is then refit on its actual returns — mechanically similar to the regression fit, with two channels carrying information forward from the previous window's posterior: (a) the prior over $\theta$, fit to a parametric family from the previous window's posterior samples (Beta for $\beta$, Gamma for $\nu_{shift}$, Normal/TruncatedNormal for the rest), and (b) the state the refit's own recursion starts from. NUTS produces an updated posterior; this step has no evaluation role of its own.

The updated posterior and its end-of-window state seed the next window's forecasts (both styles), and the process repeats. Both the volatility estimate and the noise correction ($\hat\sigma_\eta$) are computed with the same expanding, out-of-sample discipline.

## Stochastic Volatility Model

Priors for SV are centered using a preliminary linearized (Kalman filter) fit on prior-period data, analogous to EGARCH's MLE-based centering — see theoretical.md for why this approach is used, implementation.md for the code.

A Standard Stochastic Volatility model, fit via Particle MCMC (PMCMC) — see theoretical.md/sampling.md for the model definition and why PMCMC is required, and implementation.md for how this is coded.

Follows the same regression/sequential structure as EGARCH (see Regression Conditions / Sequential Updating): priors developed on 2013-2017 data, fit via PMCMC on regression-period data, then sequentially updated across the 2022-2025 windows.

## SV: Sequential

Same two-forecasts-plus-refit structure as EGARCH: Sequential (above), but the filtering step works differently since SV's state carries its own independent noise and can't be recovered from returns alone (see theoretical.md, Why Particle MCMC (PMCMC) Is Required).

1. **Blind forecast (t+126)**: as EGARCH — the recursion is propagated forward from the window-start state using simulated shocks for the entire window in one pass.
2. **Filtered forecast (t,t+1)**: each day's state is instead obtained via a genuine particle filter — propagate/reweight/resample through that day's real return — capturing the state *before* the reweight step folds that return in, so it stays $\mathcal F_t$-measurable rather than already conditioned on $r_t$. The $h$-day-ahead HV forecast is still blind-simulated forward from that state, for the same reason as EGARCH.
3. **Refitting**: same two carry-forward channels as EGARCH — a prior over $\theta$ fit from the previous window's posterior, and a state seed. For SV, this state seed also warm-starts the refit's own particle filter, so every proposed $\theta$'s likelihood is evaluated starting from the true carried-forward state rather than that proposal's own stationary distribution.