## Procedure Section

This file details the procedure used for the project

There are 3 sets of data that need to be extracted from Yahoo Finance.

2013 - 2017 for developing priors

2018 - 2021 for testing whether models under regression conditions

2022 - 2025 for testing whether models under sequential updating

After that, regression testing and sequential updating is required.

## Data Periods

The project splits SPY data into three periods, each serving a distinct methodological role:

| Period | Date Range | Purpose |
|---|---|---|
| Prior | 2013–2017 | Developing model priors |
| Regression | 2018–2021 | In-sample testing |
| Sequential | 2022–2025 | Out-of-sample testing via sequential updating |

3 models will be tested, a baseline rolling average, a time series model and a stochastic model.

Note that each model will be developed and then tested on both regression and sequential data. At the end of this notebook, the models will be compared to each other

The model's effectiveness is tested using different metrics like QLIKE, Realized Volatility Coverage, PIT Histogram and Return Coverage. 

## Regression Conditions

Priors developed on 2013–2017 data are updated using 2018–2021 data to obtain posterior samples for model parameters.

This is an in-sample test: the fitted parameters are propagated forward through the regression-period data itself to obtain implied volatility paths.

## Sequential Updating

Starting with posterior samples from the regression period, models are sequentially updated over windows spanning 2022–2025.

In each window:
1. Volatility is forecast using the previous window's posterior samples
2. The posterior is then updated by treating the previous window's posterior as the new prior (via fitting posterior samples to the original prior distributions), conditioned on the current window's data

## Baseline: Rolling Average

The rolling average is used as a naive baseline for comparison against the Bayesian models. It has no parameters and is not itself Bayesian — see theoretical.md for its definition and the implications of this.

Below, the metrics used to evaluate the models are explained.

## Historical Volatility (Benchmark)

Historical volatility — the 21-day rolling standard deviation of log returns — serves as the empirical benchmark against which all three models' volatility estimates are evaluated. See theoretical.md for its definition, derivation, and the assumptions it relies on.

### What is QLIKE?

The QLIKE loss compares pointwise estimates of model volatility against realized volatility.

\begin{equation*}
    \text{QLIKE}(\hat{\sigma}_t, RV_t) = \frac{RV_t^2}{\hat{\sigma}_t^2} 
    - \log\!\left(\frac{RV_t^2}{\hat{\sigma}_t^2}\right) - 1
\end{equation*}

### Realized Volatility Coverage

$\hat\sigma_t^{empirical}$ (derived from actual realized volatility) is compared to $\hat\sigma_t^{model}$ (the model's estimate). The noise model is log-normal:

$$\hat\sigma_t^{empirical} = \sigma_t^{true} \cdot \exp(\eta_t), \quad \eta_t \sim N(0, \sigma_\eta^2)$$

The noise estimate is computed as:

$$\hat\sigma_\eta = \text{std}\left(\log \hat\sigma_t^{empirical} - \log \hat\sigma_t^{model}\right)$$

Realized volatility coverage is the percentage of actual realized volatility estimates falling within the model's Bayesian credible intervals.

### Return Calibration Test (PIT)

Actual returns are compared to the model's estimated return distribution via the Probability Integral Transform (Diebold et al., 1997). The fraction of the model's estimated return distribution lying below the actual return at each $t$ should be uniformly distributed across the period. Uniformity is tested via the Kolmogorov-Smirnov test (Massey, 1951); $p < 0.05$ indicates poor calibration.

Return coverage can also be computed across different credible interval widths, analogous to RV coverage.