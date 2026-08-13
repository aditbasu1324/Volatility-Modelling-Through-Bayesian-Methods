# General Overview

First, the outputs from 2022-2025 regression and sequential are compared to identify how well models perform and whether or not they can generalize effectively to a sequential setting.

Then later on, the 2018-2021 period containing COVID and the 2022-2025 regression periods will be compared to assess how well the models respond to a period containing a sudden shock.

## General Results

In terms of historical volatility metrics, SV>EGARCH>Baseline in both the regression and sequential setting. Each model has better HV coverage, lower QLIKE mean and a lower noise term than the model worse than it.

With regards to volatility properties, SV is the only model that passes volatility clustering in both settings while EGARCH is the only model that captures the leverage effect. All models pass the shape and directional persistence test. Although all models pass shape test, EGARCH does significantly better on this test than SV or the baseline.

On the COVID test, EGARCH adapts best to sudden shocks i.e the COVID spike. However, over the entirety of the 2018-2021 period it is still generally worse than SV including the Post-COVID coverage after the shocks.

## Sequential vs Regression Overview

### Notes

Note that for baseline, the regression and sequential forecasts are identical except for the measurement noise.

Additionally, Window 7 is the most volatile period here so the model's performance on this window is highlighted separately.

### Results

Historical volatility metrics don't generalize as well to a sequential setting which can be seen through the 4x, 5x increase in QLIKE mean for EGARCH and SV respectively. Additionally, this gets worse in more volatile periods as seen from the 6x increase in QLIKE mean in Window 7.

For all the volatility tests, models that pass or fail in the regression setting continue to pass or fail in the sequential setting i.e the volatility tests do generalize. However, the models do get worse at the volatility property tests (score lower p values) in the sequential setting with a notable exception being SV at the shape test.

The evidence for these results along with associated summaries are given in the tables below.

### Tables: 2022-2025, Regression vs. Sequential

## How to Read the Results Below

- **QLIKE**: lower is better; no fixed threshold, primarily useful for ranking models against each other.
- **HV Coverage**: for a stated CI (e.g. 90%), the target is that exact percentage — below target means overconfident (too-narrow) intervals; above target means underconfident (too-wide) intervals.
- **Interval width**: given similar coverage, narrower is more informative; a wide interval can trivially achieve "good" coverage while carrying little real information.
- **Measurement noise ($\hat\sigma_\eta$) progression**: growing over the sequential period indicates the gap between forecast and empirical benchmark is widening, not narrowing.
- **Forecast-actual correlation**: higher indicates the forecast genuinely tracks real movements; near-zero indicates little to no relationship.
- **PIT/KS p-value**: p>0.05 indicates no evidence of miscalibration (a "pass"); the KS statistic itself indicates the size of any deviation, useful for comparing severity even when p-values are hard to compare across differing sample sizes.
- **Ljung-Box / Engle-Ng p-values**: p>0.05 indicates that the requisite volatility fact has already been captured by the models since the residuals have no evidence of the tested patterns (no directional persistence, no clustering, no leverage-driven bias); the coefficient/statistic size indicates how severe a failure is, not just whether it's significant.
- **Metrics with a fixed ideal value** (correlation: ideal=1; coverage: ideal=stated CI level; PIT/Ljung-Box/Engle-Ng p-values: no fixed "ideal" number, but a clear pass/fail threshold at 0.05) are read by their distance from that fixed point, not by ratio-to-baseline.
- **Metrics with no fixed ideal** (QLIKE: no natural zero or target value, only meaningful in comparison) are read as ratios relative to baseline.

#### 1. QLIKE

##### QLIKE — 2022-2025, Pooled (Regression vs. Sequential)

| | Baseline | EGARCH | SV |
|---|---|---|---|
| Regression | 0.3955 | 0.0726 | 0.0551 |
| Sequential | 0.3955 | 0.2944 | 0.2703 |

##### QLIKE (Regression) — By Window

| Window | Baseline | EGARCH | SV |
|---|---|---|---|
| Window 1 | 0.2578 | 0.0108 | 0.0467 |
| Window 2 | 0.1093 | 0.0514 | 0.0308 |
| Window 3 | 0.0842 | 0.0470 | 0.0223 |
| Window 4 | 0.1296 | 0.0916 | 0.0247 |
| Window 5 | 0.1702 | 0.0862 | 0.0430 |
| Window 6 | 0.9491 | 0.0600 | 0.0630 |
| Window 7 | 1.0283 | 0.1299 | 0.1519 |
| Window 8 | 0.4464 | 0.1112 | 0.0614 |

##### QLIKE (Sequential, t,t+1 filtered) — By Window

| Window | Baseline | EGARCH | SV |
|---|---|---|---|
| Window 1 | 0.2578 | 0.1516 | 0.1822 |
| Window 2 | 0.1093 | 0.0779 | 0.0914 |
| Window 3 | 0.0842 | 0.3591 | 0.1459 |
| Window 4 | 0.1296 | 0.2637 | 0.1305 |
| Window 5 | 0.1702 | 0.2186 | 0.1422 |
| Window 6 | 0.9491 | 0.2403 | 0.2979 |
| Window 7 | 1.0283 | 0.7744 | 0.9344 |
| Window 8 | 0.4464 | 0.2834 | 0.2519 |

##### Sequential Forecast-Actual Correlation (t,t+1 filtered)

| | Baseline | EGARCH | SV |
|---|---|---|---|
| Correlation | 0.4946 | 0.5409 | 0.5766 |

EGARCH, SV QLIKE Mean increases by 4x, 5x respectively i.e models don't generalize. 

SV outperforms EGARCH in both the regression and sequential setting.
Both outperform baseline even in the sequential setting (EGARCH has 0.75x QLIKE mean of baseline, SV has 0.7x QLIKE mean of baseline).

With regards to the window 7, QLIKE mean increases by 6x compared to the regression setting i.e the error worsens in more volatile periods.

The baseline is substantially worse than EGARCH, SV even in the sequential setting, but it does perform better than the EGARCH, SV for some windows (see window 3,4 in sequential table)

These results are further supported by the correlation table above.

#### 2. HV Coverage

##### HV Coverage (90% CI) — 2022-2025, Pooled (Regression vs. Sequential)

| | Baseline | EGARCH | SV |
|---|---|---|---|
| Regression | 91.23% | 82.26% | 90.93% |
| Sequential | 93.68% | 93.37% | 93.37% |

![2022-2025 Regression Forecast Comparison: Baseline vs. EGARCH vs. SV](results/comparison_hv_2022_regression.png)

![Sequential Forecast Comparison: Baseline vs. EGARCH vs. SV](results/comparison_hv_sequential.png)

##### HV Coverage (90%, Regression) — By Window

| Window | Baseline | EGARCH | SV |
|---|---|---|---|
| Window 1 | 96.77% | 100.00% | 95.16% |
| Window 2 | 100.00% | 94.49% | 99.21% |
| Window 3 | 98.39% | 95.16% | 98.39% |
| Window 4 | 96.03% | 70.63% | 93.65% |
| Window 5 | 100.00% | 80.65% | 91.13% |
| Window 6 | 80.47% | 83.59% | 92.19% |
| Window 7 | 65.57% | 67.21% | 69.67% |
| Window 8 | 92.45% | 63.21% | 86.79% |

##### HV Coverage (90%, Sequential) — By Window

| Window | Baseline | EGARCH | SV |
|---|---|---|---|
| Window 1 | 100.00% | 92.74% | 100.00% |
| Window 2 | 100.00% | 100.00% | 100.00% |
| Window 3 | 100.00% | 84.68% | 94.35% |
| Window 4 | 98.41% | 96.83% | 96.03% |
| Window 5 | 100.00% | 97.58% | 90.32% |
| Window 6 | 86.72% | 100.00% | 84.38% |
| Window 7 | 69.67% | 81.15% | 82.79% |
| Window 8 | 94.34% | 93.40% | 100.00% |

##### HV Coverage — 2022-2025, Regression — By CI Level

| CI Level | Baseline | EGARCH | SV |
|---|---|---|---|
| 50% | 53.41% | 43.43% | 52.29% |
| 60% | 62.69% | 52.60% | 62.79% |
| 70% | 73.80% | 60.55% | 73.09% |
| 80% | 83.89% | 70.03% | 82.06% |
| 85% | 87.46% | 75.43% | 86.34% |
| 90% | 91.23% | 82.16% | 91.03% |
| 95% | 93.27% | 91.95% | 95.11% |
| 99% | 97.66% | 97.35% | 98.47% |

##### HV Coverage — 2022-2025, Sequential — By CI Level

| CI Level | Baseline | EGARCH | SV |
|---|---|---|---|
| 50% | 64.93% | 49.24% | 50.97% |
| 60% | 77.37% | 60.86% | 57.70% |
| 70% | 85.73% | 73.09% | 69.11% |
| 80% | 91.74% | 82.36% | 80.94% |
| 85% | 92.46% | 87.67% | 88.79% |
| 90% | 93.68% | 93.17% | 93.07% |
| 95% | 96.94% | 96.94% | 95.41% |
| 99% | 99.69% | 98.78% | 99.29% |

##### Measurement Noise (sigma_eta) — Sequential (2022-2025), First vs. Final Window

| | Baseline | EGARCH | SV |
|---|---|---|---|
| Window 1 | 0.5822 | 0.2042 | 0.1782 |
| Final Window | 0.5031 | 0.3443 | 0.2934 |

During the regression period EGARCH undercovers throughout. The SV and baseline both overcover early on while undercovering later on, but the baseline model is consistently worse throughout the coverage by a significant margin. 

Relative to the baseline, the measurement noise is 0.35x of EGARCH and 0.3x of SV in regression so the predictions from the SV are also the most informative.

In the sequential period, the distinction isn't as clear between SV and EGARCH while the baseline overcovers everything. Generally EGARCH and SV overcover for higher percentage intervals and they perform similarly in sequential setting.

In sequential, the measurement noise for all models rise. Interestingly, the measurement noise falls from window 1 to final window for baseline while the noise rises for SV, EGARCH.

For Window 7, the models all equally undercover the model in regression, covering around 70% for the 90% confidence interval. The models have higher coverage during sequential especially EGARCH, SV although this could be due to the larger noise estimate.

#### 3. PIT/KS

##### PIT/KS — 2022-2025, Regression

| | Baseline | EGARCH | SV |
|---|---|---|---|
| KS statistic | 0.0415 | 0.0266 | 0.0380 |
| p-value | 0.0619 | 0.4699 | 0.1087 |

![PIT Histograms — 2022-2025, Regression](results/pit_histograms_2022_regression.png)

##### PIT/KS — 2022-2025, Sequential

| | Baseline | EGARCH | SV |
|---|---|---|---|
| KS statistic | 0.0415 | 0.0338 | 0.0367 |
| p-value | 0.0619 | 0.1974 | 0.1321 |

![PIT Histograms — Sequential](results/pit_histograms_sequential.png)

In the regression period, all 3 models pass the test with EGARCH, SV, baseline being the order of best to worst.

The above trend holds in sequential, though not as strongly. In particular, the EGARCH value for the p-value more than halfs. Surprisingly, the p value is higher for SV in sequential setting compared to regression, which means refitting the model is actually improving SV's ability to capture the shape despite not having future knowledge. 

#### 4. Serial Dependence

##### Ljung-Box (lag=5) — 2022-2025, Regression

| | Baseline | EGARCH | SV |
|---|---|---|---|
| p-value, $z_t$ (level) | 0.9313 | 0.7035 | 0.7124 |
| p-value, $z_t^2$ (squared) | 0.0113 | 0.0210 | 0.1016 |

##### Ljung-Box (lag=5) — 2022-2025, Sequential

| | Baseline | EGARCH | SV |
|---|---|---|---|
| p-value, $z_t$ (level) | 0.9313 | 0.7899 | 0.7187 |
| p-value, $z_t^2$ (squared) | 0.0113 | 0.0095 | 0.0540 |

##### Engle-Ng Sign-Bias — 2022-2025, Regression

| | Baseline | EGARCH | SV |
|---|---|---|---|
| $\beta_1$ (sign_lag coef) | -0.1388 | -0.0435 | -0.1080 |
| p-value | 0.0056 | 0.3170 | 0.0030 |

##### Engle-Ng Sign-Bias — 2022-2025, Sequential

| | Baseline | EGARCH | SV |
|---|---|---|---|
| $\beta_1$ (sign_lag coef) | -0.1388 | -0.0400 | -0.2068 |
| p-value | 0.0056 | 0.3555 | 0.0082 |

All models pass the directional persistence test.

Only SV passes the volatility clustering test in both regression and sequential settings. This is unexpected since EGARCH is expected to capture these volatility properties. 

EGARCH passes the leverage effect test in both settings while the other models fail by a substantial margin (neither is every above p<0.01 in either setting).

## Stress Test Overview-COVID

### Notes

Here, the two regression periods 2018-2021 (COVID period) and the 2022-2025 (standard period) are compared to each other. 

In particular, there is COVID spike (30th Jan-30th April 2020). The COVID coverage is also separated into three regimes a pre-COVID, COVID spike, and post-COVID for further analysis

### Results

Based on historical volatility metrics, EGARCH reacts best to the COVID spike although it is still worse than SV with regards to the overall COVID period. 

With regards to volatility facts, EGARCH passes volatility clustering in the COVID period whereas in the standard period, only SV passsed volatility clustering. Additionally, all 3 models fail the leverage test in this period whereas all 3 passed in the standard period though EGARCH is still better at incorporating the leverage effect.

Additionally, the COVID breakdown demonstrates that models are better at capturing volatility facts under individual regimes as opposed to the entire period For instance, SV fails on the volatility clustering test but passes this test for every individual regime.

The evidence for these results and a full breakdown of each section's results are given in tables below that are also summarized. 

### Tables: 2018-2021 vs. 2022-2025 (COVID Stress Test)

#### 1. QLIKE

##### QLIKE (mean) — Regression, 2018-2021 vs. 2022-2025

| | Baseline | EGARCH | SV |
|---|---|---|---|
| 2018-2021 | 1.2659 | 0.0992 | 0.0702 |
| 2022-2025 | 0.3955 | 0.0726 | 0.0551 |

##### QLIKE — Regression, COVID Breakdown (2018-2021)

| | Baseline | EGARCH | SV |
|---|---|---|---|
| Pre-COVID | 1.2380 | 0.1112 | 0.0687 |
| COVID | 6.3625 | 0.1622 | 0.1974 |
| Post-COVID | 0.5425 | 0.0739 | 0.0525 |

Overall, the QLIKE mean is larger for the overall 2018-2021 period particularly for baseline across all models. 

During the COVID spike, the QLIKE mean for all 3 models rises (5x for baseline, 1.5x for EGARCH, 3x for SV compared to Pre-COVID). EGARCH has the lowest QLIKE mean during this spike, whereas SV has the lowest QLIKE mean in pre-COVID, post-COVID and overall.

#### 2. HV Coverage

##### HV Coverage (90% CI) — Regression, 2018-2021 vs. 2022-2025

| | Baseline | EGARCH | SV |
|---|---|---|---|
| 2018-2021 | 89.78% | 85.43% | 90.89% |
| 2022-2025 | 93.68% | 82.26% | 90.93% |

![Regression Forecast Comparison: Baseline vs. EGARCH vs. SV](results/comparison_hv_regression.png)

##### HV Coverage (90% CI) — Regression, COVID Breakdown (2018-2021)

| | Baseline | EGARCH | SV |
|---|---|---|---|
| Pre-COVID | 88.70% | 80.65% | 90.80% |
| COVID | 52.38% | 87.30% | 66.67% |
| Post-COVID | 96.68% | 91.29% | 94.78% |

##### HV Coverage — Regression (2018-2021) — By CI Level

| CI Level | Baseline | EGARCH | SV |
|---|---|---|---|
| 50% | 52.28% | 43.83% | 50.91% |
| 60% | 61.41% | 52.33% | 61.74% |
| 70% | 70.54% | 62.45% | 72.98% |
| 80% | 80.46% | 73.89% | 83.10% |
| 85% | 84.42% | 80.87% | 86.74% |
| 90% | 89.78% | 85.43% | 90.89% |
| 95% | 95.34% | 92.31% | 94.74% |
| 99% | 99.11% | 97.17% | 98.68% |

##### Mean 90% CI Width — Regression (2018-2021)

| | Baseline | EGARCH | SV |
|---|---|---|---|
| Mean Width | 0.0230 | 0.0076 | 0.0059 |

##### Measurement Noise (sigma_eta) — Regression, 2018-2021 vs. 2022-2025

| | Baseline | EGARCH | SV |
|---|---|---|---|
| 2018-2021 | 0.5822 | 0.2042 | 0.1782 |
| 2022-2025 | 0.3921 | 0.1702 | 0.1619 |

Overall, the coverage is better during the 2018-2021 period at the 90% confidence interval. 

During the COVID spike, EGARCH is the only model with good coverage.
EGARCH also provides best coverage Post-COVID but massively undercovers Pre-COVID. 

SV is best at Pre-COVID and is consistently better than the baseline across all sections.

The predictions in 2022-2025 are more valuable because the measurement noise is lower for all the models.

#### 3. PIT/KS

##### PIT/KS — Regression, 2018-2021

| | Baseline | EGARCH | SV |
|---|---|---|---|
| KS statistic | 0.0481 | 0.0276 | 0.0472 |
| p-value | 0.0184 | 0.4195 | 0.0219 |

![PIT Histograms — Regression](results/pit_histograms_regression.png)

##### PIT/KS Statistic — Regression, COVID Breakdown (2018-2021)

| | Baseline | EGARCH | SV |
|---|---|---|---|
| Pre-COVID | 0.0479 | 0.0358 | 0.0551 |
| COVID | 0.1014 | 0.1526 | 0.1320 |
| Post-COVID | 0.0698 | 0.0596 | 0.0515 |

##### PIT/KS p-value — Regression, COVID Breakdown (2018-2021)

| | Baseline | EGARCH | SV |
|---|---|---|---|
| Pre-COVID | 0.1766 | 0.5047 | 0.0806 |
| COVID | 0.4939 | 0.0912 | 0.1962 |
| Post-COVID | 0.0312 | 0.0955 | 0.2059 |

##### PIT/KS — 2022-2025, Regression

| | Baseline | EGARCH | SV |
|---|---|---|---|
| KS statistic | 0.0415 | 0.0266 | 0.0380 |
| p-value | 0.0619 | 0.4699 | 0.1087 |

Only EGARCH passes on COVID period, whereas every model passed on 2022-2025 data i.e only EGARCH can capture the shape of shocks of the magnitude of COVID

Considering the subperiod COVID breakdown,

All models pass pre-COVID period and COVID spike i.e they can model the shape of these individual regimes well. Surprisingly the baseline is best and EGARCH is worst on the COVID spike contrary to expectations.

In the post-COVID section, EGARCH and SV models both pass.

#### 4. Serial Dependence

##### Ljung-Box (lag=5) — Regression, 2018-2021

| | Baseline | EGARCH | SV |
|---|---|---|---|
| p-value, $z_t$ (level) | 0.9176 | 0.8144 | 0.7092 |
| p-value, $z_t^2$ (squared) | 0.0000 | 0.0682 | 0.0439 |

##### Ljung-Box (lag=5) — Regression, Pre-COVID

| | Baseline | EGARCH | SV |
|---|---|---|---|
| p-value, $z_t$ (level) | 0.8098 | 0.8864 | 0.9377 |
| p-value, $z_t^2$ (squared) | 0.0000 | 0.1757 | 0.3206 |

##### Ljung-Box (lag=5) — Regression, COVID

| | Baseline | EGARCH | SV |
|---|---|---|---|
| p-value, $z_t$ (level) | 0.3314 | 0.1992 | 0.2197 |
| p-value, $z_t^2$ (squared) | 0.0000 | 0.4605 | 0.1241 |

##### Ljung-Box (lag=5) — Regression, Post-COVID

| | Baseline | EGARCH | SV |
|---|---|---|---|
| p-value, $z_t$ (level) | 0.0884 | 0.3889 | 0.2651 |
| p-value, $z_t^2$ (squared) | 0.0000 | 0.2578 | 0.1625 |

##### Ljung-Box (lag=5) — 2022-2025, Regression

| | Baseline | EGARCH | SV |
|---|---|---|---|
| p-value, $z_t$ (level) | 0.9313 | 0.7035 | 0.7124 |
| p-value, $z_t^2$ (squared) | 0.0113 | 0.0210 | 0.1016 |

##### Engle-Ng Sign-Bias — Regression, 2018-2021

| | Baseline | EGARCH | SV |
|---|---|---|---|
| $\beta_1$ (sign_lag coef) | -0.2689 | -0.0940 | -0.1353 |
| p-value | 0.0000 | 0.0457 | 0.0000 |

##### Engle-Ng Sign-Bias — Regression, Pre-COVID

| | Baseline | EGARCH | SV |
|---|---|---|---|
| $\beta_1$ (sign_lag coef) | -0.2932 | -0.1054 | -0.1442 |
| p-value | 0.0007 | 0.0671 | 0.0009 |

##### Engle-Ng Sign-Bias — Regression, COVID

| | Baseline | EGARCH | SV |
|---|---|---|---|
| $\beta_1$ (sign_lag coef) | -0.5797 | -0.3624 | -0.3691 |
| p-value | 0.0497 | 0.0902 | 0.0162 |

##### Engle-Ng Sign-Bias — Regression, Post-COVID

| | Baseline | EGARCH | SV |
|---|---|---|---|
| $\beta_1$ (sign_lag coef) | -0.1769 | -0.0254 | -0.0789 |
| p-value | 0.0311 | 0.7480 | 0.1037 |

##### Engle-Ng Sign-Bias — 2022-2025, Regression

| | Baseline | EGARCH | SV |
|---|---|---|---|
| $\beta_1$ (sign_lag coef) | -0.1388 | -0.0435 | -0.1080 |
| p-value | 0.0056 | 0.3170 | 0.0030 |

Directional persistence passes for all models in both periods.

Volatility clustering test only passed for EGARCH in 2018-2021, but then only passes for SV not EGARCH in 2022-2025. 

Leverage effect fails for all 3 in 2018-2021 and only passes for EGARCH in 2022-2025 period. Even still the p value in the 2018-2021 section is significantly higher than both SV and baseline (p=0.0457 for EGARCH compared to p=0.0000 for both baseline and SV).

Considering the subperiod breakdown,

SV passes volatility clustering in all individual periods and EGARCH passes all the leverage tests despite failing their respective overall tests. 

All models pass directional persistence in all periods. SV and EGARCH pass volatility clustering in all periods. EGARCH passes leverage test in all periods while SV only passes this test in post-COVID coverage.



