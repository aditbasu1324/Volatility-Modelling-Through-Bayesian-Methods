# Extensions

This document tracks what has changed since the original coursework submission (see [`docs/original/`](original/)), and what further extensions/standard practices remain unexplored.

## Standard Practitioner Techniques Not Currently Used

The following are established methods in the volatility forecasting/evaluation literature that this project does not currently implement. Listed here as acknowledged scope boundaries and candidate future work, not as flaws in the current methodology.

### Forecast Validation

- **Mincer-Zarnowitz regression**: regress realized volatility on the model's forecast, $RV_t = a + b\cdot\hat\sigma_t^{model} + \epsilon_t$. A well-calibrated forecast should give $a=0, b=1$. A simple, standard regression-based calibration check, complementary to the QLIKE/coverage/PIT battery already implemented.
- **Out-of-sample $R^2$**: compares forecast MSE against a naive benchmark (e.g. historical mean, random walk) — a standard "does this model add value at all" check, distinct from the current metrics.
- **Model Confidence Set** (Hansen, Lunde, Nason): a formal procedure for identifying the *set* of models statistically indistinguishable from the best, given multiple candidates — more principled than running several separate pairwise DM tests across the three models currently compared.

### Forecast Combination

- **Forecast averaging/ensembling**: combining forecasts from multiple models (simple average, or a fitted combination weight) frequently outperforms any single constituent model (Bates & Granger, 1969, onward) — not explored here, despite having three ready-made candidate models.

### Options-Market-Based Approaches

- **Implied volatility as a predictor/benchmark**: options markets embed a forward-looking volatility estimate via Black-Scholes inversion. Practitioners commonly compare statistical forecasts against implied volatility, or use it as an additional model input. The project's own README motivation section mentions mispriced options, but no part of the current methodology touches implied volatility.

### Cross-Validation Design

- **Expanding window vs. rolling/sequential window**: this project uses a sequential-update approach that carries forward posteriors window-to-window. An alternative is an *expanding* window (retaining all historical data, never discarding old windows) — a genuine design tradeoff (more data vs. slower adaptation to regime changes) not explored here.

### Portfolio/Trading Applications

- **Volatility targeting**: using the forecast to actively scale portfolio exposure to a constant target risk level — this would operationalize the README's "dynamic hedging" motivation into an actual backtestable strategy, connecting to the economic/utility-value evaluation gap noted in theoretical.md.

### Microstructure-Robust Realized Measures

- **Realized kernels, bipower variation, subsampling/averaging estimators**: standard techniques for handling microstructure noise in high-frequency RV construction. Not yet relevant given the current daily-frequency data, but directly applicable once intraday data (see data.md) is incorporated.

Add in EMWA, add in other metrics as well.
## Test Coverage Summary

| Dimension being tested | Covered by | Not covered / gap |
|---|---|---|
| Window-average magnitude | QLIKE, HV comparison | — |
| Window-average uncertainty honesty | RV coverage | — |
| Day-level distributional calibration | PIT/KS | Can't separate level vs. shape error |
| Directional error persistence | ACF of $z_t$ | — |
| Volatility-clustering error persistence | ACF of $z_t^2$ | — |
| Leverage/asymmetry response | *(not covered)* | Needs sign($r_{t-1}$) regression |
| Tail-specific calibration | *(not covered)* | Needs Kupiec/Christoffersen |
| Multi-horizon path accuracy | *(not covered)* | Needs term-structure evaluation |
| Economic/trading value | *(not covered)* | Needs backtested strategy |
| Within-window timing accuracy | *(not covered)* | Needs finer-than-21-day benchmark (intraday) |