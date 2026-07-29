### Data

Historical volatility (HV) is inherently retrospective: computing $\hat\sigma_t^{empirical}$ for any day requires the following 21 days of returns, which are never available in real time. This applies uniformly throughout the dataset (including at period and window boundaries) and reflects HV's role purely as a benchmark for post-hoc evaluation, not as a live/real-time forecasting quantity.



Rolling/historical volatility are computed on the full continuous return series, then sliced by period. Values near any boundary (period transitions, or model-internal window transitions) reflect returns from the adjacent period/window, since both are inherently backward/forward-looking by construction — this is a universal property of these estimators, not specific to any one boundary, and does not affect the underlying models' own out-of-sample forecast validity (see theoretical.md).