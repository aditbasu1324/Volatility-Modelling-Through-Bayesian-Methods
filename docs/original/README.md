# Volatility-Modelling-Through-Bayesian-Methods

This is a university project assigned under Bayesian Machine Learning Module.

The goal of the project was to model volatility using Bayesian Methods.

This section contains the original work submitted, but this project will be improved upon and the errors found will be listed below. 

Conceptual Error:

Realized volatility (needed to account for difference in computing this for baseline as opposed to standard model (used RMS everywhere))

Historical volatility is more accurate than realized volatility in this context since its daily returns (plan was initially to do intraday returns so the name stuck)

Incorrectly used 1/h-1 for the model side instead of 1/h

Adding metrics to the baseline that were not thought of initially. including the time series metrics

Change the structure of the code to include COVID stuff.

Consider the metrics in more detail and improve the theoretical justification of the models
Consider the prior predictive checks more clearly
Create functions to make the code more organized and more reproducible for the future.

Understand the historical volatility comparison in more detail (the expectation with regards to Ft).
Coding error in MCMC with using prior_std, 
Additionally the PIT for sequential was based on implied paths not the forecasted paths

changed a bunch of the sampling code, including addition of the method 1, method 2 discussion the 2000 as opposed to 8000 values carry forward