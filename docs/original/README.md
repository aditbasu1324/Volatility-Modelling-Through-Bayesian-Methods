# Volatility-Modelling-Through-Bayesian-Methods

This is a university project assigned under Bayesian Machine Learning Module.

The goal of the project was to model volatility using Bayesian Methods.

This section contains the original work submitted, but this project will be improved upon and the errors found will be listed below. 

Conceptual Error:

Realized volatility (needed to account for difference in computing this for baseline as opposed to standard model (used RMS everywhere))

Historical volatility is more accurate than realized volatility in this context since its daily returns (plan was initially to do intraday returns so the name stuck)

Incorrectly used 1/h-1 for the model side instead of 1/h

Adding metrics to the baseline that were not thought of initially. 

Change the structure of the code to include COVID stuff.

Consider the metrics in more detail and improve the theoretical justification of the models