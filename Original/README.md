# Volatility-Modelling-Through-Bayesian-Methods

This is a university project assigned under Bayesian Machine Learning Module.

The goal of the project was to model volatility using Bayesian Methods.

This section contains the original work submitted, but this project will be improved upon; the improved version is in the Improved Folder.

## Known Issues (Fixed in Improved/)

**1. Historical volatility was computed incorrectly.**
The wrong version of historical volatility was used for the models, including using 1/(h-1) instead of 1/h on the model side. The baseline was also computed inconsistently with the models (RMS used throughout instead of matching the baseline's own convention). "Realized volatility" was the working name for this quantity from an earlier plan to use intraday returns; since the project ended up using daily returns, historical volatility is the more accurate term and the more accurate comparison target. This also required revisiting what the comparison should expect of $F_{t}$, and fixing an alignment error this caused in PMCMC. Corrected in Metrics.md and the code of the improved version.

**2. PIT was computed incorrectly.**
In the sequential setting, PIT used the implied paths rather than the forecasted paths. Corrected in the code of the improved version.

**3. The regression vs. sequential comparison was conflated with the COVID stress test.**
Running that comparison only on the COVID period mixed two separate questions: whether the models generalize (regression vs. sequential) and how they perform under stress (COVID). These are now split into two separate comparisons, with the code restructured accordingly.

**Other implementation/code improvements**
- Reorganized the code into functions for better organization and reproducibility.
- Added the missing metrics (including the serial diagnostic tests) for the baseline, which previously didn't have them computed.
- Considered the metrics in more depth and improved the theoretical justification of the models.
- Fixed a coding error in the MCMC step's use of `prior_std`.
- Improved the sampling code, including a method 1 vs. method 2 discussion and carrying forward 2000 draws instead of 8000.

