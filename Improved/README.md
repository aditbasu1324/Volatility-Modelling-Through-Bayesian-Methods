# Latent Volatility Modelling Through Bayesian Machine Learning

This is an improved version of the original file with some changes and corrections made.

This project aims to estimate and identify the latent volatility trends in the market. 

In particular, credible intervals for latent volatility are generated through Bayesian inference methods using daily S&P 500 market data.

The volatility models used in this project, are developed on priors, tested on regression conditions (in-sample testing) and then tested under sequential updating (out-of-sample-testing).


## Value of Volatility Modelling 

Accurately estimating the latent volatility is particularly useful for market traders. 

In particular, traders can dynamically hedge their trading positions where theoretically the size of their position should be inversely proportion to volatility. 

Additionally, they can potentially switch strategies depending on if the latent volatility passes a certain threshold (or a certain credible interval). 

Furthermore, they can easily identify mispriced options since quoted options generally have an associated implied volatility. 

All of these techniques can be used to generate a profit.

## Project Structure

- **[Theoretical.md](Theoretical.md)** — the theory behind each model, and why the evaluation metrics are constructed the way they are.
- **[Metrics.md](Metrics.md)** — precise definitions of each evaluation metric (QLIKE, HV Coverage, PIT/KS, Serial Dependence) and how to interpret them.
- **[Procedural.md](Procedural.md)** — the step-by-step procedure followed: data splits, fitting order, and what each comparison (Regression vs. Sequential, COVID stress test) is designed to isolate.
- **[Results.md](Results.md)** — the results themselves: tables, a guide to reading them, and the write-up.
- **[Project%20Code.ipynb](Project%20Code.ipynb)** — the full implementation.
- **[Additional%20Information/](Additional%20Information/)** — supplementary detail, not required for the main results:
  - `Model Implementation.md` — implementation-level detail behind the procedure.
  - `Sampling.md` — the PMCMC sampling procedure for the stochastic volatility model.
  - `Sources of Error.md` — known limitations and sources of error.
  - `Extension.md` — possible extensions to the project.