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