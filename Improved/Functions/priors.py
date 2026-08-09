import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist

def fit_beta_params(samples):
    a, b, _, _ = beta_dist.fit(np.array(samples), floc=0, fscale=1)
    return float(a), float(b)

def fit_gamma_params(samples):
    mu, var = np.array(samples).mean(), np.array(samples).var()
    return float(mu**2 / var), float(mu / var)

def fit_halfnorm_scale(samples):
    '''Method-of-moments fit of a half-Normal scale parameter to posterior samples.'''
    # For half-Normal, E[X] = scale * sqrt(2/pi)  =>  scale = mean / sqrt(2/pi)
    return float(np.array(samples).mean() / np.sqrt(2/np.pi))