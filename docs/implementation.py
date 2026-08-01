import numpy as np
from scipy.special import gammaln            # for gammaln, used in E_abs_z_t
from scipy import stats as scipy_stats
import pandas as pd

def rolling_vol(returns, h=21):
    '''Return Rolling Volatility'''
    return returns.rolling(h).std()



