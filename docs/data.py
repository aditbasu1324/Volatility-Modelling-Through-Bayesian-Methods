import numpy as np
import pandas as pd

def data_returns(data):
    '''Compute log returns from a price series r_t = ln(P_t/P_(t-1))'''
    return np.log(data["Close"]).diff().dropna()

def historical_vol(returns, h=21):
    '''Compute forward-looking historical volatility: std of r_{t+1},...,r_{t+h}.'''
    return returns.rolling(h).std().shift(-h)

def align_series(series_a, series_b):
    '''Align two Series on their common dates, returning aligned 1D arrays + the shared index.'''
    common_dates = series_a.index.intersection(series_b.index)
    a_aligned = series_a.loc[common_dates]
    b_aligned = series_b.loc[common_dates]
    return a_aligned.values.flatten(), b_aligned.values.flatten(), common_dates

def period_indices(dates, boundaries):
    '''Convert a list of boundary dates into (start_idx, end_idx) pairs for each period.
    Works for any number of periods — COVID (3) or sequential windows (8).'''
    return [(np.searchsorted(dates, boundaries[i]), np.searchsorted(dates, boundaries[i+1]))
            for i in range(len(boundaries) - 1)]