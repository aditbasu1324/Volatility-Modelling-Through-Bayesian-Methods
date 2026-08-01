import numpy as np
import pandas as pd

def data_returns(data):
    '''Compute log returns from a price series r_t = ln(P_t/P_(t-1))'''
    return np.log(data["Close"]).diff().dropna()

def align_series(series_a, series_b):
    '''Align two Series on their common dates, returning aligned 1D arrays + the shared index.'''
    common_dates = series_a.index.intersection(series_b.index)
    a_aligned = series_a.loc[common_dates]
    b_aligned = series_b.loc[common_dates]
    return a_aligned.values.flatten(), b_aligned.values.flatten(), common_dates



