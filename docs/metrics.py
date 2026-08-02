import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm

from .data import period_indices

def qlike(sigma, rv):
    assert len(sigma) == len(rv), f"Length mismatch: sigma={len(sigma)}, rv={len(rv)}"
    ratio = rv**2 / sigma**2
    return ratio - np.log(ratio) - 1

def metric_by_period(forecast, actual, dates, boundaries, metric_fn):
    '''Compute a per-day metric, then average within each period.'''
    values = metric_fn(forecast, actual)
    results = {}
    for i, (start_idx, end_idx) in enumerate(period_indices(dates, boundaries)):
        results[f"period_{i+1}"] = {
            'start': boundaries[i], 'end': boundaries[i+1],
            'value': values[start_idx:end_idx].mean()
        }
    return results       # depends on: period_indices (data.py)
def mask_by_period(inside_mask, dates, boundaries):
    '''Average a precomputed boolean mask within each period.'''
    results = {}
    for i, (start_idx, end_idx) in enumerate(period_indices(dates, boundaries)):
        results[f"period_{i+1}"] = {
            'start': boundaries[i], 'end': boundaries[i+1],
            'value': np.mean(inside_mask[start_idx:end_idx])
        }
    return results          # depends on: period_indices (data.py)

def covid_breakdown(forecast, actual, dates, covid_start_date, covid_end_date, metric_fn):
    '''pre/covid/post breakdown of a metric — uses dates.min()/max() as the outer bounds,
    which correctly represents "everything this specific (possibly trimmed) series has available".'''
    boundaries = [dates.min(), covid_start_date, covid_end_date, dates.max()]
    result = metric_by_period(forecast, actual, dates, boundaries, metric_fn)
    return {'pre': result['period_1']['value'], 'covid': result['period_2']['value'], 'post': result['period_3']['value']}           # depends on: metric_by_period (same file)

def hv_coverage_intervals(forecast, actual, level=90, sigma_eta=None):
    if sigma_eta is None:
        sigma_eta = np.std(np.log(actual) - np.log(forecast))
    z = scipy_stats.norm.ppf(1 - (100 - level) / 200)
    lower = forecast * np.exp(-z * sigma_eta)
    upper = forecast * np.exp(+z * sigma_eta)
    inside = (actual >= lower) & (actual <= upper)
    return lower, upper, inside, sigma_eta

def hv_coverage(forecast, actual, levels=[50, 60, 70, 80, 85, 90, 95, 99]):
    sigma_eta = np.std(np.log(actual) - np.log(forecast))
    results = {}
    for level in levels:
        _, _, inside, _ = hv_coverage_intervals(forecast, actual, level, sigma_eta)
        results[level] = np.mean(inside)
    return sigma_eta, results               # depends on: hv_coverage_intervals (same file)

def hv_coverage_covid_breakdown(inside_mask, dates, covid_start_date, covid_end_date):
    boundaries = [dates.min(), covid_start_date, covid_end_date, dates.max()]
    result = mask_by_period(inside_mask, dates, boundaries)
    return {'pre': result['period_1']['value'], 'covid': result['period_2']['value'], 'post': result['period_3']['value']}  # depends on: mask_by_period (same file)

def sigma_eta_sequential(forecast, actual, forecast_regression, actual_regression, dates, window_boundaries):
    '''Expanding sigma_eta: starts with regression residuals, accumulates each completed sequential window.'''
    all_log_resid = list(np.log(actual_regression) - np.log(forecast_regression))
    log_resid_sequential = np.log(actual) - np.log(forecast)

    sigma_eta_by_window = {}
    for i, (start_idx, end_idx) in enumerate(period_indices(dates, window_boundaries)):
        sigma_eta_by_window[f"window_{i+1}"] = np.std(all_log_resid)
        all_log_resid.extend(log_resid_sequential[start_idx:end_idx])

    return sigma_eta_by_window        # depends on: period_indices (data.py)

def hv_coverage_sequential(forecast, actual, dates, window_boundaries, sigma_eta_by_window, level=90):
    '''RV coverage using a different (expanding) sigma_eta per window.'''
    z = scipy_stats.norm.ppf(1 - (100 - level) / 200)
    lower = np.zeros_like(forecast)
    upper = np.zeros_like(forecast)

    for i, (start_idx, end_idx) in enumerate(period_indices(dates, window_boundaries)):
        sigma_eta_i = sigma_eta_by_window[f"window_{i+1}"]
        lower[start_idx:end_idx] = forecast[start_idx:end_idx] * np.exp(-z * sigma_eta_i)
        upper[start_idx:end_idx] = forecast[start_idx:end_idx] * np.exp(+z * sigma_eta_i)

    inside = (actual >= lower) & (actual <= upper)
    return lower, upper, inside    # depends on: period_indices (data.py)

def compute_pit(returns, sigma, dist='normal', df=None):
    '''Compute u_t = F_t(r_t) and run KS test against Uniform(0,1).
    dist: 'normal' or 't' (Student-t, requires df).'''
    if dist == 'normal':
        u = scipy_stats.norm.cdf(returns / sigma)
    elif dist == 't':
        u = scipy_stats.t.cdf(returns / sigma, df=df)
    else:
        raise ValueError("dist must be 'normal' or 't'")
    ks_stat, p_value = scipy_stats.kstest(u, 'uniform')
    return u, ks_stat, p_value

def compute_z(u, dates):
    '''Transform u_t to z_t = Phi^{-1}(u_t) as a date-indexed Series, dropping non-finite values.'''
    z_raw = pd.Series(scipy_stats.norm.ppf(u), index=dates)
    return z_raw[np.isfinite(z_raw)]

def ljung_box_tests(z, lags=[1, 5, 10]):
    '''Run Ljung-Box on z_t (level) and z_t^2 (squared).'''
    lb_level = acorr_ljungbox(z, lags=lags, return_df=True)
    lb_sq    = acorr_ljungbox(z**2, lags=lags, return_df=True)
    return lb_level, lb_sq

def sign_bias_test(z, returns, dates):
    '''Engle-Ng sign-bias test: regress z_t^2 on sign(r_{t-1}), aligned strictly by date.'''
    returns_series = pd.Series(returns, index=dates)
    z_sq = z**2
    sign_lag = np.sign(returns_series).shift(1)

    aligned = pd.concat([z_sq, sign_lag], axis=1, join='inner').dropna()
    aligned.columns = ['z_sq', 'sign_lag']

    X = sm.add_constant(aligned['sign_lag'])
    model = sm.OLS(aligned['z_sq'], X).fit(cov_type='HC3')
    return model