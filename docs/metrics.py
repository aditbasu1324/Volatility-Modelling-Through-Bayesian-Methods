import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
import matplotlib.pyplot as plt

def historical_vol(returns, h=21):
    '''Compute forward-looking historical volatility: std of r_{t+1},...,r_{t+h}.'''
    return returns.rolling(h).std().shift(-h)

def qlike(sigma, rv):
    ratio = rv**2 / sigma**2
    return ratio - np.log(ratio) - 1

covid_start_date = pd.Timestamp("2020-01-30")
covid_end_date = pd.Timestamp("2020-04-30")
def covid_indices(dates, start_date=covid_start_date, end_date=covid_end_date):
    return np.searchsorted(dates, start_date), np.searchsorted(dates, end_date)

def covid_breakdown(forecast, actual, start_idx, end_idx, metric_fn):
    values = metric_fn(forecast, actual)
    return {
        'pre':   values[:start_idx].mean(),
        'covid': values[start_idx:end_idx].mean(),
        'post':  values[end_idx:].mean(),
    }


def print_covid_breakdown(breakdown, dates, start_idx, end_idx, label="", metric_name="Metric"):
    print(f"{label} — COVID start: {dates[start_idx].date()} (obs {start_idx})")
    print(f"{label} — COVID end:   {dates[end_idx].date()} (obs {end_idx})")
    print(f"{label} — Pre-COVID {metric_name}:  {breakdown['pre']:.4f}")
    print(f"{label} — COVID {metric_name}:      {breakdown['covid']:.4f}")
    print(f"{label} — Post-COVID {metric_name}: {breakdown['post']:.4f}")

def hv_coverage(forecast, actual, levels=[50, 60, 70, 80, 85, 90, 95, 99]):
    '''Compute log-normal noise-adjusted RV coverage at multiple CI levels.'''
    sigma_eta = np.std(np.log(actual) - np.log(forecast))
    results = {}
    for level in levels:
        _, _, inside, _ = hv_coverage_intervals(forecast, actual, level, sigma_eta)
        results[level] = np.mean(inside)
    return sigma_eta, results


def hv_coverage_intervals(forecast, actual, level=90, sigma_eta=None):
    '''Return lower/upper bounds and inside-mask for a single CI level.
    If sigma_eta is not provided, it is computed from forecast/actual.'''
    if sigma_eta is None:
        sigma_eta = np.std(np.log(actual) - np.log(forecast))
    z = norm.ppf(1 - (100 - level) / 200)
    lower = forecast * np.exp(-z * sigma_eta)
    upper = forecast * np.exp(+z * sigma_eta)
    inside = (actual >= lower) & (actual <= upper)
    return lower, upper, inside, sigma_eta


def hv_coverage_covid_breakdown(inside_mask, covid_start_idx, covid_end_idx):
    '''Split an inside-CI boolean mask into pre/during/post-COVID coverage rates.'''
    return {
        'pre':   np.mean(inside_mask[:covid_start_idx]),
        'covid': np.mean(inside_mask[covid_start_idx:covid_end_idx]),
        'post':  np.mean(inside_mask[covid_end_idx:]),
    }


# ── Print helpers ─────────────────────────────────────────
def print_hv_coverage(sigma_eta, results, label=""):
    print(f"{label} — Measurement noise std: {sigma_eta:.4f}")
    print("Coverage at different CI levels:")
    print("-" * 40)
    for level, coverage in results.items():
        print(f"{level}% CI coverage: {coverage:.2%}")


def print_hv_coverage_covid(breakdown, label="", level=90):
    print(f"At {level}% CI coverage")
    print(f"{label} — Pre-COVID coverage:  {breakdown['pre']:.2%}")
    print(f"{label} — COVID coverage:      {breakdown['covid']:.2%}")
    print(f"{label} — Post-COVID coverage: {breakdown['post']:.2%}")

def plot_hv_coverage(forecast, actual, dates, lower, upper, inside,
                      covid_start_date=None, covid_end_date=None,
                      label="Model", title=None):
    '''Plot forecast vs. actual with a shaded CI band and miss markers.'''
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(dates, lower, upper, alpha=0.3, color='blue', label='90% CI')
    ax.plot(dates, forecast, color='darkblue', lw=1.5, label=f'{label} Forecast')
    ax.plot(dates, actual, color='red', lw=1, label='Actual Historical Volatility')

    miss_idx = np.where(~inside)[0]
    miss_dates = dates[miss_idx]
    miss_pct = (1 - np.mean(inside)) * 100

    ax.scatter(miss_dates, actual[miss_idx], color='darkred', s=15, zorder=5,
               label=f'Misses ({miss_pct:.1f}%)')

    if covid_start_date is not None:
        ax.axvline(covid_start_date, color='black', linestyle='--', alpha=0.7, lw=1, label='COVID spike start')
    if covid_end_date is not None:
        ax.axvline(covid_end_date, color='black', linestyle=':', alpha=0.7, lw=1, label='COVID spike end')

    ax.set_title(title or f"{label}: Forecast vs. Actual with 90% CI")
    ax.set_xlabel("Dates")
    ax.set_ylabel("Volatility")
    ax.legend()
    plt.tight_layout()
    plt.show()


def compute_pit(returns, sigma, dist='normal', df=None):
    '''Compute u_t = F_t(r_t) and run KS test against Uniform(0,1).
    dist: 'normal' or 't' (Student-t, requires df).'''
    if dist == 'normal':
        u = norm.cdf(returns / sigma)
    elif dist == 't':
        u = stats.t.cdf(returns / sigma, df=df)
    else:
        raise ValueError("dist must be 'normal' or 't'")
    ks_stat, p_value = kstest(u, 'uniform')
    return u, ks_stat, p_value


def print_pit_result(ks_stat, p_value, label=""):
    print(f"{label} — KS statistic: {ks_stat:.4f}")
    print(f"{label} — p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"{label} — Reject null: model is not well calibrated")
    else:
        print(f"{label} — Fail to reject null: no evidence of miscalibration")


def plot_pit(u, label=""):
    '''Histogram + empirical CDF diagnostic plots for PIT values.'''
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(u, bins=20, density=True, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axhline(1.0, color='red', linestyle='--', label='Ideal uniform density')
    axes[0].set_title(f"Histogram of PIT values (u_t) — {label}")
    axes[0].set_xlabel("u_t")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    sorted_u = np.sort(u)
    empirical_cdf = np.arange(1, len(sorted_u)+1) / len(sorted_u)
    axes[1].plot(sorted_u, empirical_cdf, color='steelblue', label='Empirical CDF of $u_t$')
    axes[1].plot([0, 1], [0, 1], color='red', linestyle='--', label='Uniform(0,1) CDF')
    axes[1].set_title(f"PIT: Empirical CDF vs. Uniform — {label}")
    axes[1].set_xlabel("u_t")
    axes[1].set_ylabel("Cumulative probability")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def compute_z(u, dates):
    '''Transform u_t to z_t = Phi^{-1}(u_t) as a date-indexed Series, dropping non-finite values.'''
    z_raw = pd.Series(norm.ppf(u), index=dates)
    return z_raw[np.isfinite(z_raw)]


def ljung_box_tests(z, lags=[1, 5, 10]):
    '''Run Ljung-Box on z_t (level) and z_t^2 (squared).'''
    lb_level = acorr_ljungbox(z, lags=lags, return_df=True)
    lb_sq    = acorr_ljungbox(z**2, lags=lags, return_df=True)
    return lb_level, lb_sq


def print_ljung_box(lb_level, lb_sq, label=""):
    print(f"Ljung-Box test on z_t (level) — {label}:")
    print(lb_level)
    print(f"\nLjung-Box test on z_t^2 (squared, ARCH-LM style) — {label}:")
    print(lb_sq)


def plot_acf_diagnostics(z, lags=20, label=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(z, lags=lags, ax=axes[0])
    axes[0].set_title(f"ACF of $z_t$ (level) — {label}")
    plot_acf(z**2, lags=lags, ax=axes[1])
    axes[1].set_title(f"ACF of $z_t^2$ (squared) — {label}")
    plt.tight_layout()
    plt.show()


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


def print_sign_bias(model, label=""):
    print(f"Engle-Ng Sign-Bias Test — {label}:")
    print(model.summary())

# ── Window boundaries ────────────────────────────────────────────
window_boundaries = pd.date_range(start="2022-01-01", end="2026-01-01", freq="6MS")

def window_indices(dates, window_boundaries):
    '''Return list of (start_idx, end_idx) pairs for each window.'''
    return [(np.searchsorted(dates, window_boundaries[i]), np.searchsorted(dates, window_boundaries[i+1]))
            for i in range(len(window_boundaries) - 1)]

def qlike_by_window(forecast, actual, dates, window_boundaries):
    '''Mean QLIKE within each window.'''
    results = {}
    for i, (start_idx, end_idx) in enumerate(window_indices(dates, window_boundaries)):
        window_qlike = qlike(forecast[start_idx:end_idx], actual[start_idx:end_idx]).mean()
        results[f"window_{i+1}"] = {
            'start': window_boundaries[i], 'end': window_boundaries[i+1], 'qlike': window_qlike
        }
    return results

def print_qlike_by_window(results, label=""):
    print(f"QLIKE by window — {label}:")
    for name, r in results.items():
        print(f"  {name} ({r['start'].date()} to {r['end'].date()}): {r['qlike']:.4f}")

def sigma_eta_sequential(forecast, actual, forecast_regression, actual_regression, dates, window_boundaries):
    '''Expanding sigma_eta: starts with regression residuals, accumulates each completed sequential window.'''
    all_log_resid = list(np.log(actual_regression) - np.log(forecast_regression))
    log_resid_sequential = np.log(actual) - np.log(forecast)

    sigma_eta_by_window = {}
    for i, (start_idx, end_idx) in enumerate(window_indices(dates, window_boundaries)):
        sigma_eta_by_window[f"window_{i+1}"] = np.std(all_log_resid)
        all_log_resid.extend(log_resid_sequential[start_idx:end_idx])

    return sigma_eta_by_window

def hv_coverage_sequential(forecast, actual, dates, window_boundaries, sigma_eta_by_window, level=90):
    '''RV coverage using a different (expanding) sigma_eta per window.'''
    z = norm.ppf(1 - (100 - level) / 200)
    lower = np.zeros_like(forecast)
    upper = np.zeros_like(forecast)

    for i, (start_idx, end_idx) in enumerate(window_indices(dates, window_boundaries)):
        sigma_eta_i = sigma_eta_by_window[f"window_{i+1}"]
        lower[start_idx:end_idx] = forecast[start_idx:end_idx] * np.exp(-z * sigma_eta_i)
        upper[start_idx:end_idx] = forecast[start_idx:end_idx] * np.exp(+z * sigma_eta_i)

    inside = (actual >= lower) & (actual <= upper)
    return lower, upper, inside

def coverage_by_window(inside, dates, window_boundaries):
    '''% inside CI within each window, given a precomputed inside-mask.'''
    results = {}
    for i, (start_idx, end_idx) in enumerate(window_indices(dates, window_boundaries)):
        results[f"window_{i+1}"] = {
            'start': window_boundaries[i], 'end': window_boundaries[i+1],
            'coverage': np.mean(inside[start_idx:end_idx])
        }
    return results

def print_coverage_by_window(results, label="", level=90):
    print(f"{level}% CI Coverage by window — {label}:")
    for name, r in results.items():
        print(f"  {name} ({r['start'].date()} to {r['end'].date()}): {r['coverage']:.2%}")

def plot_hv_coverage_windows(forecast, actual, dates, lower, upper, inside, window_boundaries, label="Model"):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(dates, lower, upper, alpha=0.3, color='blue', label='90% CI')
    ax.plot(dates, forecast, color='darkblue', lw=1.5, label=f'{label} Forecast')
    ax.plot(dates, actual, color='red', lw=1, label='Actual Historical Volatility')

    miss_idx = np.where(~inside)[0]
    miss_pct = (1 - np.mean(inside)) * 100
    ax.scatter(dates[miss_idx], actual[miss_idx], color='darkred', s=15, zorder=5,
               label=f'Misses ({miss_pct:.1f}%)')

    for i, boundary in enumerate(window_boundaries):
        if dates[0] <= boundary <= dates[-1]:
            ax.axvline(boundary, color='black', linestyle='--', alpha=0.5, lw=1,
                       label='Window boundary' if i == 0 else None)

    ax.set_title(f"{label}: Forecast vs. Actual with 90% CI (Sequential Windows)")
    ax.set_xlabel("Dates")
    ax.set_ylabel("Volatility")
    ax.legend()
    plt.tight_layout()
    plt.show()