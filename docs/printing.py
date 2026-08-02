import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

def print_date(dates, title):
      print(f"{title}: {dates[0].date()} to {dates[-1].date()}, "
            f"n={len(dates)}")

def print_covid_breakdown(breakdown, label="", metric_name="Metric"):
    print(f"{label} — Pre-COVID {metric_name}:  {breakdown['pre']:.4f}")
    print(f"{label} — COVID {metric_name}:      {breakdown['covid']:.4f}")
    print(f"{label} — Post-COVID {metric_name}: {breakdown['post']:.4f}")

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

def print_metric_by_window(results, label="", metric_name="Metric"):
    print(f"{metric_name} by window — {label}:")
    for name, r in results.items():
        print(f"  {name} ({pd.Timestamp(r['start']).date()} to {pd.Timestamp(r['end']).date()}): {r['value']:.4f}")

def print_coverage_by_window(results, label="", level=90):
    print(f"{level}% CI Coverage by window — {label}:")
    for name, r in results.items():
        print(f"  {name} ({pd.Timestamp(r['start']).date()} to {pd.Timestamp(r['end']).date()}): {r['value']:.2%}")

def print_pit_result(ks_stat, p_value, label=""):
    print(f"{label} — KS statistic: {ks_stat:.4f}")
    print(f"{label} — p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"{label} — Reject null: model is not well calibrated")
    else:
        print(f"{label} — Fail to reject null: no evidence of miscalibration")

def print_ljung_box(lb_level, lb_sq, label=""):
    print(f"Ljung-Box test on z_t (level) — {label}:")
    print(lb_level)
    print(f"\nLjung-Box test on z_t^2 (squared, ARCH-LM style) — {label}:")
    print(lb_sq)

def print_sign_bias(model, label=""):
    print(f"Engle-Ng Sign-Bias Test — {label}:")
    print(model.summary())

def plot_hv_coverage(forecast, actual, dates, lower, upper, inside, level=90, label="Model",
                      boundary_dates=None, boundary_labels=None, boundary_styles=None):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(dates, lower, upper, alpha=0.3, color='blue', label=f'{level}% CI')
    ax.plot(dates, forecast, color='darkblue', lw=1.5, label=f'{label} Forecast')
    ax.plot(dates, actual, color='red', lw=1, label='Actual Historical Volatility')

    miss_idx = np.where(~inside)[0]
    miss_pct = (1 - np.mean(inside)) * 100
    ax.scatter(dates[miss_idx], actual[miss_idx], color='darkred', s=15, zorder=5, label=f'Misses ({miss_pct:.1f}%)')

    if boundary_dates is not None:
        for i, b in enumerate(boundary_dates):
            if dates[0] <= b <= dates[-1]:
                style = boundary_styles[i] if boundary_styles else '--'
                lbl = boundary_labels[i] if boundary_labels else ('Boundary' if i == 0 else None)
                ax.axvline(b, color='black', linestyle=style, alpha=0.6, lw=1, label=lbl)

    ax.set_title(f"{label}: Forecast vs. Actual with {level}% CI")
    ax.set_xlabel("Dates"); ax.set_ylabel("Volatility"); ax.legend()
    plt.tight_layout(); plt.show()       # unified version, handles both COVID and windows

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

def plot_acf_diagnostics(z, lags=20, label=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(z, lags=lags, ax=axes[0])
    axes[0].set_title(f"ACF of $z_t$ (level) — {label}")
    plot_acf(z**2, lags=lags, ax=axes[1])
    axes[1].set_title(f"ACF of $z_t^2$ (squared) — {label}")
    plt.tight_layout()
    plt.show()


def print_hv_coverage_posterior(sigma_eta, results, label=""):
    print(f"{label} — Measurement noise std: {sigma_eta:.4f}")
    print("Coverage at different CI levels (posterior + noise):")
    print("-" * 40)
    for level, coverage in results.items():
        print(f"{level}% CI coverage: {coverage:.2%}")   