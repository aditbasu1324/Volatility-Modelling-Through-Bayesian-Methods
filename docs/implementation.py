import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax
jax.config.update("jax_enable_x64", True)  # MUST be before any JAX operations
import jax.numpy as jnp
import numpyro
numpyro.set_host_device_count(4)
import matplotlib.pyplot as plt
import scipy 
from scipy import stats as scipy_stats
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
import pandas as pd
from scipy import integrate
from .data import align_series
from .metrics import sigma_to_hv_correct, sigma_sq_windowed_per_path
       
def rolling_vol(returns, h=21):
    '''Return Rolling Volatility'''
    return returns.rolling(h).std()

def E_abs_z_t(nu):
    """E[|z|] for standardized t with unit variance — log-space to avoid overflow for large nu."""
    log_num = np.log(2) + 0.5*np.log(nu-2) + scipy.special.gammaln((nu+1)/2)
    log_den = 0.5*np.log(np.pi) + np.log(nu-1) + scipy.special.gammaln(nu/2)
    return np.exp(log_num - log_den)

nu_grid = np.linspace(2.1,100, 1000)
e_abs_grid = E_abs_z_t(nu_grid)

def E_abs_z_interp_numpy(nu, nu_grid=nu_grid, e_abs_grid=e_abs_grid):
    '''NumPy equivalent of the JAX E_abs_z_interp used inside egarch_model — safe for any nu.'''
    return np.interp(nu, nu_grid, e_abs_grid)

def simulate_egarch_t_vectorized(omega, alpha, beta, gamma, nu, mu, n, ls2_init=None):
    '''Simulate n days for a batch of S parameter draws (all arrays of length S).
    ls2_init: optional per-draw starting state (array of length S). If None, uses the
    stationary initial condition omega/(1-beta) — used for prior predictive checking.
    If provided (e.g. carried-forward state), used for sequential forecast generation.
    Returns (r, sigma), each of shape (S, n).'''
    S = len(omega)
    log_sigma2 = np.zeros((S, n))
    r = np.zeros((S, n))
    e_abs = E_abs_z_interp_numpy(nu)

    if ls2_init is None:
        log_sigma2[:, 0] = np.clip(omega / (1 - beta), -20, 2)
    else:
        log_sigma2[:, 0] = np.clip(ls2_init, -20, 2)

    for t in range(1, n):
        scale = np.sqrt((nu - 2) / nu)
        z = np.random.standard_t(nu) * scale
        r[:, t] = mu + np.exp(0.5 * log_sigma2[:, t-1]) * z
        log_sigma2[:, t] = np.clip(
            omega + beta * log_sigma2[:, t-1] + gamma * z + alpha * (np.abs(z) - e_abs),
            -20, 2
        )
    return r, np.sqrt(np.exp(log_sigma2))

egarch_prior_params = {
    "beta_ab": (47, 3),
    "alpha_scale": 0.05,
    "gamma_scale": 0.05,
    "mu_scale": 0.001,
    "uncond_vol_scale": 0.003,
    "nu_gamma_scale": 2,
}

def prior_predictive_simulate_egarch(
    prior_returns_1d, omega_egarch, alpha_egarch, gamma_egarch, beta_egarch, nu_egarch, mu_egarch, prior_std,
    prior_params=egarch_prior_params,
    n_paths=1000, seed=42
):
    '''EGARCH-specific: sample parameters from priors, reject invalid draws, simulate forward.
    prior_params: dict of prior hyperparameters — pass explicitly so this matches egarch_model's own prior exactly.
    Returns (vol_paths, ret_paths, valid_paths).'''
    np.random.seed(seed)
    n = len(prior_returns_1d)

    alpha_arr = scipy_stats.truncnorm.rvs(
        (0 - alpha_egarch) / prior_params["alpha_scale"], np.inf,
        loc=alpha_egarch, scale=prior_params["alpha_scale"], size=n_paths
    )
    beta_arr  = np.random.beta(*prior_params["beta_ab"], size=n_paths)
    gamma_arr = np.random.normal(gamma_egarch, prior_params["gamma_scale"], size=n_paths)
    mu_arr    = np.random.normal(mu_egarch, prior_params["mu_scale"], size=n_paths)
    nu_arr    = 2.1 + np.random.gamma(shape=(nu_egarch - 2.1) / 2, scale=prior_params["nu_gamma_scale"], size=n_paths)
    uncond_vol_arr = scipy_stats.truncnorm.rvs(
        (0 - prior_std) / prior_params["uncond_vol_scale"], (0.02 - prior_std) / prior_params["uncond_vol_scale"],
        loc=prior_std, scale=prior_params["uncond_vol_scale"], size=n_paths
    )
    omega_arr = np.log(uncond_vol_arr**2) * (1 - beta_arr)

    valid_mask = np.array([np.isfinite(E_abs_z_t(nu_i)) for nu_i in nu_arr]) & (np.abs(beta_arr) < 1)
    alpha_v, beta_v, gamma_v, mu_v, nu_v, omega_v = (
        alpha_arr[valid_mask], beta_arr[valid_mask], gamma_arr[valid_mask],
        mu_arr[valid_mask], nu_arr[valid_mask], omega_arr[valid_mask]
    )

    ret_paths, vol_paths = simulate_egarch_t_vectorized(omega_v, alpha_v, beta_v, gamma_v, nu_v, mu_v, n)

    explosive_mask = (np.any(np.isnan(ret_paths), axis=1) | np.any(np.isinf(ret_paths), axis=1) |
                       (np.max(np.abs(ret_paths), axis=1) > 0.5))
    vol_paths = vol_paths[~explosive_mask]
    ret_paths = ret_paths[~explosive_mask]
    valid_paths = vol_paths.shape[0]

    print(f"Valid paths: {valid_paths}/{n_paths} ({valid_paths/n_paths*100:.0f}%)")
    return vol_paths, ret_paths, valid_paths

def prior_predictive_check(vol_paths, ret_paths, valid_paths, prior_returns_1d, prior_dates, hv_prior,
                            model_label="Model", color="blue", n_paths=1000, save_path=None):
    '''Model-agnostic: aggregate (Method 2 mean, per-path band), align, and plot.
    Works for any model's vol_paths/ret_paths (EGARCH, SV, ...).
    color: base color for the plotted band/mean/simulated lines (e.g. "blue" for EGARCH, "green" for SV).'''
    save_path = save_path or f"{model_label.lower()}_prior_predictive.pdf"
    dark_color = f"dark{color}"

    # ── Aggregate: mean (Method 2) and band (per-path, then percentile) ──
    mean_correct = sigma_to_hv_correct(vol_paths, h=21)
    sigma_sq_windowed = sigma_sq_windowed_per_path(vol_paths, h=21)
    lower = np.sqrt(np.nanpercentile(sigma_sq_windowed, 5, axis=0))
    upper = np.sqrt(np.nanpercentile(sigma_sq_windowed, 95, axis=0))

    # ── Build date-indexed series, align ──
    mean_correct_series = pd.Series(mean_correct, index=prior_dates[:len(mean_correct)]).dropna()
    lower_series = pd.Series(lower, index=prior_dates[:len(lower)]).dropna()
    upper_series = pd.Series(upper, index=prior_dates[:len(upper)]).dropna()

    forecast_prior, actual_prior, dates_prior = align_series(mean_correct_series, hv_prior)
    lower_aligned = lower_series.reindex(dates_prior).values
    upper_aligned = upper_series.reindex(dates_prior).values

    inside_prior = (actual_prior >= lower_aligned) & (actual_prior <= upper_aligned)
    miss_pct_prior = (1 - np.mean(inside_prior)) * 100

    # ── Plot ──
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].fill_between(dates_prior, lower_aligned, upper_aligned,
                          alpha=0.2, color=color, label="5th-95th percentile (per-path, sqrt applied last)")
    axes[0].plot(dates_prior, forecast_prior, color=dark_color,
                 linewidth=1.5, label="E[HV] across paths (correct order)")
    axes[0].plot(dates_prior, actual_prior, color="red",
                 linewidth=1.5, label="Actual Historical Volatility")
    miss_idx = np.where(~inside_prior)[0]
    axes[0].scatter(dates_prior[miss_idx], actual_prior[miss_idx],
                     color='darkred', s=15, zorder=5, label=f'Misses ({miss_pct_prior:.1f}%)')
    axes[0].set_title(f"{model_label} Prior Predictive Check Volatility: ({valid_paths}/{n_paths} accepted paths)")
    axes[0].set_ylabel("Historical Volatility"); axes[0].set_xlabel("Date"); axes[0].legend()

    ret_lower = np.percentile(ret_paths, 5, axis=0)
    ret_upper = np.percentile(ret_paths, 95, axis=0)
    ret_mean  = np.mean(ret_paths, axis=0)
    axes[1].fill_between(prior_dates, ret_lower, ret_upper, alpha=0.2, color=color, label="5th-95th percentile")
    axes[1].plot(prior_dates, ret_mean, color=dark_color, linewidth=1.5, label="Simulated Prior Mean")
    axes[1].plot(prior_dates, prior_returns_1d, color="red", linewidth=1.5, label="Actual log returns")
    axes[1].set_title("Log Returns"); axes[1].set_ylabel("Log Returns"); axes[1].set_xlabel("Date"); axes[1].legend()

    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    print(f"Miss rate (visual diagnostic only): {miss_pct_prior:.1f}%")
    print("Rejection rate indicates prior stationarity constraint tightness")

    return {
        "forecast": forecast_prior, "actual": actual_prior, "dates": dates_prior,
        "lower": lower_aligned, "upper": upper_aligned, "miss_pct": miss_pct_prior,
    }


def make_egarch_model(mu_egarch, alpha_egarch, beta_egarch, gamma_egarch, nu_egarch, prior_std, e_abs_grid):
    '''Returns a NumPyro model function, with all MLE-fit values closed over explicitly.'''
    e_abs_grid_t = jnp.array(e_abs_grid)

    def E_abs_z_interp(nu):
        idx      = (nu - 2.1) / (100 - 2.1) * 999
        idx_low  = jnp.clip(jnp.floor(idx).astype(int), 0, 998)
        idx_high = jnp.clip(idx_low + 1, 0, 999)
        frac     = idx - jnp.floor(idx)
        return e_abs_grid_t[idx_low] * (1-frac) + e_abs_grid_t[idx_high] * frac

    def egarch_model(returns):
        mu_ret     = numpyro.sample("mu_ret",     dist.Normal(mu_egarch, 0.001))
        alpha      = numpyro.sample("alpha",      dist.TruncatedNormal(alpha_egarch, 0.05, low=0))
        beta       = numpyro.sample("beta",       dist.Beta(47, 3))
        gamma      = numpyro.sample("gamma",      dist.Normal(gamma_egarch, 0.05))
        nu_shift   = numpyro.sample("nu_shift",   dist.Gamma((nu_egarch-2.1)/2, 0.5))
        nu         = nu_shift + 2.1
        uncond_vol = numpyro.sample("uncond_vol", dist.TruncatedNormal(prior_std, 0.003, low=0, high=0.02))
        omega      = jnp.log(uncond_vol**2) * (1 - beta_egarch)

        e_abs = E_abs_z_interp(nu)

        def egarch_step(ls2, r):
            z       = (r - mu_ret) / jnp.sqrt(jnp.exp(jnp.clip(ls2, -20.0, 2.0)))
            ls2_new = omega + beta*ls2 + gamma*z + alpha*(jnp.abs(z) - e_abs)
            return jnp.clip(ls2_new, -20.0, 2.0), ls2_new

        ls2_init = jnp.log(uncond_vol**2)
        _, log_sigma2 = jax.lax.scan(egarch_step, ls2_init, returns[:-1])
        log_sigma2_full = jnp.concatenate([ls2_init.reshape(1), log_sigma2])
        sigma = jnp.sqrt(jnp.exp(log_sigma2_full))
        scale = jnp.clip(sigma * jnp.sqrt((nu-2)/nu), 1e-6, 10)

        numpyro.sample("obs", dist.StudentT(nu, mu_ret, scale), obs=returns)
        numpyro.deterministic("sigma", sigma)

    return egarch_model

def forward_simulate_egarch_window(samples_prior, ls2_start_per_draw, T_w):
    '''Forward-simulate T_w days using each draw's own parameters + own carried state.'''
    S = len(samples_prior["alpha"])
    alpha_s  = np.array(samples_prior["alpha"])
    beta_s   = np.array(samples_prior["beta"])
    gamma_s  = np.array(samples_prior["gamma"])
    nu_s     = np.array(samples_prior["nu_shift"]) + 2.1
    uncond_s = np.array(samples_prior["uncond_vol"])
    omega_s  = np.log(uncond_s**2) * (1 - beta_s)
    e_abs_s  = E_abs_z_interp_numpy(nu_s)

    sigma_paths = np.zeros((S, T_w))
    ls2 = ls2_start_per_draw.copy()
    for t in range(T_w):
        sigma_paths[:, t] = np.sqrt(np.exp(np.clip(ls2, -20, 2)))
        z = np.random.standard_t(nu_s) * np.sqrt((nu_s - 2) / nu_s)
        ls2 = np.clip(omega_s + beta_s*ls2 + gamma_s*z + alpha_s*(np.abs(z) - e_abs_s), -20, 2)
    return sigma_paths

def make_egarch_model_sequential(mu_prior_mu, mu_prior_std, alpha_prior_mu, alpha_prior_std,
                                   beta_a, beta_b, gamma_prior_mu, gamma_prior_std,
                                   nu_shape, nu_rate, uncond_prior_mu, uncond_prior_std,
                                   beta_egarch, e_abs_grid, ls2_seed_scalar):
    '''Returns a NumPyro model function for one sequential window's refit, with all
    window-specific priors and the carried-forward state closed over explicitly.'''
    e_abs_grid_t = jnp.array(e_abs_grid)

    def E_abs_z_interp(nu):
        idx      = (nu - 2.1) / (100 - 2.1) * 999
        idx_low  = jnp.clip(jnp.floor(idx).astype(int), 0, 998)
        idx_high = jnp.clip(idx_low + 1, 0, 999)
        frac     = idx - jnp.floor(idx)
        return e_abs_grid_t[idx_low]*(1-frac) + e_abs_grid_t[idx_high]*frac

    def egarch_model_sequential(returns):
        mu_ret   = numpyro.sample("mu_ret",   dist.Normal(mu_prior_mu, mu_prior_std))
        alpha    = numpyro.sample("alpha",    dist.TruncatedNormal(alpha_prior_mu, alpha_prior_std, low=0))
        beta     = numpyro.sample("beta",     dist.Beta(beta_a, beta_b))
        gamma    = numpyro.sample("gamma",    dist.Normal(gamma_prior_mu, gamma_prior_std))
        nu_shift = numpyro.sample("nu_shift", dist.Gamma(nu_shape, nu_rate))
        nu       = nu_shift + 2.1
        uncond_vol = numpyro.sample("uncond_vol", dist.TruncatedNormal(uncond_prior_mu, uncond_prior_std, low=0, high=0.03))
        omega    = jnp.log(uncond_vol**2) * (1 - beta_egarch)

        e_abs = E_abs_z_interp(nu)

        def egarch_step(ls2, r):
            z = (r - mu_ret) / jnp.sqrt(jnp.exp(jnp.clip(ls2, -20.0, 2.0)))
            ls2_new = omega + beta*ls2 + gamma*z + alpha*(jnp.abs(z) - e_abs)
            return jnp.clip(ls2_new, -20.0, 2.0), ls2_new

        ls2_init = jnp.float64(ls2_seed_scalar)
        _, log_sigma2 = jax.lax.scan(egarch_step, ls2_init, returns[:-1])
        log_sigma2_full = jnp.concatenate([ls2_init.reshape(1), log_sigma2])
        sigma = jnp.sqrt(jnp.exp(log_sigma2_full))
        numpyro.deterministic("sigma", sigma)
        scale = jnp.clip(sigma * jnp.sqrt((nu-2)/nu), 1e-6, 10)
        numpyro.sample("obs", dist.StudentT(nu, mu_ret, scale), obs=returns)

    return egarch_model_sequential

def e_log_zsq_t(nu):
    # E[ln(z^2)] for standardised t_nu
    integrand = lambda z: np.log(z**2) * scipy_stats.t.pdf(z, df=nu) 
    result, _ = integrate.quad(integrand, -20, 20)
    return result

sv_prior_params = {
    "mu_scale": 1,
    "phi_ab": (7, 1),
    "sigma_eta_scale": 0.5,
    "mu_return_scale": 0.001,
}

def simulate_sv_vectorized(mu, phi, sigma_eta, mu_return, n, ls2_init=None):
    '''Simulate n days for a batch of S parameter draws (all arrays of length S).
    ls2_init: optional per-draw starting state. If None, uses the stationary initial condition —
    used for prior predictive checking. If provided, used for sequential forecast generation.
    Returns (r, sigma), each of shape (S, n).'''
    S = len(mu)
    log_sigma2 = np.zeros((S, n))
    r = np.zeros((S, n))

    if ls2_init is None:
        stationary_std = sigma_eta / np.sqrt(1 - phi**2)
        log_sigma2[:, 0] = np.clip(np.random.normal(mu, stationary_std), -20, 2)
    else:
        log_sigma2[:, 0] = np.clip(ls2_init, -20, 2)

    r[:, 0] = mu_return + np.exp(0.5 * log_sigma2[:, 0]) * np.random.normal(0, 1, S)

    for t in range(1, n):
        log_sigma2[:, t] = np.clip(
            mu + phi*(log_sigma2[:, t-1] - mu) + np.random.normal(0, sigma_eta, S),
            -20, 2
        )
        r[:, t] = mu_return + np.exp(0.5 * log_sigma2[:, t]) * np.random.normal(0, 1, S)

    return r, np.sqrt(np.exp(log_sigma2))

def prior_predictive_simulate_sv(prior_returns_1d, mu_kf, mu_return, prior_params=sv_prior_params,
                                   n_paths=1000, seed=42):
    '''SV-specific: sample parameters from priors, reject invalid draws, simulate forward.
    Returns (vol_paths, ret_paths, valid_paths).'''
    np.random.seed(seed)
    n = len(prior_returns_1d)

    mu_s_arr        = np.random.normal(mu_kf, prior_params["mu_scale"], n_paths)
    phi_s_arr       = np.random.beta(*prior_params["phi_ab"], size=n_paths)
    sigma_eta_s_arr = np.abs(np.random.normal(0, prior_params["sigma_eta_scale"], n_paths))
    mu_return_s_arr = np.random.normal(mu_return, prior_params["mu_return_scale"], n_paths)

    valid_mask = np.abs(phi_s_arr) < 1
    mu_v, phi_v, sigma_eta_v, mu_return_v = (
        mu_s_arr[valid_mask], phi_s_arr[valid_mask],
        sigma_eta_s_arr[valid_mask], mu_return_s_arr[valid_mask]
    )

    rt_sv, vol_sv = simulate_sv_vectorized(mu_v, phi_v, sigma_eta_v, mu_return_v, n)

    explosive_mask = (
        np.any(np.isnan(rt_sv), axis=1) | np.any(np.isinf(rt_sv), axis=1) |
        np.any(np.isnan(vol_sv), axis=1) | np.any(np.isinf(vol_sv), axis=1) |
        (np.max(np.abs(rt_sv), axis=1) > 0.5) |
        (np.max(vol_sv, axis=1) > 0.3) |
        (np.min(vol_sv, axis=1) < 1e-6)
    )
    vol_sv = vol_sv[~explosive_mask]
    rt_sv = rt_sv[~explosive_mask]
    valid_paths_sv = vol_sv.shape[0]

    print(f"SV valid paths: {valid_paths_sv}/{n_paths} ({valid_paths_sv/n_paths*100:.0f}%)")
    return vol_sv, rt_sv, valid_paths_sv

def particle_filter_sv(returns, mu, phi, sigma_eta, mu_return, N):
    T = len(returns)
    
    # Initialize particles from stationary distribution
    init_std = sigma_eta / np.sqrt(1 - phi**2)
    particles = np.random.normal(mu, init_std, N)
    log_likelihood = 0.0
    filtered_means = []
    
    for t in range(T):
        # 1. Propagate — state equation
        particles = mu + phi*(particles - mu) + \
                    np.random.normal(0, sigma_eta, N)
        
        # 2. Reweight — observation equation
        # how likely is the return given the volatility estimate
        sigma_t  = np.sqrt(np.exp(particles))
        log_w     = scipy_stats.norm.logpdf(returns[t], loc=mu_return, scale=sigma_t)
        max_lw    = log_w.max()                          
        log_w    -= max_lw
        weights   = np.exp(log_w) # just likelihood (the previous 2 lines are only there for numerical reasons)
        weights  /= weights.sum()
        
        log_likelihood += max_lw + np.log(np.mean(np.exp(log_w)))   
        # 3. Resample — systematic resampling
        # eliminate particles with low weights, so create try and duplicate high weights and discard low weights
        # so all N particles have similar weight
        cumsum = np.cumsum(weights)
        u      = np.random.uniform(0, 1/N) + np.arange(N)/N # note the left term is a common increase to all the particles (not n distinct)
        idx    = np.searchsorted(cumsum, u) # this is the key part of the code, 
                                            # as particles with more weight cover more interval length
        particles = particles[idx]
        
        filtered_means.append(particles.mean())
    
    return np.array(filtered_means), log_likelihood, particles

sv_prior_params = {
    "mu_scale": 1,
    "phi_ab": (7, 1),
    "sigma_eta_scale": 0.5,
    "mu_return_scale": 0.001,
}

def pmcmc_regression(returns, n_mcmc=1000, N=500,
                     mu_init=None, phi_init=0.93,
                     sigma_eta_init=None, mu_return_init=None):
    '''If mu_init/sigma_eta_init/mu_return_init are None, caller must supply them explicitly —
    no longer silently defaults to notebook-level globals.'''
    if mu_init is None or sigma_eta_init is None or mu_return_init is None:
        raise ValueError("mu_init, sigma_eta_init, and mu_return_init must be provided explicitly.")

    theta_current = np.array([mu_init, phi_init, sigma_eta_init, mu_return_init])
    filtered_means_current, ll_current, _ = particle_filter_sv(returns, *theta_current, N=N)

    T = len(returns)
    samples    = np.zeros((n_mcmc, 4))
    ll_history = np.zeros(n_mcmc)
    filtered_means_history = np.zeros((n_mcmc, T))   # NEW: store the accepted path at every iteration
    accepted   = 0

    proposal_std   = np.array([0.2, 0.02, 0.05, 0.0002])
    target_rate    = 0.20
    adapt_interval = 50
    adapt_end      = 500
    scale_factor   = 0.85

    def log_prior(theta):
        mu_p, phi_p, sigma_eta_p, mu_ret_p = theta
        lp  = scipy_stats.norm.logpdf(mu_p,     mu_init,      1.0)
        lp += scipy_stats.beta.logpdf(phi_p,    7,          1)
        lp += scipy_stats.halfnorm.logpdf(sigma_eta_p, scale=0.5)
        lp += scipy_stats.norm.logpdf(mu_ret_p, mu_return_init,  0.001)
        return lp

    print("Running Regression PMCMC...")
    for i in range(n_mcmc):
        if i > 0 and i < adapt_end and i % adapt_interval == 0:
            current_rate = accepted / i
            if current_rate < target_rate:
                proposal_std *= scale_factor
            else:
                proposal_std /= scale_factor
            proposal_std = np.clip(proposal_std, 1e-4, 1.0)
            print(f"Step {i}: rate={current_rate:.2%}, proposal_std={proposal_std.round(4)}")
        elif i % 100 == 0:
            print(f"Step {i}/{n_mcmc}, acceptance rate: {accepted/(i+1) if i>0 else 0:.2%}")

        theta_proposed = theta_current + np.random.normal(0, proposal_std, 4)

        if np.abs(theta_proposed[1]) >= 1:
            samples[i] = theta_current; ll_history[i] = ll_current
            filtered_means_history[i] = filtered_means_current   # FIX: store current on rejection too
            continue
        if theta_proposed[2] <= 0:
            samples[i] = theta_current; ll_history[i] = ll_current
            filtered_means_history[i] = filtered_means_current
            continue

        filtered_means_proposed, ll_proposed, _ = particle_filter_sv(returns, *theta_proposed, N=N)

        log_alpha = (ll_proposed + log_prior(theta_proposed)) - (ll_current + log_prior(theta_current))

        if np.log(np.random.uniform()) < log_alpha:
            theta_current = theta_proposed
            ll_current    = ll_proposed
            filtered_means_current = filtered_means_proposed   # NEW: update on acceptance
            accepted     += 1

        samples[i]    = theta_current
        ll_history[i] = ll_current
        filtered_means_history[i] = filtered_means_current   # NEW: always store whatever is current

    print(f"Final acceptance rate: {accepted/n_mcmc:.2%}")
    return samples, ll_history, filtered_means_history

def forward_simulate_sv_window(samples_prior, ls2_start_per_draw, T_w):
    '''Forward-simulate T_w days using each posterior draw's own parameters + own carried state.'''
    mu_s, phi_s, sigma_eta_s, mu_return_s = (
        samples_prior[:, 0], samples_prior[:, 1], samples_prior[:, 2], samples_prior[:, 3]
    )
    _, sigma_paths = simulate_sv_vectorized(mu_s, phi_s, sigma_eta_s, mu_return_s, T_w, ls2_init=ls2_start_per_draw)
    return sigma_paths

def pmcmc_sequential(returns, n_mcmc=1000, N=500,
                     mu_init=None, phi_init=None,
                     sigma_eta_init=None, mu_return_init=None,
                     mu_prior_mu=None, mu_prior_std=1.0,
                     phi_prior_a=7, phi_prior_b=1,
                     sigma_eta_prior_scale=0.5,
                     mu_return_prior_mu=None, mu_return_prior_std=0.001):

    mu_prior_mu        = mu_prior_mu or mu_init
    mu_return_prior_mu = mu_return_prior_mu or mu_return_init

    theta_current = np.array([mu_init, phi_init, sigma_eta_init, mu_return_init])
    filtered_means_current, ll_current, _ = particle_filter_sv(returns, *theta_current, N=N)   # FIX: capture

    T = len(returns)
    samples    = np.zeros((n_mcmc, 4))
    ll_history = np.zeros(n_mcmc)
    filtered_means_history = np.zeros((n_mcmc, T))   # FIX: new storage array
    accepted   = 0

    proposal_std   = np.array([0.2, 0.02, 0.05, 0.0002])
    target_rate    = 0.20
    adapt_interval = 50
    adapt_end      = 500
    scale_factor   = 0.85

    def log_prior(theta):
        mu_p, phi_p, sigma_eta_p, mu_ret_p = theta
        lp  = scipy_stats.norm.logpdf(mu_p,     mu_prior_mu,        mu_prior_std)
        lp += scipy_stats.beta.logpdf(phi_p,    phi_prior_a,        phi_prior_b)
        lp += scipy_stats.halfnorm.logpdf(sigma_eta_p, scale=sigma_eta_prior_scale)
        lp += scipy_stats.norm.logpdf(mu_ret_p, mu_return_prior_mu, mu_return_prior_std)
        return lp

    print("Running Sequential PMCMC...")
    for i in range(n_mcmc):
        if i > 0 and i < adapt_end and i % adapt_interval == 0:
            current_rate = accepted / i
            if current_rate < target_rate:
                proposal_std *= scale_factor
            else:
                proposal_std /= scale_factor
            proposal_std = np.clip(proposal_std, 1e-4, 1.0)
            print(f"Step {i}: rate={current_rate:.2%}, proposal_std={proposal_std.round(4)}")
        elif i % 100 == 0:
            print(f"Step {i}/{n_mcmc}, acceptance rate: {accepted/(i+1) if i>0 else 0:.2%}")

        theta_proposed = theta_current + np.random.normal(0, proposal_std, 4)

        if np.abs(theta_proposed[1]) >= 1 or theta_proposed[2] <= 0:
            samples[i] = theta_current; ll_history[i] = ll_current
            filtered_means_history[i] = filtered_means_current   # FIX: store on rejection
            continue

        filtered_means_proposed, ll_proposed, _ = particle_filter_sv(returns, *theta_proposed, N=N)

        log_alpha = (ll_proposed + log_prior(theta_proposed)) - (ll_current + log_prior(theta_current))

        if np.log(np.random.uniform()) < log_alpha:
            theta_current = theta_proposed
            ll_current    = ll_proposed
            filtered_means_current = filtered_means_proposed   # FIX: update on acceptance
            accepted     += 1

        samples[i]    = theta_current
        ll_history[i] = ll_current
        filtered_means_history[i] = filtered_means_current   # FIX: always store current

    print(f"Final acceptance rate: {accepted/n_mcmc:.2%}")
    return samples, ll_history, filtered_means_history   # FIX: return all three
