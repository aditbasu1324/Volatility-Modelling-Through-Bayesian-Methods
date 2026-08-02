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
                            model_label="Model", n_paths=1000, save_path=None):
    '''Model-agnostic: aggregate (Method 2 mean, per-path band), align, and plot.
    Works for any model's vol_paths/ret_paths (EGARCH, SV, ...).'''
    save_path = save_path or f"{model_label.lower()}_prior_predictive.pdf"

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
                          alpha=0.2, color="blue", label="5th-95th percentile (per-path, sqrt applied last)")
    axes[0].plot(dates_prior, forecast_prior, color="darkblue",
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
    axes[1].fill_between(prior_dates, ret_lower, ret_upper, alpha=0.2, color="blue", label="5th-95th percentile")
    axes[1].plot(prior_dates, ret_mean, color="darkblue", linewidth=1.5, label="Simulated Prior Mean")
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
def egarch_model(returns):
    mu_ret     = numpyro.sample("mu_ret",     dist.Normal(mu_egarch, 0.001))
    alpha      = numpyro.sample("alpha",      dist.TruncatedNormal(alpha_egarch, 0.05, low=0))
    beta       = numpyro.sample("beta",       dist.Beta(47, 3))
    gamma      = numpyro.sample("gamma",      dist.Normal(gamma_egarch, 0.05))
    nu_shift   = numpyro.sample("nu_shift",   dist.Gamma((nu_egarch-2.1)/2, 0.5))
    nu         = nu_shift + 2.1
    uncond_vol = numpyro.sample("uncond_vol", dist.TruncatedNormal(
                                    prior_std, 0.003, low=0, high=0.02))
    omega      = jnp.log(uncond_vol**2) * (1 - beta_egarch)  # jnp + fixed beta (can't use beta as otherwise divergences occurs)

    e_abs_grid_t = jnp.array(e_abs_grid)
    def E_abs_z_interp(nu):
        idx      = (nu - 2.1) / (100 - 2.1) * 999
        idx_low  = jnp.clip(jnp.floor(idx).astype(int), 0, 998)
        idx_high = jnp.clip(idx_low + 1, 0, 999)
        frac     = idx - jnp.floor(idx)
        return e_abs_grid_t[idx_low] * (1-frac) + e_abs_grid_t[idx_high] * frac

    e_abs = E_abs_z_interp(nu)

    def egarch_step(ls2, r):
        z       = (r - mu_ret) / jnp.sqrt(jnp.exp(jnp.clip(ls2, -20.0, 2.0)))
        ls2_new = omega + beta*ls2 + gamma*z + alpha*(jnp.abs(z) - e_abs)
        return jnp.clip(ls2_new, -20.0, 2.0), ls2_new
        
    ls2_init = jnp.log(uncond_vol**2) 
    
    _, log_sigma2 = jax.lax.scan(egarch_step, ls2_init, returns[:-1])
    # log_sigma2 shape: (T-1,) — σ_1...σ_{T-1}
    
    # Prepend σ_0
    log_sigma2_full = jnp.concatenate([ls2_init.reshape(1), log_sigma2])
    # log_sigma2_full shape: (T,) — σ_0...σ_{T-1}

    sigma = jnp.sqrt(jnp.exp(log_sigma2_full))
    scale = jnp.clip(sigma * jnp.sqrt((nu-2)/nu), 1e-6, 10)
    
    # Now use full returns — all T observations
    numpyro.sample("obs", dist.StudentT(nu, mu_ret, scale), obs=returns) #not returns[1:]
    numpyro.deterministic("sigma", sigma)



