Model Implementation 

Explains how complex techniques like MCMC and PMCMC are implemented

For EGARCH MCMC

## MCMC via NumPyro

### The General Idea

NumPyro implements Hamiltonian Monte Carlo (specifically NUTS — the No-U-Turn Sampler), which differs from textbook Metropolis-Hastings in how proposals are generated: rather than proposing a random step and accepting/rejecting based on the posterior ratio alone, NUTS uses the *gradient* of the log-posterior to propose informed moves through parameter space, dramatically improving sampling efficiency in high-dimensional or correlated posteriors (like a GARCH-family model's parameters). The accept/reject mechanism at the core of Metropolis-Hastings is still present — NUTS is best understood as a smarter proposal mechanism layered on top of the same fundamental MCMC acceptance framework.

### Method

1. Sample parameters from their priors (as specified in the Priors section above).
2. Propagate the EGARCH recursion forward through the regression period's returns, generating the implied $\sigma_t$ path for that specific parameter draw.
3. Use this $\sigma_t$ path to evaluate the likelihood of the full regression-period returns under a Student-t observation model, and let NUTS use this (and its gradient) to inform the next proposal.
4. Repeat across many draws to build up the posterior.

### Implementation Techniques

**Interpolation grid for $E|z|$**: the exact form of $E[|z|]$ as a function of $\nu$ (needed in the EGARCH recursion) involves gamma functions that can behave poorly under JAX's automatic differentiation at extreme parameter values. A precomputed grid of $E|z|$ values across a range of $\nu$, linearly interpolated at each step, avoids this while remaining fast and differentiable.

**Return indexing**: since $\sigma_t$ depends on the shock $z_{t-1}$ (via $r_{t-1}$), the recursion is scanned over `returns[:-1]` to produce $\sigma_1,...,\sigma_{T-1}$; $\sigma_0$ is set to its initial value directly. The full return series (all $T$ observations) is then used for the likelihood, since $r_t=\mu+\sigma_t z_t$ requires a $\sigma_t$ for every observed $t$, including $t=0$.

### Known Approximations

Two implementation choices, made to avoid divergences during sampling, introduce small inconsistencies relative to the exact theoretical specification:

- **$\omega$ uses the MLE-fitted $\hat\beta$, not the sampled $\beta$**: the stationarity relation $\omega=(1-\beta)\log\sigma_\infty^2$ is applied using the fixed point estimate $\hat\beta_{egarch}$ rather than each draw's own sampled $\beta$. This means a given posterior draw's actual implied long-run variance may deviate slightly from $\sigma_\infty^2$ if that draw's sampled $\beta$ differs from $\hat\beta_{egarch}$ — a pragmatic stabilization choice, not an exact implementation of the stationarity condition for every individual draw.
- **Fixed initial condition**: $\log\sigma_0^2$ is initialized from the empirical `prior_std` for every draw, rather than from that draw's own sampled `uncond_vol` — introducing a similar small mismatch at the start of each simulated path specifically.

Both approximations affect only the very early part of the fitted path and the precise long-run level implied by any single draw; they do not affect the overall validity of the recursion or the sampling procedure itself.

