Model Implementation 

This explains the basis for complicated techniques like MCMC and PMCMC and explains how they are implemented.

## MCMC

## MCMC: Why It Works

### The Goal

Bayes' theorem gives the posterior:
$$p(\theta\mid\text{data}) = \frac{p(\text{data}\mid\theta)\,p(\theta)}{p(\text{data})}$$

The normalizing constant $p(\text{data}) = \int p(\text{data}\mid\theta)p(\theta)\,d\theta$ is generally intractable to compute directly — no closed form exists, and numerical integration is infeasible in more than a few dimensions (here, 6: $\mu_{ret},\alpha,\beta,\gamma,\nu,\sigma_\infty$). This intractability is the entire motivation for MCMC.

The key workaround: you don't need $p(\text{data})$ to *sample* from the posterior — only to evaluate it *up to a proportionality constant*:
$$\pi(\theta) := p(\theta\mid\text{data}) \propto p(\text{data}\mid\theta)\,p(\theta)$$
The right-hand side is exactly computable for any specific $\theta$, using only the likelihood and prior.

### Markov Chains and Stationary Distributions

A Markov chain is a sequence of random variables $\theta_0,\theta_1,\theta_2,...$ where each state depends only on the previous one, via a transition kernel $K(\theta_{i+1}\mid\theta_i)$.

A distribution $\pi^*$ is a **stationary distribution** of the chain if:
$$\int \pi^*(\theta)\,K(\theta'\mid\theta)\,d\theta = \pi^*(\theta') \quad \text{for all }\theta'$$
— i.e., if the current state is distributed as $\pi^*$, so is the next state.

**The core strategy of MCMC**: construct a transition kernel $K$ whose stationary distribution is exactly $\pi(\theta)=p(\theta\mid\text{data})$. Simulating the chain — actually generating $\theta_0,\theta_1,\theta_2,...$ by repeatedly applying $K$ — then produces a sequence whose marginal distribution converges to $\pi$, regardless of where the chain started (this convergence requires the chain to be **ergodic**: irreducible, meaning it can reach any region of parameter space, and aperiodic, meaning it doesn't cycle deterministically — both hold essentially automatically for continuous, well-behaved posteriors with reasonable proposals).

### Detailed Balance: The Condition That Makes This Possible

A sufficient (not necessary) condition guaranteeing $\pi$ is a stationary distribution of $K$ is **detailed balance**:
$$\pi(\theta)\,K(\theta'\mid\theta) = \pi(\theta')\,K(\theta\mid\theta')$$

This implies stationarity directly: integrating both sides over $\theta$,
$$\int\pi(\theta)K(\theta'\mid\theta)\,d\theta = \int\pi(\theta')K(\theta\mid\theta')\,d\theta = \pi(\theta')\int K(\theta\mid\theta')\,d\theta = \pi(\theta')$$
(using that $K(\theta\mid\theta')$ integrates to 1, being a valid transition kernel) — exactly the stationarity condition.

### Metropolis-Hastings: A Kernel Satisfying Detailed Balance

Given a proposal distribution $q(\theta'\mid\theta)$ (e.g. a random-walk step $\theta'=\theta+\epsilon$), define the acceptance probability:
$$\alpha(\theta\to\theta') = \min\left(1,\ \frac{\pi(\theta')\,q(\theta\mid\theta')}{\pi(\theta)\,q(\theta'\mid\theta)}\right)$$
and the resulting kernel $K(\theta'\mid\theta) = q(\theta'\mid\theta)\,\alpha(\theta\to\theta')$.

**This construction satisfies detailed balance for any $\pi$ and any $q$** — a direct algebraic check confirms it:
$$\pi(\theta)K(\theta'\mid\theta) = \min\big(\pi(\theta)q(\theta'\mid\theta),\ \pi(\theta')q(\theta\mid\theta')\big) = \pi(\theta')K(\theta\mid\theta')$$
(both sides reduce to the same expression, since the $\min(1,\cdot)$ factor cancels the larger of the two terms identically either way). This is why the specific $\min(1,\cdot)$ form of $\alpha$ was chosen — it is engineered precisely to make this cancellation hold.

**Why the intractable $p(\text{data})$ never needs to be computed**: the acceptance ratio only involves $\pi(\theta')/\pi(\theta)$:
$$\frac{\pi(\theta')}{\pi(\theta)} = \frac{p(\text{data}\mid\theta')p(\theta')/p(\text{data})}{p(\text{data}\mid\theta)p(\theta)/p(\text{data})} = \frac{p(\text{data}\mid\theta')p(\theta')}{p(\text{data}\mid\theta)p(\theta)}$$
$p(\text{data})$ appears in both numerator and denominator and cancels exactly.

### Putting It Together

1. Choose $\pi(\theta) = p(\theta\mid\text{data})$ as the target.
2. Construct $K$ via the Metropolis-Hastings acceptance formula above — this guarantees $\pi$ satisfies detailed balance for this specific $K$, for whatever $\pi$ is plugged in.
3. Detailed balance guarantees $\pi$ is *a* stationary distribution of $K$.
4. Ergodicity guarantees $\pi$ is the *unique* stationary distribution, and that repeatedly simulating the chain converges to it regardless of starting point.
5. Simulating the chain long enough, then discarding an initial burn-in period (`num_warmup`) where convergence is still in progress, yields a sequence of samples whose distribution approximates $p(\theta\mid\text{data})$ — usable for Monte Carlo estimation (e.g. posterior means, credible intervals) via the ergodic theorem for Markov chains, the MCMC analog of the law of large numbers.

The remaining practical weakness of this basic construction — a naive random-walk $q$ proposes blindly, leading to low acceptance rates and slow exploration in high-dimensional, correlated posteriors like this one — is what motivates NUTS, covered separately.

## Implementation: How This Is Actually Computed

Assuming the theory above (target $\pi(\theta)=p(\theta\mid\text{data})$, detailed balance, ergodicity), this section covers how each piece is actually evaluated in code.

### Step 1: Propose a Parameter Vector $\theta'$

$\theta' = (\mu_{ret}, \alpha, \beta, \gamma, \nu, \sigma_\infty)$ — one draw of all 6 parameters simultaneously (see NUTS section for exactly how this proposal is generated).

### Step 2: Recurse Through the Volatility Equation

Given $\theta'$, the log-variance path is not observed directly — it must be computed by running the EGARCH recursion forward, one step at a time, starting from an initial value:

$$\log\sigma_0^2 = \text{initial value (e.g. from } \sigma_\infty^2\text{)}$$
$$\log\sigma_t^2 = \omega + \beta\log\sigma_{t-1}^2 + \gamma z_{t-1} + \alpha\left(|z_{t-1}| - E|z_{t-1}|\right), \quad z_{t-1} = \frac{r_{t-1}-\mu_{ret}}{\sigma_{t-1}}$$

This is a genuinely **sequential** computation — $\sigma_t$ cannot be computed without first computing $\sigma_{t-1}$, which requires $\sigma_{t-2}$, and so on back to $\sigma_0$. In code, this is implemented via `jax.lax.scan`, which applies the same step function repeatedly across the whole return series, carrying the running $\log\sigma_{t-1}^2$ forward at each step and collecting every $\log\sigma_t^2$ along the way — functionally equivalent to a `for` loop, but compiled by JAX into an efficient, differentiable operation (critical for NUTS, which needs gradients through this entire recursion).

**Numerical stability practicalities**:
- $\log\sigma_t^2$ is clipped to a bounded range (e.g. $[-20, 2]$) at each step, preventing the recursion from diverging to numerically invalid values during the exploratory (often initially poor) proposals a sampler makes before converging.
- $E|z_{t-1}|$ depends on $\nu$ through a Gamma-function expression that is numerically unstable to differentiate directly; instead, it's precomputed on a fine grid of $\nu$ values and linearly interpolated at each step — avoiding repeated, unstable evaluation of the exact formula inside the recursion.

### Step 3: Compute $p(\text{data}\mid\theta')$ from the Resulting $\sigma_t$ Path

Once the full path $\sigma_0,\sigma_1,...,\sigma_{T-1}$ has been computed for this specific $\theta'$, the likelihood follows directly from the observation equation $r_t=\mu_{ret}+\sigma_t z_t$, $z_t\sim t_\nu(0,1)$:

$$p(r_t\mid\sigma_t,\theta') = \frac{1}{\sigma_t}f_{t_\nu}\left(\frac{r_t-\mu_{ret}}{\sigma_t}\right)$$

and, treating each day's return as conditionally independent given its own $\sigma_t$:

$$\log p(\text{data}\mid\theta') = \sum_{t=0}^{T-1}\log p(r_t\mid\sigma_t,\theta')$$

In code, this sum is computed in one call: `numpyro.sample("obs", dist.StudentT(nu, mu_ret, scale), obs=returns)` evaluates the Student-t log-density at every observed return simultaneously (using the just-computed $\sigma_t$ path to build `scale`), and NumPyro internally sums these into the single log-likelihood value used in the acceptance/proposal machinery.

### Step 4: Combine with the Prior, Evaluate the Posterior Ratio

$$\log\pi(\theta') = \log p(\text{data}\mid\theta') + \log p(\theta')$$

$\log p(\theta')$ is just the sum of the log-densities of each prior distribution (Normal, Beta, Gamma, TruncatedNormal) evaluated at $\theta'$'s components — cheap, closed-form, no recursion needed. Whatever sampler is used (plain MH or NUTS), it operates on this combined quantity — the recursion (Steps 2–3) is by far the most expensive part of evaluating $\pi$ at any given $\theta$.


### Where NUTS Differs From This Basic Implementation

Everything above — the recursion, the likelihood computation, combining with the prior — is identical regardless of which sampler is used; it's simply "how do I evaluate $\log\pi(\theta)$ at a given point," needed by any MCMC method.

What NUTS adds, specifically:

1. **Automatic differentiation through the whole recursion**: NUTS needs $\nabla_\theta\log\pi(\theta)$, not just $\log\pi(\theta)$ itself. Since the recursion (Step 2) and likelihood (Step 3) are built from JAX operations, JAX's autodiff can differentiate all the way through the `lax.scan` loop automatically — computing how a small change in $\alpha,\beta,\gamma,\nu,\mu_{ret},\sigma_\infty$ would change the final log-posterior, without manually deriving any derivatives.
2. **Leapfrog steps using this gradient**: rather than proposing $\theta'$ blindly, the gradient computed in (1) is used to simulate a short physics-like trajectory (the leapfrog equations from before), producing a candidate $\theta'$ that's informed by which direction increases posterior density.
3. **Adaptive trajectory length and step size**: the No-U-Turn criterion and dual-averaging step-size tuning (both covered earlier) remove the need to manually choose how far to simulate the trajectory or how large each step should be.

**In short**: Steps 1–4 above (recursion → likelihood → prior → posterior ratio) are the actual computational core that *any* sampler needs to evaluate at every proposed point; NUTS's contribution is entirely in *how efficiently and cleverly it chooses which points to evaluate this at*, using the gradient of exactly this same $\log\pi(\theta)$ computation.

**Return indexing**: since $\sigma_t$ depends on the shock $z_{t-1}$ (via $r_{t-1}$), the recursion is scanned over `returns[:-1]` to produce $\sigma_1,...,\sigma_{T-1}$; $\sigma_0$ is set to its initial value directly. The full return series (all $T$ observations) is then used for the likelihood, since $r_t=\mu+\sigma_t z_t$ requires a $\sigma_t$ for every observed $t$, including $t=0$.

### Known Approximations

In order to avoid divergences during sampling, need to slightly adapt the model for practical purposes:

- **$\omega$ uses the MLE-fitted $\hat\beta$, not the sampled $\beta$**: the stationarity relation $\omega=(1-\beta)\log\sigma_\infty^2$ is applied using the fixed point estimate $\hat\beta_{egarch}$ rather than each draw's own sampled $\beta$. This means a given posterior draw's actual implied long-run variance may deviate slightly from $\sigma_\infty^2$ if that draw's sampled $\beta$ differs from $\hat\beta_{egarch}$ — a pragmatic stabilization choice, not an exact implementation of the stationarity condition for every individual draw.

The approximation affect only the very early part of the fitted path and the precise long-run level implied by any single draw; they do not affect the overall validity of the recursion or the sampling procedure itself.

## MCMC via NumPyro

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

In order to avoid divergences during sampling, need to slightly adapt the model for practical purposes:

- **$\omega$ uses the MLE-fitted $\hat\beta$, not the sampled $\beta$**: the stationarity relation $\omega=(1-\beta)\log\sigma_\infty^2$ is applied using the fixed point estimate $\hat\beta_{egarch}$ rather than each draw's own sampled $\beta$. This means a given posterior draw's actual implied long-run variance may deviate slightly from $\sigma_\infty^2$ if that draw's sampled $\beta$ differs from $\hat\beta_{egarch}$ — a pragmatic stabilization choice, not an exact implementation of the stationarity condition for every individual draw.

The approximationaffect only the very early part of the fitted path and the precise long-run level implied by any single draw; they do not affect the overall validity of the recursion or the sampling procedure itself.

