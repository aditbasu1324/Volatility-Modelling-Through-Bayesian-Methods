## MCMC: Why It Works

### The Goal

Bayes' theorem: $p(\theta\mid\text{data}) = \frac{p(\text{data}\mid\theta)p(\theta)}{p(\text{data})}$.

$p(\text{data}) = \int p(\text{data}\mid\theta)p(\theta)\,d\theta$ is generally intractable (no closed form; infeasible to integrate numerically in 6 dimensions here). This is the entire motivation for MCMC.

The workaround: sampling only requires evaluating the posterior *up to a constant*:
$$\pi(\theta) := p(\theta\mid\text{data}) \propto p(\text{data}\mid\theta)\,p(\theta)$$
— exactly computable at any specific $\theta$, using only the likelihood and prior.

### Markov Chains and Stationary Distributions

A Markov chain $\theta_0,\theta_1,...$ evolves via a transition kernel $K(\theta_{i+1}\mid\theta_i)$. $\pi^*$ is a **stationary distribution** if:
$$\int \pi^*(\theta)K(\theta'\mid\theta)\,d\theta = \pi^*(\theta') \quad \forall\theta'$$

**Strategy**: construct $K$ with stationary distribution $\pi(\theta)=p(\theta\mid\text{data})$. Simulating the chain then converges to $\pi$ regardless of starting point, provided the chain is **ergodic** (irreducible + aperiodic — holds automatically for continuous, well-behaved posteriors with reasonable proposals).

### Detailed Balance

A sufficient condition for $\pi$ to be stationary:
$$\pi(\theta)K(\theta'\mid\theta) = \pi(\theta')K(\theta\mid\theta')$$
Integrating both sides over $\theta$ gives $\int\pi(\theta)K(\theta'\mid\theta)\,d\theta = \pi(\theta')$ directly (using $\int K(\theta\mid\theta')\,d\theta=1$) — exactly the stationarity condition.

### Metropolis-Hastings

Given proposal $q(\theta'\mid\theta)$, define:
$$\alpha(\theta\to\theta') = \min\left(1,\ \frac{\pi(\theta')q(\theta\mid\theta')}{\pi(\theta)q(\theta'\mid\theta)}\right), \qquad K(\theta'\mid\theta) = q(\theta'\mid\theta)\alpha(\theta\to\theta')$$

This satisfies detailed balance for **any** $\pi,q$:
$$\pi(\theta)K(\theta'\mid\theta) = \min\big(\pi(\theta)q(\theta'\mid\theta),\ \pi(\theta')q(\theta\mid\theta')\big) = \pi(\theta')K(\theta\mid\theta')$$
— the $\min(1,\cdot)$ form is engineered specifically to make this cancellation hold.

**Why $p(\text{data})$ cancels**:
$$\frac{\pi(\theta')}{\pi(\theta)} = \frac{p(\text{data}\mid\theta')p(\theta')/p(\text{data})}{p(\text{data}\mid\theta)p(\theta)/p(\text{data})} = \frac{p(\text{data}\mid\theta')p(\theta')}{p(\text{data}\mid\theta)p(\theta)}$$

### Putting It Together

1. Target $\pi(\theta)=p(\theta\mid\text{data})$.
2. MH's $K$ satisfies detailed balance for this $\pi$, for any $q$ — shown above.
3. Detailed balance $\Rightarrow$ $\pi$ is *a* stationary distribution.
4. Ergodicity $\Rightarrow$ $\pi$ is the *unique* stationary distribution, reached from any start.
5. Simulate the chain, discard burn-in (`num_warmup`), use remaining samples for Monte Carlo estimation (posterior means, credible intervals) via the ergodic theorem.

**Remaining weakness**: a naive random-walk $q$ proposes blindly, giving low acceptance and slow exploration in high-dimensional, correlated posteriors like this one — this motivates NUTS.

### NUTS: Extending Metropolis-Hastings with Gradient Information

**Setup**: introduce auxiliary momentum $p\in\mathbb{R}^d$ (same dimension as $\theta$, artificial — discarded each iteration), $p\sim N(0,M)$. Define potential energy $U(\theta)=-\log\pi(\theta)$ and joint density $\pi(\theta,p)\propto\exp(-U(\theta)-\frac12 p^TM^{-1}p)$. Marginalizing out $p$ recovers $\pi(\theta)$ exactly.

**Leapfrog integrator** (one step, size $\epsilon$) — this is where $\nabla\log\pi(\theta)$ enters:
$$p_{1/2} = p + \frac{\epsilon}{2}\nabla\log\pi(\theta), \qquad \theta' = \theta + \epsilon M^{-1}p_{1/2}, \qquad p' = p_{1/2} + \frac{\epsilon}{2}\nabla\log\pi(\theta')$$
Repeating this $L$ times traces a trajectory $(\theta,p)\to\cdots\to(\theta_L,p_L)$, pushed toward higher posterior density at every step by the gradient.

**Basic HMC acceptance** (still Metropolis-Hastings):
$$\alpha = \min\big(1,\ \exp(-U(\theta_L)-K(p_L)+U(\theta_0)+K(p_0))\big), \quad K(p)=\tfrac12 p^TM^{-1}p$$
This is the same MH acceptance rule as before, applied to the joint $(\theta,p)$ system — detailed balance still holds, since leapfrog is reversible and volume-preserving.

**NUTS's addition — adaptive trajectory length**: rather than fixing $L$, NUTS doubles the trajectory in a randomly chosen direction at each iteration, stopping via the **No-U-Turn criterion**: for trajectory endpoints $\theta^-,\theta^+$ with momenta $p^-,p^+$,
$$(\theta^+-\theta^-)\cdot p^- \ge 0 \quad\text{and}\quad (\theta^+-\theta^-)\cdot p^+ \ge 0$$
Doubling continues while both hold; the moment either goes negative, the trajectory has curved back on itself and doubling stops. The next sample is drawn uniformly from the valid (slice-sampling-qualified) points generated across the whole trajectory — not just its endpoint — which preserves detailed balance exactly despite the trajectory's length varying dynamically.

**Step size $\epsilon$**: tuned automatically during warmup via dual averaging, targeting a specified acceptance rate (e.g. 0.95).

**Summary**: Steps 1–4 (evaluating $\log\pi(\theta)$) are unchanged for any sampler. NUTS replaces only the proposal mechanism — blind random-walk $\to$ gradient-driven leapfrog trajectory with adaptive length — while the correctness guarantee (detailed balance, ergodicity, convergence) established for basic Metropolis-Hastings continues to hold unchanged.


## Particle Filter and PMCMC

### The Particle Filter, Conceptually

The particle filter maintains a set of $N$ "particles" — each representing one plausible value of the latent $\ln\sigma_t^2$ at time $t$. At each time step:

1. **Propagate**: each particle's $\ln\sigma_t^2$ is advanced forward one step using the SV recursion, with its own independently sampled $\eta_t$.
2. **Weight**: each particle is assigned a weight based on how likely the actual observed return $r_t$ is, given that particle's own implied $\sigma_t$ — particles whose implied volatility poorly explains the actual return receive low weight; particles that explain it well receive high weight.
3. **Resample**: particles are resampled (with replacement, proportional to their weights) — low-weight particles are discarded, high-weight particles are duplicated, concentrating the particle set on the region of latent-volatility-space consistent with what's actually been observed so far.

### The Particle Filter, Formally

At time $t$, given $N$ particles $\{\ln\sigma_{t-1}^{2,(n)}\}_{n=1}^N$ (each already weighted/resampled from the previous step):

**Propagate** each particle forward using the SV recursion, drawing a fresh, independent shock per particle:
$$\ln\sigma_t^{2,(n)} = \mu_h + \phi\left(\ln\sigma_{t-1}^{2,(n)} - \mu_h\right) + \eta_t^{(n)}, \qquad \eta_t^{(n)} \overset{iid}{\sim} N(0,\sigma_\eta^2)$$

**Weight** each particle by the likelihood of the actual observed return, given that particle's own implied $\sigma_t^{(n)}=\exp(\tfrac12\ln\sigma_t^{2,(n)})$:
$$w_t^{(n)} \propto p\left(r_t \mid \sigma_t^{(n)}\right) = \frac{1}{\sigma_t^{(n)}}\phi\left(\frac{r_t-\mu_r}{\sigma_t^{(n)}}\right)$$
(the Gaussian density, since $r_t=\mu_r+\sigma_t\epsilon_t$, $\epsilon_t\sim N(0,1)$), normalized so $\sum_n w_t^{(n)}=1$.

**Resample**: draw $N$ new particles from $\{\ln\sigma_t^{2,(n)}\}$ with probability proportional to $w_t^{(n)}$ (with replacement) — this concentrates particles in regions consistent with the data observed so far, preventing the particle set from degenerating onto a small number of increasingly-implausible values as $t$ grows (a failure mode known as weight degeneracy).



### PMCMC: Plugging This Into Metropolis-Hastings

Recall from MCMC (above) that the acceptance ratio only ever needs $p(\text{data}\mid\theta')/p(\text{data}\mid\theta)$. PMCMC's key theoretical result (Andrieu, Doucet & Holenstein, 2010 — the "pseudo-marginal" argument) is that substituting the particle filter's *unbiased estimate* $\hat{p}(\text{data}\mid\theta')$ in place of the true (uncomputable) $p(\text{data}\mid\theta')$ **still yields a Markov chain with exactly the correct target posterior** — not an approximation to it. This holds regardless of how noisy any single estimate $\hat{p}$ is (though a noisier estimate, e.g. from too few particles, will make the resulting chain mix more slowly, hurting practical efficiency even though correctness is preserved).

This is why PMCMC's overall structure inherits everything already established for plain MCMC (detailed balance, ergodicity, burn-in) unchanged — the only thing that's different is *how* the likelihood term inside the acceptance ratio is obtained: computed exactly (EGARCH) versus estimated via a particle filter (SV) — mirroring exactly how NUTS was framed as "MCMC, plus a smarter proposal" rather than a separate framework.