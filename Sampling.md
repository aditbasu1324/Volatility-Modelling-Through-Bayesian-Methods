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
