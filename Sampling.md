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
