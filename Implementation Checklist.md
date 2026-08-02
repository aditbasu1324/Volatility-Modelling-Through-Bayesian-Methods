# Common Bug Classes When Repurposing This Codebase

A checklist of recurring issues found while building this project, worth checking whenever extending to a new model (e.g. SV) or repurposing this code elsewhere.

## 1. Posterior Draw Count (S) Mismatches
Different fitting stages (e.g. regression vs. sequential refits) can produce posteriors with different numbers of draws. Combining per-draw arrays across stages (e.g. `np.concatenate`) then fails, or silently misaligns.
**Fix pattern**: resample the larger set down at initialization, using one consistent index for both parameters and any carried-forward state.

## 2. Aggregation Order (Method 1 vs. Method 2)
Averaging across draws *after* a nonlinear transform (sqrt, a metric formula) instead of *before* gives a different, often biased, result (Jensen's inequality). Also applies to averaging $\sigma$ instead of $\sigma^2$ before squaring.
**Check**: for any new metric, confirm which order its own *definition* requires — not which is more convenient to code.

## 3. Stale / Mismatched Date-Index Alignment
Two series with different NaN-trimming patterns (rolling-window edges, `.dropna()`, different `align_series` calls) can produce shape mismatches, or worse, same-length arrays that don't correspond date-for-date.
**Check**: always confirm which specific alignment call feeds which downstream variable; don't assume two "aligned" series share the same dates without checking.

## 4. Variable Name Collisions / Stale Reassignment
The same variable name reused for genuinely different things at different pipeline stages (e.g. "raw concatenated" vs. "aligned, final"). Whichever cell runs *last* silently wins, regardless of logical order.
**Fix pattern**: use distinct names for raw vs. processed versions of the same conceptual object.

## 5. Numerical Overflow in Custom Math Functions
Functions involving gamma functions, large exponents, etc. (e.g. `scipy.special.gamma`) can silently overflow for extreme parameter values.
**Fix pattern**: use log-space computation (`gammaln`) or a bounded interpolation grid instead of the raw closed-form formula.

## 6. Missing Clipping in Hand-Rolled Recursions
A recursion re-implemented in plain NumPy (for prior checks, forward simulation) can miss safety clipping present in the "official" JAX/NUTS version.
**Fix pattern**: consolidate into one shared, vectorized function used by all call sites, rather than maintaining parallel implementations.

## 7. Import / Naming Alias Inconsistency
Using two different names for the same module (e.g. `stats` vs. `scipy_stats`) across different code blocks causes `NameError` once consolidated into files with a single import convention.
**Fix pattern**: standardize on one alias per library, used consistently everywhere.

## 8. Forgetting to Capture Per-Draw State During an Expensive Run
Not saving intermediate per-draw arrays (e.g. via `numpyro.deterministic`, or appending to a history array inside a hand-written MCMC loop) forces a wasteful re-run just to retrieve data that could have been captured for free the first time.
**Check**: for any new expensive sampling loop, confirm all per-draw outputs needed later are captured during the original run.

## 9. Functions Defined After Their First Use
Code that worked in an ad-hoc notebook (relying on cell-execution order) can break once moved into properly-ordered library files, if a function or default-argument value is referenced before it's defined.
**Fix pattern**: define config/constants first, then functions that reference them as defaults; or resolve `None` defaults inside the function body instead of at definition time.

## 10. Calling a Stale/Superseded Function Signature
After consolidating or fixing a function, old call sites elsewhere may still use the previous signature — a version-drift risk specific to iterating heavily on the same functions over a long session.
**Check**: after any function signature change, search for all call sites, not just the one you're currently editing.

## 11. Missing Directory Creation Before File Saves
`open(..., "wb")` fails with "no such file or directory" if the target folder doesn't exist yet.
**Fix pattern**: always call `Path(...).mkdir(parents=True, exist_ok=True)` before the first save into a new results subfolder.

---

**Most likely to recur for a new model (e.g. SV)**: #1 (posterior size vs. refit size), #5/#6 (a new recursion needs its own stability/clipping check), #8 (confirm any new hand-written sampling loop captures per-draw state consistently).