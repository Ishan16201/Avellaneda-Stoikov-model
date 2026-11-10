# Mathematical derivations

This document collects the models and estimators implemented in the
repository. "Theory" below refers to analytical results; "simulation"
references the event-driven synthetic exchange; "empirical" refers to
quantities estimated from simulated data.

## 1. Avellaneda–Stoikov baseline

Midprice follows driftless Brownian motion `dS = sigma dW`. The market maker
has CARA utility `-exp(-gamma x)` and quotes bid/ask offsets `delta_b`,
`delta_a` from the midprice. Fill arrivals at each quote are independent
Poisson processes with intensity

```
lambda(delta) = A exp(-k delta),   delta >= 0
```

Under the standard first-order asymptotic expansion in `gamma sigma^2 tau`
(time remaining `tau = T - t`), with

```
c = log(1 + gamma/k) / gamma        (limit gamma -> 0: 1/k)
r = S - q gamma sigma^2 tau
h = c + gamma sigma^2 tau / 2
bid = r - h,   ask = r + h
```

Equivalently, offsets relative to `S` are

```
delta_b = c + (1/2 + q) gamma sigma^2 tau
delta_a = c + (1/2 - q) gamma sigma^2 tau
```

This is the classical approximation, not an exact finite-horizon HJB
solution. `A` cancels from the optimal offsets; it is retained as a
configuration input for expected-fill reporting. Quotes are rounded outward
to integer ticks and clamped to avoid crossing the observed opposite best.

## 2. Hawkes market process

Six event channels (limit buy/sell, cancel buy/sell, market buy/sell). The
multivariate exponential Hawkes intensity for channel `i` is

```
lambda_i(t) = mu_i + sum_j alpha_ij beta_ij
              sum_{s<t, event j} exp(-beta_ij (t - s))
```

implemented in intensity-state form: each accepted event of type `tau`
adds `alpha_{i,tau} beta_{i,tau}` to excitation state `(i,tau)`, and every
state entry decays as `exp(-beta_ij dt)`. `lambda_i = mu_i + sum_j state_ij`.

Simulation uses Ogata thinning: from the current post-jump intensity bound
`Lambda`, draw `dt ~ Exp(Lambda)`; at the candidate time decay the state,
recompute `lambda`, and accept with probability `lambda/Lambda`. Rejected
candidates repeat from the decayed state. The event type is drawn from the
normalized post-decay intensities. The same kernel
(`simulate_one_hawkes_path`) is compiled for CPU and CUDA; path `p` uses
SplitMix64 stream `path_rng_state(seed, p)`.

Stationarity requires the branching matrix `alpha` (row sums over children
per parent) to have spectral radius below one; `validate_stationary`
enforces this before generation.

## 3. Queue-tracker estimator

For each own order, `quantity_ahead` estimates displayed volume ahead in
price-time priority and `quantity_behind` volume behind. Depth snapshots are
diffed per `(side, price)` level:

- Increase not explained by a pending own arrival is credited to
  `quantity_behind` of orders at that level.
- Decrease not explained by a public trade or own action is a cancellation;
  it is allocated ahead/behind by `CancellationAllocation`
  (`BehindFirst`, `AheadFirst`, `ProRata`).
- Public trades deplete `quantity_ahead` first, then `quantity_behind`,
  excluding quantities already reported as own fills at the same
  `(timestamp, price, passive side)`.

These are estimators over aggregate displayed depth; anonymous
cancellation identity is unobservable, so allocations are modeling
assumptions, not exchange facts.

## 4. Hawkes-aware adverse-markout adjustment

A regularized three-class softmax trained only on training simulations
estimates `P(direction at horizon H | lambda, OFI)` where

```
OFI = (limit_buy - limit_sell - cancel_buy + cancel_sell
       + market_buy - market_sell) / sum_i lambda_i
```

features are `log(1 + lambda_i)` plus OFI. Conditional average up/down move
magnitudes are training means. The policy adds a side-specific premium

```
penalty_bid = P(fill bid within H | queue, opposing intensity)
              * E[(S_t - S_{t+H})_+ | state]
penalty_ask = P(fill ask within H | queue, opposing intensity)
              * E[(S_{t+H} - S_t)_+ | state]
```

to the Avellaneda–Stoikov offsets. The fill probability uses the deterministic
expected-queue-depletion approximation

```
P = 1 - exp(-lambda_opposing * mean_size * H / (ahead + own/2))
```

This is a heuristic markout adjustment, not a structural Hawkes-HJB
solution, and the training target (midprice direction) is not the
fill-conditional markout.

## 5. Reduced-state dynamic program

Finite-horizon DP over integer inventory `q` and time-to-go `tau` with
terminal penalty `phi q^2` and optional running penalty `c q^2`:

```
V(0,q) = -phi q^2
dV/dtau = -c q^2
  + max(0, max_db lambda(db) [db h C + V(tau,q+h) - V(tau,q)])
  + max(0, max_da lambda(da) [da h C + V(tau,q-h) - V(tau,q)])
lambda(delta) = A exp(-k delta)
```

Quotes are chosen from a finite offset grid with an explicit stand-aside
action; the inventory bound is hard. Backward updates enforce a monotone CFL
bound on fill, activity-jump, and upwind rates. In the zero-penalty,
no-activity limit the discrete optimum approaches the risk-neutral offset
`delta = 1/k`; this validates the expected-revenue Hamiltonian but is *not*
equivalent to the CARA Avellaneda–Stoikov HJB (the objectives differ).

An optional one-dimensional activity approximation uses
`dx/dt = beta(mu - x)` with jumps of size `eta` at rate `x`, scaling quote
arrival rates by `x/mu` on `[0, hawkes_activity_max_per_second]`. It is an
exploratory reduced state, not a multivariate Hawkes solution.

## 6. Differentiable surrogate

A smooth sample-average objective `J(theta) = (1/N) sum_path R_path(theta)`
over causal exogenous `SurrogatePath` observations. 22 parameters drive
sigmoid-bounded quote offsets, fill probabilities, expected sizes, and
cancel probabilities (see README for the exact feature map). The per-step
reward is

```
r = C (e_bid delta_bid + e_ask delta_ask + q delta_mid
       - e_bid m_bid - e_ask m_ask)
    - fee (e_bid + e_ask) - c dt q_next^2
R_path = sum_t r_t - phi q_terminal^2
```

with `e_side = max_qty p_side sigmoid((L -+ q)/T)` capacity gates. Gradients
use forward-mode dual numbers propagated through all arithmetic (22
derivatives per value). Optimizer: gradient ascent on `J` with global-norm
clipping, box projection to `+-parameter_absolute_limit`, Adam or SGD, and
constant/linear/cosine schedules. Evaluation is a sample average over a
fixed batch; it is not an exact gradient of the discrete event simulator.

## 7. Score-function policy optimizer

For the exact simulator, the categorical actor's episode return `R` (marked
P&L less configured inventory penalties) is optimized with the
score-function estimator

```
grad J ~= (1/B) sum_episode ((R - baseline)/reward_scale)
          sum_t grad log pi_theta(a_t | s_t)
```

Only the policy likelihood is differentiated; the exchange's discrete
events are not. A smoothed prior-epoch reward baseline, reward scaling,
gradient clipping, Adam, schedules, bounded weights, and validation early
stopping with checkpointing are implemented. Train/validation/OOS seed
partitions are enforced disjoint.

## 8. Benchmark statistics

Per policy/scenario/split over `n` episode P&L samples `x_i`:

- mean `x̄`, sample standard deviation `s`, Student-t confidence interval
  `x̄ +- t_{n-1,1-alpha/2} s / sqrt(n)` (t quantile from the regularized
  incomplete beta function).
- Sharpe: mean/std of timer-sampled equity returns, annualized with a
  configurable positive capital base, zero risk-free rate, 365.25-day year.
- Maximum drawdown from the equity path; loss VaR at nearest-rank
  confidence quantile; expected shortfall = mean loss at or beyond VaR.
- Fill rate, spread capture (pre-execution-midpoint execution cost),
  adverse-selection markout `AS_h = s (mid(t+h) - exec_price)` (positive is
  favorable), implementation shortfall vs submission midpoint, effective /
  realized spread and price impact decomposition
  `effective = realized + 2*impact`, queue wait, inventory
  `integral |q| dt` and concentration `integral q^2 dt/(L^2 T)`, turnover,
  and net transaction costs.

Paired comparisons match policies on identical seeds within a scenario and
report the distribution of per-seed P&L differences with the same t
interval. Pooled "ALL" rows aggregate episodes across scenarios for
summaries only; paired inference is per scenario because only same-scenario
runs share the underlying random tape.

## 9. Performance model notes

- Exchange episodes are CPU-only; independent episodes parallelize with
  near-linear scaling bounded by allocator and memory bandwidth.
- Batched Hawkes generation is embarrassingly parallel across paths
  (one thread or CUDA thread per path); the kernel is `exp`-bound (~36
  exponential decays per candidate event).
- Surrogate gradient evaluation parallelizes over contiguous path chunks
  with deterministic merge order; results are bit-identical for a fixed
  worker count up to last-bit summation order.
