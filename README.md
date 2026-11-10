# Project 3 — Hawkes LOB integration

This repository provides a small C++20 integration layer and a basic agent
demonstration over the completed
[Hawkes-driven limit-order-book simulator](../Hawkes-driven-limit-order-book-simulator.).
It reuses the simulator's deterministic matching engine, Hawkes generator,
strategy broker, accounting, and event evaluator; it does not copy or replace
those components.

The simulator remains a separate dependency. CMake defaults
`PROJECT3_LOB_SOURCE_DIR` to the sibling repository folder and can be pointed
elsewhere when configuring.

## Interfaces

- [`project3::TradingAgent`](include/project3/market_agent.hpp) receives
  callbacks at start, after each market event, on public trades, on fills,
  optional timer ticks, and at stop.
- [`project3::MarketView`](include/project3/market_agent.hpp) is a value
  snapshot containing timestamp, all displayed price levels and quantities,
  best bid/ask, midprice, spread, current inventory/cash, and Hawkes
  intensities. It contains no future events or RNG access.
- [`project3::OrderGateway`](include/project3/market_agent.hpp) submits
  limit/market orders and cancels by returned order ID.
- [`project3::ExchangeAdapter`](include/project3/exchange_adapter.hpp) bridges
  those interfaces to the existing `StrategyEvaluator`.
- [`project3::MarketMakerState`](include/project3/market_agent.hpp) exposes the
  broker's authoritative cash, signed inventory, mark-to-market equity,
  realized/unrealized P&L, gross charged fees, rebates, and maximum inventory
  exposure, plus order-level execution notifications.
- [`project3::QueueTracker`](include/project3/execution_layer.hpp) tracks each
  own order's estimated quantity ahead, remaining quantity, and quantity
  behind. A confirmed replacement receives a new ID and a new queue sequence.

The broker remains the accounting source of truth. Fees are debited exactly
once per fill; `total_fees_currency` is net fees (gross charged fees less
rebates), while `total_charged_fees_currency` and
`total_rebates_currency` report the components separately. Equity follows
`cash + signed_inventory * mark_price * currency_per_tick`.

## Queue assumptions

The exchange internally applies price-time priority exactly. The agent-side
queue tracker is an estimate based on aggregate displayed depth, public trades,
and own fill/cancel notifications; it does not expose hidden order identities.
It assumes orders arrive immediately for initial queue placement. Positive
unexplained depth changes are treated as additions behind the order. A negative
depth change not explained by a public trade or known own action is treated as
a cancellation and allocated according to `CancellationAllocation`:

- `BehindFirst` (default) is conservative about queue improvement.
- `AheadFirst` is optimistic.
- `ProRata` splits the inferred cancellation proportionally.

Real feeds generally do not identify which anonymous order cancelled, and
multiple changes between observations can be indistinguishable. These
allocations are modeling assumptions, not facts about cancellation position.
The tracker labels queue positions estimated and should not be treated as exact
exchange truth, particularly with nonzero latency.

The underlying evaluator processes the event stream chronologically. At a
market-event callback, intensities include the excitation from the event just
observed. At timer and stop callbacks, they are decayed to callback time using
only observed events. The agent never receives the future event tape,
matching-engine object, or random generator.

Execution notifications report each fill separately, including remaining
quantity and order status, so partial fills remain explicit. Orders submitted
by Project 3 receive IDs from a descending high-ID range, separate in ordinary
runs from the simulator's ascending generated IDs.

## Build and run

From this repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/project3_demo
```

If the simulator is elsewhere:

```sh
cmake -S . -B build \
  -DPROJECT3_LOB_SOURCE_DIR=/path/to/Hawkes-driven-limit-order-book-simulator
```

The demo is intentionally a basic two-sided passive quoting agent, not an
Avellaneda–Stoikov or optimized policy. It shows market snapshots, intensity
access, order submission, fill notifications, and cancellation at shutdown.

## Avellaneda–Stoikov baseline

The first policy implementation is available through
[`project3/avellaneda_stoikov.hpp`](include/project3/avellaneda_stoikov.hpp).
Run its reproducible sample with:

```sh
./build/avellaneda_stoikov_demo
```

The implemented model uses a driftless Brownian midprice,
`dS = sigma dW`, constant volatility, CARA utility with risk aversion
`gamma`, and independent unit-fill Poisson arrivals at each quote with
`lambda(delta) = A exp(-k delta)`. Time remaining is `tau = max(T-t, 0)`.
Inventory is measured in lots, prices in ticks, volatility in ticks per
square-root second, and gamma in inverse tick-lots. The quote-offset function
implements the standard first-order Avellaneda–Stoikov approximation:

```text
c       = log(1 + gamma/k) / gamma       (limit gamma -> 0: 1/k)
r       = S - q gamma sigma^2 tau
delta_b = c + (1/2 + q) gamma sigma^2 tau
delta_a = c + (1/2 - q) gamma sigma^2 tau
bid     = S - delta_b
ask     = S + delta_a
```

The classical approximation sets the quote center to `r` and the optimal
half-spread to `h = c + gamma sigma^2 tau / 2`. Substituting
`bid = r-h` and `ask = r+h` gives the displayed offsets relative to `S`.
Equivalently, the full spread is `2 log(1 + gamma/k)/gamma +
gamma sigma^2 tau`. The scale `A` sets theoretical fill rates but cancels
from this first-order optimal offset. `log1p` and its small-gamma limit are
used for numerical stability; zero risk aversion is handled continuously.
Prices are rounded outward to integer ticks. The agent also
prevents crossing the currently observed opposite best quote, respects a
configurable inventory limit, and caps each side's quote size by both the
configured size and remaining inventory capacity.

The inventory-dependent spread expressions are the classical first-order
approximation, not an exact solution of the full finite-horizon HJB. The
implementation does not include drift, stochastic volatility, market impact,
fees/rebates in quote optimization, or Hawkes-dependent fills. The supplied
`A` and `k` are configuration inputs, not estimates calibrated from the
synthetic order flow. The analytical model assumes unit fills and a smooth
price; configured quote sizes and tick rounding are execution-layer controls,
not part of that derivation.

`evaluate_avellaneda_stoikov` returns:

- Final net P&L, gross spread capture, residual inventory P&L, net fees,
  turnover, and broker-measured maximum drawdown. Attribution reconciles as
  `total P&L = spread capture + inventory P&L - net fees`. Spread capture uses
  the broker's pre-execution-midpoint execution-cost measure.
- Quote fill rate = unique submitted quote orders receiving at least one fill,
  divided by submitted quote orders.
- Inventory samples at start and fixed evaluator timer ticks.
- Annualized Sharpe from timer-sampled equity returns, with a configurable
  positive capital base, zero risk-free rate, and a 365.25-day year.
- Per-side theoretical expected fills and empirical fill notifications per
  active order-second. The model expectation integrates `A exp(-k delta)`
  over observed quote exposure; empirical rates are measured from simulator
  execution reports. These need not agree: the simulator's Hawkes arrivals,
  random order sizes, FIFO queue position, tick rounding, inventory controls,
  quote updates, and latency are not the analytical Poisson fill model.

The reported inventory P&L is the residual gross P&L after spread capture,
rather than a standalone estimate of a latent markout component. Empirical
order exposure is based on causal agent observations and acknowledgements, so
nonzero reporting/cancellation latency can affect the comparison.

## Hawkes-aware policy and held-out comparisons

[`project3/hawkes_policy.hpp`](include/project3/hawkes_policy.hpp) exposes all
six current event intensities by name: bid/ask limit additions, bid/ask
cancellations, and market buys/sells. The model is the simulator's exponential
Hawkes state, with
`lambda_i(t) = mu_i + sum_j alpha_ij beta_ij sum_{s<t, event j} exp(-beta_ij(t-s))`.
The predictable pre-event state generates each event; policy callbacks receive
the post-observation state after the current event's excitation jump.

The directional model is empirically trained from supplied training
simulations, not fitted to evaluation labels. Its features are `log(1+lambda_i)`
for all six channels plus normalized signed order-flow imbalance:

```text
OFI = (limit_buy - limit_sell - cancel_buy + cancel_sell
       + market_buy - market_sell) / sum_i(lambda_i)
```

A regularized three-class softmax estimates the probability of an increase,
decrease, or no midprice change at a configurable forecast horizon. Conditional
average up/down move magnitudes are estimated from training samples. Future
midprices are used only to label training and held-out evaluation examples;
policy prediction uses current intensities alone.

The Hawkes-aware policy starts with the same Avellaneda–Stoikov offsets. It
then adds a side-specific expected adverse markout premium:

```text
bid penalty = P(fill at bid within H | queue, current flow)
              * E[(S_t - S_{t+H})_+ | current Hawkes state]
ask penalty = P(fill at ask within H | queue, current flow)
              * E[(S_{t+H} - S_t)_+ | current Hawkes state]
```

The directional expectations are the fitted probabilities multiplied by the
training-estimated conditional move sizes. The fill probability uses the
opposing market-order intensity, configured mean market-order size, observed
estimated quantity ahead, and half the own remaining size. It uses a
deterministic expected-queue-depletion approximation
`1-exp(-lambda_opposing * mean_size * H / (ahead + own/2))`; it deliberately
does not assume anonymous cancellations ahead can be identified. Therefore
Hawkes-aware quotes widen only on the side with estimated adverse markout, and
the amount is tied to empirically estimated price moves and a queue-based
fill probability rather than an arbitrary intensity multiplier. Quote updates
still respect inventory limits and do not cross the observed opposite best.

`compare_market_making_policies` trains on training runs, verifies evaluation
seeds are disjoint and evaluation Hawkes parameters are absent from training,
then compares classical Avellaneda–Stoikov, Hawkes-aware quoting, and a simple
symmetric maker on paired evaluation seeds. It reports per-policy mean and
standard deviation of P&L, fill rate, drawdown, turnover and inventory, plus
held-out direction accuracy and log loss against a constant training-class
base-rate predictor. The latter is the baseline for determining whether the
intensity/OFI state adds out-of-sample directional information. The caller supplies explicit
training and held-out parameter configurations and independent seeds; results
remain specific to the simulator's synthetic marks and FIFO mechanics.

This is not a structural Hawkes-HJB solution. Softmax probabilities are
empirical and can be miscalibrated under distribution shift; a held-out
configuration comparison is essential. The queue-fill approximation ignores
limit-addition priority changes, unknown cancellation allocation, and latency.
The training target is midprice direction at a fixed horizon, not the exact
conditional markout given a future fill, so adverse-markout penalties are an
approximation rather than a guarantee of improved P&L.

Run the end-to-end train/held-out comparison example with:

```sh
./build/hawkes_policy_comparison
```

## Reduced-state stochastic control

[`project3/reduced_state_solver.hpp`](include/project3/reduced_state_solver.hpp)
provides a finite-horizon dynamic program over integer inventory and time. The
objective is expected marked wealth with a terminal quadratic inventory penalty,
not exponential utility:

```text
V(0,q) = -phi q^2

dV/dtau = -c q^2
  + max(0, max_db lambda(db) [db h C + V(tau,q+h) - V(tau,q)])
  + max(0, max_da lambda(da) [da h C + V(tau,q-h) - V(tau,q)])

lambda(delta) = A exp(-k delta)
```

Here `tau` is time-to-go, `h` is the configured inventory step in lots,
`C` is currency per price tick, `phi` is the terminal penalty, and `c` is an
optional running inventory penalty. The state inventory is bounded by the
configured hard limit; outward fills are disabled at each boundary. Bid and ask
controls are optimized separately, so simultaneous opposite-side fills are
omitted as an order `dt^2` event. Quotes are selected from a finite offset grid.
Each side also has an explicit stand-aside action with zero arrival rate, so a
quote with negative continuation value is not forced.
The numerical value and offsets are linearly interpolated in time, inventory,
and (when enabled) activity; enabled/disabled boundary actions use the nearest
grid state. Queries outside the state domain clamp to its boundary.

The explicit backward-in-time update rejects grids that violate its monotone
CFL bound, based on maximum fill, activity-jump, and upwind transport rates.
Increase `time_steps` when that check fails. The optional one-dimensional
Hawkes approximation uses
`dx/dt = beta(mu-x)` between generic events and jumps of size `eta` at rate
`x`; configured ask and bid fill intensities are scaled by `x/mu`. The activity
domain is `[0, hawkes_activity_max_per_second]`; jump transitions are clipped
at its upper boundary. This does not model a full multivariate Hawkes process
or connect individual Hawkes events structurally to quote fills: activity
transitions and quote fills are separate transitions. It is an exploratory
reduced state, not an exact Hawkes-HJB solution.

This objective has linear marked-wealth utility, so Brownian price risk does
not add a term to the relative-value HJB. In the zero-penalty, no-activity
case, the optimal discrete quote approaches the risk-neutral Avellaneda–Stoikov
limit `delta = 1/k`. That limiting case validates the expected spread-revenue
Hamiltonian; it does not establish equivalence to the CARA/exponential-utility
Avellaneda–Stoikov HJB. The quadratic terminal penalty is a separate
risk-sensitive control specification.

Run the convergence and resource report with:

```sh
./build/reduced_state_convergence
```

`run_grid_convergence_study` reports value and quote changes at the initial
flat-inventory state between each supplied grid and its predecessor, distance
from the zero-risk analytical offset when applicable, the value change from
doubling the inventory limit, solver runtime, and estimated payload memory
for the value/policy arrays. Runtime excludes the auxiliary doubled-boundary
solve; memory excludes vector/container overhead. These are grid diagnostics,
not error bounds. Refine time, inventory, and action grids separately to
identify which discretization dominates, and repeat with a narrower and wider
inventory domain to expose boundary effects. The activity-domain clipping
requires a separate activity-grid/domain sensitivity check when the Hawkes
state is enabled.

The dynamic-programming tests cover the risk-neutral analytical offset,
expected-revenue value, terminal condition, inventory skew and hard limits,
interpolation, activity dependence, CFL rejection, grid convergence trends,
and resource diagnostics. No policy-gradient or exact gradient claims are made
for this discrete event/action solver.

## Differentiable market-making surrogate

[`project3/differentiable_surrogate.hpp`](include/project3/differentiable_surrogate.hpp)
implements a separate, smooth sample-average model. It does not differentiate,
wrap, or replace the event-driven exchange. A `SurrogatePath` contains causal,
exogenous per-decision observations: market-buy/sell intensities, estimated
queue ahead, expected side-specific adverse markout, midprice return, and
decision duration. The sample objective is
`J(theta) = (1/N) sum_path R_path(theta)`.

The surrogate has 22 learnable parameters: bid/ask quote intercepts,
inventory/queue/market-intensity/adverse-selection sensitivities, six
event-specific Hawkes-intensity sensitivities, spread and side-depth
sensitivities, volatility and time-remaining sensitivities, side-specific size
and cancel-propensity logits, and a midprice-feature sensitivity. For
inventory `q`, limit `L`, queue `Q`, intensity `lambda`, and adverse cost `m`,
the quote features include:

```text
z_bid = theta_0 + theta_2 q/L + theta_3 Q_bid/Q_scale
        + theta_4 log(1+lambda_sell)/I_scale + theta_5 m_bid/M_scale
        + sum_i theta_(6+i) log(1+hawkes_i)/I_scale
        + theta_12 spread + theta_13 log(1+bid_depth)
        + theta_15 log(1+volatility) + theta_16 time_remaining
        + theta_21 midprice_feature
z_ask = theta_1 - theta_2 q/L + theta_3 Q_ask/Q_scale
        + theta_4 log(1+lambda_buy)/I_scale + theta_5 m_ask/M_scale
        + sum_i theta_(6+i) log(1+hawkes_i)/I_scale
        + theta_12 spread + theta_14 log(1+ask_depth)
        + theta_15 log(1+volatility) + theta_16 time_remaining
        + theta_21 midprice_feature
delta_side = delta_min + (delta_max-delta_min) sigmoid(z_side)

p_bid = sigmoid(b0 + b_delta delta_bid + b_queue Q_bid/Q_scale
                + b_intensity log(1+lambda_sell)/I_scale)
p_ask = sigmoid(b0 + b_delta delta_ask + b_queue Q_ask/Q_scale
                + b_intensity log(1+lambda_buy)/I_scale)
```

`b0`, `b_delta`, `b_queue`, and `b_intensity` are fixed calibration inputs in
`DifferentiableSurrogateConfig`; `p_side` is a smooth fill-probability
approximation per decision interval, not an exchange fill probability
guarantee. Expected fills and inventory are propagated as:

```text
e_bid = max_qty p_bid sigmoid((L-q)/T)
e_ask = max_qty p_ask sigmoid((L+q)/T)
q_next = q + e_bid - e_ask
```

The sigmoid capacity gates and quadratic inventory penalties are smooth
approximations, not hard inventory limits. Expected side fills are fractional;
there is no queue depletion, partial-fill event sequence, price-time matching,
latency, tick rounding, cancellation, or random execution size. `m_bid` and
`m_ask` are supplied conditional adverse markout costs; the surrogate does not
infer them from hidden or future exchange state. These costs should represent
fill-conditional adverse markout not already included in the separate
inventory-held `delta_mid` term, to avoid double counting.

The per-step reward is:

```text
r = C(e_bid delta_bid + e_ask delta_ask + q * delta_mid
      - e_bid m_bid - e_ask m_ask)
    - fee (e_bid + e_ask) - c dt q_next^2
R_path = sum_t r_t - phi q_terminal^2
```

`C` is currency per tick, `fee` is currency per executed lot, `c` is the
running inventory-penalty coefficient, and `phi` is the terminal penalty.
Thus reward includes expected spread capture, inventory mark-to-market,
adverse-selection cost, fees, and smooth inventory risk costs. It is an
expected-reward objective on a fixed batch, not a guarantee of exchange P&L.

`theta_17`/`theta_18` determine smooth expected bid/ask size and
`theta_19`/`theta_20` determine smooth cancel probability. These outputs are
expected quantities/probabilities, not literal exchange order actions.

Gradients use a small in-project forward-mode dual-number implementation for
all 22 policy parameters; no external autodiff dependency is required.
`check_surrogate_gradient` compares it with central finite differences (using a
one-sided interval at parameter bounds) and reports the requested relative
error. Integration tests exercise all feature and parameter paths at multiple
parameter scales and compare gradient updates with finite differences.
Gradient ascent with global-norm clipping, box-projected parameter constraints,
Adam or plain gradient descent, and constant/linear/cosine learning-rate
schedules is available. The optimizer is full-batch and deterministic;
`make_surrogate_training_batch(seed, ...)` reproduces its fixed synthetic
exogenous observations in the same standard library/runtime environment.

Run the optimization and gradient-check example with:

```sh
./build/differentiable_surrogate_demo
```

Use the returned parameters with `smooth_surrogate_decision` to obtain
continuous offsets and expected fills for a causal observation. Evaluate any
resulting trading policy separately through the existing
`ExchangeAdapter`/`StrategyEvaluator` and exact matching engine, using held-out
seeds and market configurations. The simulated exchange remains the
ground-truth evaluation environment; surrogate gradients are not exact
gradients of its discrete event process, and surrogate reward improvement does
not imply improved simulated or live-market performance.

## Policy optimization and exact-simulator benchmarks

[`project3/policy_optimization.hpp`](include/project3/policy_optimization.hpp)
provides a two-track training and evaluation framework. The simulator-facing
categorical actor observes 18 causal features: normalized inventory, midprice
displacement from its initial observed value, spread, displayed bid/ask depth,
the six current Hawkes event intensities, an online realized-volatility
estimate, estimated queue-ahead on each side, time remaining, and whether a
quote is already open on each side. At each decision time it selects one of
eight masked actions per side: retain, cancel, or one of three quote offsets
combined with one of two configured sizes. Actions that exceed inventory
capacity or conflict with the current open-order state are masked. Replacing a
quote goes through the normal exchange order path and does not inherit queue
priority.

The exact-simulator optimizer uses an episode-return score-function estimator:

```text
J(theta) = E[R]
grad J ~= (1/B) sum_episode ((R - baseline) / reward_scale)
           sum_t grad log pi_theta(action_t | observed_state_t)
```

`R` is exact marked P&L from the simulator, less any configured running and
terminal inventory penalties. The estimator differentiates the sampled
categorical policy likelihood only; it does not differentiate the exchange's
discrete events, matching, or fill reports. A smoothed prior-epoch reward
baseline, configurable reward scale, global gradient clipping, Adam, learning
rate schedules, bounded policy weights, and validation early stopping are
implemented. Validation-selected parameters are checkpointed with optimizer
state and selected hyperparameters; optional CSV files record training and
validation learning curves. Training, validation, and OOS seed partitions are
checked for overlap, and scenario labels can represent different market
parameters.

The surrogate optimizer is separate: it collects causal snapshots from
no-agent exact-simulator runs, optimizes the smooth fractional-fill objective,
and reports that objective separately from exact simulator training,
validation, and OOS results. Its gradients remain gradients of the surrogate
alone. Both approaches are benchmarked on held-out exact simulator episodes
against the Avellaneda–Stoikov and reduced-state DP baselines. OOS summaries
include P&L and objective reward, variability, drawdown, fill rate, and maximum
inventory; training rewards are not presented as OOS results.

Run the compact end-to-end example with:

```sh
./build/policy_optimization_demo
```

The example writes model checkpoints and CSV learning curves under `build/`.
The example varies held-out Hawkes flow and price-placement marks. The
synthetic market configuration has no direct volatility-regime parameter:
volatility is estimated online from observed prices, so generalization across
volatility is only tested indirectly through market/mark scenarios, not via an
explicit simulator sigma control.

## Market-quality and stress analysis

[`project3/market_quality.hpp`](include/project3/market_quality.hpp) adds an
ex-post execution and inventory analysis layer. Pass policy factories and
`MarketStressScenario`s to `run_market_quality_study`; every policy receives
the same market configuration and seeds within a scenario. The recorder wraps
the public `TradingAgent`/`OrderGateway` boundary and only observes causal
views, orders, and execution notifications. It does not change exchange
matching or expose hidden book state.

For markout horizon `h`, let `s=+1` for a market-maker buy and `s=-1` for a
market-maker sell. The reported adverse-selection markout is
`AS_h = s * (mid(t+h) - execution_price)`. **Positive AS is favorable to the
market maker** (the later mid moved in the execution's favor); negative AS is
adverse. Buy and sell fills are summarized separately. Markouts use the first
observed mid at or after `t+h`; executions without such an observation are
reported as right-censored, not silently dropped. Currency markout is
quantity-weighted using the configured currency-per-tick.

Effective spread and price impact use the broker's exact pre-execution
midpoint, reconstructed from its signed execution-cost ledger. Implementation
shortfall instead uses the midpoint observed when the agent submitted that
order; fills with no submission midpoint are excluded from its volume
denominator:

```text
effective spread = 2 s (mid_fill_pre - execution_price)
implementation shortfall = s (execution_price - mid_submit)
realized spread_h = 2 AS_h
price impact_h = -s (mid(t+h) - mid_fill_pre)
effective spread = realized spread_h + 2 price impact_h
```

These are from the market-maker perspective: positive effective/realized
spread is favorable, positive implementation shortfall is a cost, and
positive price impact is an adverse move. Positive implementation shortfall
is a cost relative to the order's submission midpoint, while effective spread
is measured against the midpoint immediately before execution. Queue waiting
time runs from the agent's submission callback to its first fill and therefore
includes configured outbound latency. Fill probability is the fraction of submitted orders with
at least one fill; cancellation rate is successfully cancelled orders divided
by submitted orders.

Inventory duration is `integral |q(t)| dt` in lot-seconds. Inventory
concentration is `integral q(t)^2 dt / (inventory_limit^2 * horizon)`; it can
exceed one if the analysis limit is breached. Inventory paths use observed
event/timer snapshots, and breach duration/interval counts are consequently
resolved at that observation frequency. Broker-reported gross realized P&L,
unrealized P&L, net fees, gross charged fees, rebates, turnover, and net P&L
are retained per run. Net P&L reconciles as
`gross realized + unrealized - (charged fees - rebates)`.

Across each policy/scenario's seed set, the report includes mean and worst
net P&L, sample standard deviation, maximum drawdown, empirical loss VaR and
expected shortfall, fill/cancel rates, queue wait, inventory exposure, and
per-side effective spread, shortfall, and markouts. Loss is `-net P&L`; VaR
uses the nearest-rank empirical confidence quantile and expected shortfall is
the average observed loss at or beyond that quantile. Small seed counts give
coarse tail estimates and should not be mistaken for calibrated risk bounds.

`make_market_stress_scenarios` creates seven paired-seed regimes: higher
market-flow rate/size and wider placement as a volatility proxy; one-sided
market-buy flow; increased Hawkes excitation; reduced initial displayed
liquidity; cancellation-flow surge; wider initial spread; and oversized
aggressive orders to provoke price jumps. Run the demonstration with:

```sh
./build/market_quality_stress
```

The current upstream simulator accepts a fixed Hawkes/mark configuration per
run, not time-dependent parameter schedules. Therefore the named sudden
volatility and spread shocks are modeled as post-shock regimes from simulation
start, not as mid-run shocks at a chosen timestamp. Likewise, volatility and
price jumps are stress proxies generated by event rates/sizes against finite
book depth, not a separately calibrated diffusion/jump process. This limitation
is explicit so these tests do not imply capabilities the exchange does not
provide.

The integration test replays a no-agent run through both the Project 3 facade
and the simulator's original Hawkes adapter, comparing every book snapshot,
event-time intensity vector, timestamp, and public trade. Additional tests
cover reproducibility, partial fills, cancellation, queue depletion and
replacement priority, cash/inventory conservation, fee/rebate separation,
mark-to-market, and execution de-duplication. The simulator's own
test suite remains the source of matching-engine and Hawkes unit/property
coverage.

## Reproducible policy benchmark

`project3::run_market_making_benchmark` in
[`project3/benchmark.hpp`](include/project3/benchmark.hpp) evaluates frozen
`TradingAgent` factories through the market-quality recorder. Each scenario
runs every policy on the same seeds, then computes paired P&L differences by
matching the recorded seed (comparison policy minus reference policy). It
reports per-scenario and pooled P&L, sample standard deviation, confidence
interval, Sharpe, drawdowns, loss VaR/expected shortfall, fill rate, spread
capture, adverse selection, inventory risk/concentration, turnover, and net
transaction costs. The confidence interval uses the Student-t distribution's
numerical quantile, computed from the regularized incomplete beta function;
it is not a bootstrap interval. These unadjusted intervals assume independent,
approximately normal episode-level P&L (or paired differences).
Empirical VaR/expected shortfall have coarse tails unless there are many
independent episodes.

The runner enforces at least two distinct scenarios, Hawkes configurations,
volatility-proxy labels and market-flow/size settings, liquidity labels and
book/depth settings, and seeds per scenario in each split. Seeds cannot overlap
between training, validation, and final test.
Every final-test scenario must be tagged unseen, include a parameter
perturbation, and have a market configuration not used in development. It
does not tune policies or send final-test results to a trainer; policy
factories must already be frozen before calling it. Callers remain responsible
for actually implementing each declared ablation—the type system cannot prove
that a factory disabled the named feature.

Run the end-to-end example with:

```sh
./build/project3_benchmark /tmp/project3-benchmark
```

The example trains its directional Hawkes forecaster only from training
configurations, then compares it with classical Avellaneda–Stoikov, a
symmetric static maker, and five separately toggled ablations. It writes:

- `<prefix>.manifest.json`: policy versions/settings, market and broker
  configuration, split/scenario labels, Hawkes parameters, and seeds.
- `<prefix>.runs.csv`: one row per policy/scenario/seed.
- `<prefix>.summaries.csv`: per-scenario and pooled statistics, including
  confidence bounds and tail-loss estimates.
- `<prefix>.paired.csv`: same-seed policy differences and their intervals.

The example's seed count is intended to make the workflow runnable, not to
establish a performance claim. Increase independent seeds and scenario
coverage for research conclusions; use validation results for model selection
and preserve final-test results for one-time reporting. A benchmark manifest
records policy configuration as caller-supplied text, so include all learned
weights, optimizer settings, training-data identifiers, and code revisions
needed to reproduce a policy.

The synthetic exchange has no direct volatility-regime parameter.
`volatility_regime` is a scenario label/proxy (implemented through event
rates, sizes, and book conditions), not realized diffusion volatility. Stress
regimes remain fixed from run start. The short example horizon also makes
annualized Sharpe especially noisy; longer episodes and a substantially larger
seed set are required before interpreting Sharpe or tail metrics.

## Engine profiling and performance

`project3::profile_project3_engine` (see
[`project3/engine_profile.hpp`](include/project3/engine_profile.hpp))
measures the full pipeline: reference Hawkes generation, serial and
parallel exchange episodes, batched Hawkes on CPU serial/parallel and the
optional CUDA backend, surrogate optimization throughput, peak RSS, and
CPU/GPU numerical-agreement errors. Run it with:

```sh
./build/project3_engine_profile benchmark-results/engine-profile.json
```

Performance infrastructure:

- `detail::parallel_for` — dynamic index-space map for independent
  episodes (benchmark, market-quality, profile).
- `detail::ChunkedPool` — persistent pool with contiguous static chunks
  for repeated small parallel sections (surrogate gradient evaluation);
  deterministic merge order, ~3x optimizer throughput on profiled batches.
- `simulate_batched_hawkes` — Ogata-thinned exponential Hawkes paths,
  one independent SplitMix64 stream per path, shared verbatim between the
  CPU and CUDA kernels.
- Measured-bottleneck optimizations: queue-tracker depth diffs use sorted
  vectors instead of per-event map copies; each strategy callback builds
  at most one `MarketView`.

Latest measured numbers and the optimization rationale are in
[`docs/benchmark-report.md`](docs/benchmark-report.md); CUDA was not
available on the profiling host, so GPU agreement remains unverified.

## Documentation

- [Architecture and data flow](docs/architecture.md)
- [Mathematical derivations](docs/math.md)
- [API reference](docs/api.md)
- [Benchmark report](docs/benchmark-report.md)
- [Numerical validation report](docs/numerical-validation.md)
- [Technical report (paper-style)](docs/paper.md)
- [Reproducing the experiments](docs/reproducing.md)
- Committed example artifacts: `benchmark-results/`

## Scope and limitations

The API is an event-driven agent interface over complete simulated runs; the
evaluator owns stepping, event tape, and exchange state. The Hawkes event tape
is pre-generated internally by the existing simulator, but only current
observations are exposed to agents. Existing synthetic mark generation and
seed semantics are retained. The demo and tests make no empirical calibration
or profitability claims.
