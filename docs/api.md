# API documentation

All public types live in `include/project3/` under namespace `project3`.
Unless noted, configurations are plain structs with documented defaults;
all monetary values are in the simulator's configured currency units and
prices in ticks.

## Agent interface (`market_agent.hpp`)

- `TradingAgent` — callbacks: `on_start`, `on_market_event`, `on_timer`,
  `on_trade`, `on_execution`, `on_stop`. `name()` identifies the policy.
- `MarketView` — immutable snapshot: timestamp, `bids`/`asks` price levels,
  best bid/ask, midprice, spread, inventory, cash, `hawkes_intensities`,
  `hawkes_flow` (six named channels), `open_orders`, `queue_positions`, and
  `market_maker` (`MarketMakerState`: account, inventory exposure, execution
  history). No future events or RNG access.
- `OrderGateway` — `submit_limit`, `submit_market`, `cancel`. Returned IDs
  come from a descending range disjoint from tape IDs.
- `ExecutionNotification`, `QueuePosition`, `PriceLevel`,
  `HawkesFlowState` — value types described in the header.

## Exchange (`exchange_adapter.hpp`, `execution_layer.hpp`)

- `ExchangeAdapter::run(agent, market, evaluation)` →
  `StrategyEvaluationReport` (broker report plus execution history).
- `QueueTracker` — queue estimator. `observe_book`, `synchronize_book`,
  `add_order`, `apply_fill`, `cancel_order`, `reject_order`,
  `observe_trade`, `replace_order`, `position(s)`.
  `Config::cancellation_allocation` selects `BehindFirst` (default),
  `AheadFirst`, or `ProRata`.

## Policies

- `avellaneda_stoikov.hpp` — `AvellanedaStoikovConfig`,
  `AvellanedaStoikovAgent`, `evaluate_avellaneda_stoikov` (P&L attribution,
  fill rates, Sharpe, drawdown).
- `hawkes_policy.hpp` — `HawkesFlowState`, `HawkesDirectionModel::train`,
  `predict`, `compare_market_making_policies`.
- `reduced_state_solver.hpp` — `ReducedStateConfig`,
  `solve_reduced_state`, `run_grid_convergence_study`.
- `differentiable_surrogate.hpp` — see `docs/math.md` §6. Key functions:
  `evaluate_differentiable_surrogate`, `smooth_surrogate_decision`,
  `differentiable_surrogate_gradient`, `finite_difference_surrogate_gradient`,
  `check_surrogate_gradient`, `make_surrogate_training_batch`,
  `optimize_differentiable_surrogate`. `DifferentiableSurrogateConfig::
  evaluation_worker_count` (default 1 = serial reference) and
  `SurrogateOptimizerConfig::worker_count` (0 = serial) control parallel
  evaluation; parallel results are deterministic for a fixed worker count.
- `policy_optimization.hpp` — categorical actor, score-function optimizer,
  train/validation/OOS evaluation with seed-partition enforcement.

## Analysis and benchmarking

- `market_quality.hpp` — `run_market_quality_study`,
  `MarketStressScenario`, `make_market_stress_scenarios`. Records causal
  execution analytics: markouts, effective/realized spread, implementation
  shortfall, price impact, queue waits, inventory exposure, broker P&L.
- `benchmark.hpp` — `run_market_making_benchmark`,
  `BenchmarkExperimentConfig` (policies, ablations, train/validation/
  final-test scenarios, output prefix). Enforces: ≥2 scenarios per split,
  disjoint seeds across splits, unseen+perturbed final-test configurations
  not used in development, all five ablation kinds, primary markout horizon
  present in market-quality horizons. Writes `.manifest.json`, `.runs.csv`,
  `.summaries.csv`, `.paired.csv`.

## Performance (`batch_hawkes.hpp`, `engine_profile.hpp`, `detail/parallel.hpp`)

- `simulate_batched_hawkes(config, backend)` — `CpuSerial`, `CpuParallel`,
  `Cuda`. Returns per-path event counts, last event time, terminal
  intensities, wall time. `batched_hawkes_max_intensity_error` compares
  backends; `cuda_batched_hawkes_available` reports build support.
- `profile_project3_engine(config)` → `EngineProfileReport`: events/s,
  paths/s serial & parallel, CPU speedup, Hawkes events/s, optimization
  iterations/s, GPU speedup (0 when CUDA absent), per-stage timings, peak
  RSS, CPU-vs-CPU and CPU-vs-CUDA intensity errors, optional JSON output.
- `detail::parallel_for(count, workers, fn)` — dynamic index map.
- `detail::ChunkedPool::instance().run(count, workers, fn)` — persistent
  pool, contiguous static chunks, deterministic merge order; the caller is
  participant 0. Not reentrant.

## Versioning

`version.hpp` exposes `library_version()`; the manifest records
`PROJECT3_VERSION`, `PROJECT3_GIT_REVISION`, worker count, and all policy
`model_version`/`configuration` strings.
