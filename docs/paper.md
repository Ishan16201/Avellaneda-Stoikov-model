# Project 3 — technical report

*Simulation- and code-level results only. Nothing in this report constitutes
evidence of real-world profitability.*

## 1. Summary

This work integrates a Hawkes-driven limit-order-book simulator with a
policy experimentation stack: agent-facing market views, queue estimation,
five policy families (Avellaneda–Stoikov, Hawkes-aware, reduced-state DP,
differentiable surrogate, score-function RL-style optimizer), a rigorous
train/validation/final-test benchmark with mandatory ablations, and a
performance toolchain (parallel Monte Carlo, persistent thread pool,
CPU/GPU Hawkes backends, engine profiler).

## 2. Theoretical results

- The Avellaneda–Stoikov implementation reproduces the classical
  first-order approximation; in the risk-neutral limit the reduced-state
  DP approaches the same `delta = 1/k` offset, cross-validating the
  expected-revenue Hamiltonian (see `docs/math.md` §1, §5).
- The DP's monotone CFL condition gives a verifiable stability bound for
  the discrete event/action solver.
- Score-function and forward-mode-AD gradients are unbiased/consistent
  estimators *of their respective surrogate objectives*; neither is an
  exact gradient of the discrete exchange.

## 3. Simulation results

- Facade replay matches the reference adapter event-for-event
  (integration tests).
- On the committed example benchmark (8 policies × 24 cells × 4 seeds,
  synthetic 3 s episodes), final-test mean P&L ranged from 1.9 to 15.9
  currency units with wide t-intervals; no policy separation is
  statistically established at this seed count (see
  `docs/benchmark-report.md`).
- Ablations behaved as a sanity check should: disabling Hawkes information
  changed nothing measurable in this run, flagging that the directional
  model's apparent benefit did not survive paired evaluation — the
  framework's intended outcome.

## 4. Empirical / performance results

Measured on Apple Silicon, 10 threads, Release build:

- Exchange episodes: ~210–270 paths/s serial, ~610–670 paths/s parallel
  (~2.3–2.6×); ~310–540k exchange events/s.
- Batched Hawkes: ~9.2M events/s serial, ~58M events/s parallel (~6.5×).
- Surrogate optimizer: ~1,160 → ~3,700 iterations/s with the persistent
  pool (~3.2×); serial remains default for bit-exactness.
- Peak RSS ~9–10 MB for the profiled workload.
- CUDA backend: implemented, CPU-validated structurally, **not
  numerically validated on this host** (no CUDA device).

## 5. Optimizations (measured bottlenecks only)

Sampling (`sample` on macOS) identified, in order: `QueueTracker` map
copies, repeated `MarketView` construction/vector copies, allocation
pressure, then the `exp`-bound Hawkes kernel. Applied: sorted-vector depth
merge, single-view-per-callback caching, fused kernel decay/intensity
pass, deterministic chunked parallel surrogate evaluation via
`ChunkedPool`. Exchange-episode micro throughput improved ~1.7×;
surrogate optimization ~3.2×.

## 6. Limitations and future work

- The exchange is a synthetic event simulator, not a calibrated market;
  marks, sizes, and Hawkes parameters are configuration inputs. No
  real-data calibration or live-trading claims are made.
- "Volatility regime" is a proxy through event rates/sizes/book depth;
  there is no explicit diffusion-volatility or jump-diffusion control,
  and stress regimes are fixed from t=0.
- Queue positions are estimates over aggregate depth with configured
  cancellation-allocation assumptions; latency is modeled as fixed
  outbound/inbound delays, not a stochastic feed.
- The Hawkes-aware markout adjustment is a heuristic, not a structural
  Hawkes-HJB solution; the softmax direction model can be miscalibrated
  under distribution shift.
- The reduced-state DP uses a one-dimensional activity approximation;
  it is not a multivariate Hawkes control solution.
- The surrogate optimizes a smooth fractional-fill objective; its
  gradients do not transfer exactly to the discrete exchange.
- CUDA path is unverified without CUDA hardware; CPU remains reference.
- Example seed counts are demonstration-scale; research conclusions need
  substantially more independent seeds and scenarios.
- Future directions: mid-run regime schedules, calibrated jump-diffusion
  marks, structural Hawkes-HJB approximations, GPU exchange batching,
  persistent-allocator arenas for episode parallelism, and larger
  out-of-sample scenario suites.

## 7. Distinguishing evidence types

| Claim type | Where |
|---|---|
| Theoretical | AS approximation, DP CFL bound, estimator definitions — `docs/math.md` |
| Simulation | P&L/fill/risk metrics, ablation outcomes — `benchmark-results/`, `docs/benchmark-report.md` |
| Empirical (performance) | Throughput, speedups, memory — `benchmark-results/engine-profile.json`, `docs/benchmark-report.md` |
| Unverified | CUDA agreement, profitability — explicitly flagged above |
