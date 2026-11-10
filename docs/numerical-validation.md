# Numerical validation report

Scope: correctness and reproducibility evidence for the components changed
or added in this work, plus the standing validation already enforced by the
suite. All commands assume the build in `build/` (see
`docs/reproducing.md`).

## Test suite

`ctest --test-dir build --output-on-failure` runs
`project3_integration_tests`, which covers:

- Event-tape replay equivalence: Project 3 facade vs the simulator's
  original Hawkes adapter — identical book snapshots, event-time
  intensities, timestamps, and public trades.
- Seeded reproducibility of full runs.
- Partial fills, cancellation, queue depletion, replacement priority,
  cash/inventory conservation, fee/rebate separation, mark-to-market,
  execution de-duplication.
- Avellaneda–Stoikov quote limits and baseline evaluation.
- Hawkes state exposure and held-out policy comparison.
- Reduced-state DP: analytical risk-neutral offset, terminal condition,
  inventory skew/limits, interpolation, CFL rejection, grid convergence,
  resource diagnostics.
- Surrogate: forward-mode gradients vs central finite differences
  (`check_surrogate_gradient`), all feature/parameter paths, multiple
  scales.
- Policy optimization study: train/validation/OOS partition checks.
- Market-quality analysis: spread decomposition identity
  `effective = realized + 2·impact`, inventory-limit breach detection,
  stress-scenario stability.
- Benchmark runner: exact summary/paired counts, seed-overlap rejection,
  duplicate-final-configuration rejection, manifest/CSV contents, exact
  Student-t critical value for n=4 (3.182446305284263), paired identical-
  seed zero differences.

Result at time of writing: **all tests pass**.

## Batched Hawkes backend agreement

`simulate_batched_hawkes` implements the same `simulate_one_hawkes_path`
kernel for `CpuSerial`, `CpuParallel`, and `Cuda`. Validation:

- CPU serial vs CPU parallel: identical per-path event counts and terminal
  intensities (`batched_hawkes_max_intensity_error = 0`) because each path
  is an independent SplitMix64 stream; scheduling does not affect results.
- CPU vs CUDA (when compiled with `PROJECT3_ENABLE_CUDA=ON`): same kernel
  source and RNG; `exp` may differ in the last bit between host and device
  libm, so agreement is reported as a maximum absolute intensity error,
  not a bitwise claim. **On the development host CUDA was unavailable
  (Apple Silicon, no nvcc); the CUDA path is compiled optionally and was
  not numerically validated on this machine.** Treat GPU numbers as
  unverified until run on CUDA hardware.
- Kernel refactor (fused decay+intensity, carried bounds) preserves the
  RNG consumption order: identical seeds produce identical event counts
  (verified: 11,940,504 total events before/after refactor on a fixed
  3000-path workload).

## Surrogate parallel evaluation

- `evaluation_worker_count=1` (default) is the serial reference and is
  bit-identical to previous releases.
- Parallel mode splits paths into contiguous chunks merged in fixed
  order; results are deterministic for a fixed worker count and differ
  from serial only in last-bit floating-point summation order.
- Verified: `differentiable_surrogate_gradient` at workers=1 vs workers=10
  on a 4096-path batch returned identical `gradient[0]` (-0.161206); the
  optimizer produced identical final reward (3.39042) with
  `worker_count=0` vs `10`.

## Queue-tracker and view-caching changes

- `observe_book` merge produces the same per-level deltas as the previous
  map-union implementation (union of last snapshot and observed levels in
  sorted order; duplicate prices keep last-write-wins semantics).
- `AgentGateway` view caching returns the same `MarketView` content the
  previous rebuild produced within a single strategy callback; the queue
  tracker is still refreshed after book/trade observations and execution
  reports before the agent sees the view.
- Covered by the replay-equivalence, queue-depletion/replacement, and
  market-quality tests above.

## Benchmark-runner invariants (tested)

- ≥2 scenarios, Hawkes configs, volatility/liquidity labels, seeds per
  scenario per split; final-test scenarios unseen + perturbed.
- Seed sets disjoint across training/validation/final test (overlap
  rejected with `std::invalid_argument`).
- Final-test market configuration must not equal any development
  configuration (rejected).
- All five ablation kinds required; policies frozen before invocation.
- Paired comparisons only within identical scenario+seed cells.

## Known validation gaps

- CUDA backend unverified on this host (no NVIDIA GPU/nvcc).
- Example seed counts (4–8 per cell) are too small for calibrated tail
  risk; treat VaR/ES/Sharpe as workflow demonstrations.
- "Volatility regime" is a flow/book proxy, not a calibrated diffusion
  parameter; stress regimes are fixed from t=0, not mid-run shocks.
