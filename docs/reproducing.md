# Reproducing the experiments

## Prerequisites

- CMake ≥ 3.25, a C++20 compiler, POSIX threads.
- The Hawkes LOB simulator source tree as a sibling directory named
  `Hawkes-driven-limit-order-book-simulator.` (or pass
  `-DPROJECT3_LOB_SOURCE_DIR=/path/to/simulator`).
- Optional: CUDA toolkit + `nvcc` for the GPU Hawkes backend
  (`-DPROJECT3_ENABLE_CUDA=ON`).

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

A fresh clone reproduces the library revision via `PROJECT3_GIT_REVISION`
(recorded in benchmark manifests and profile JSON).

## Main experiments

Policy benchmark (writes manifest, per-run, summary, and paired CSVs):

```sh
./build/project3_benchmark benchmark-results/project3
```

The example trains its directional forecaster only on training
configurations, freezes all policies, then evaluates train/validation/
final-test splits. Final-test results are written once and never fed back
into training. To change seeds/scenarios, edit
`examples/benchmark_experiment.cpp`; increase `seeds` per scenario and add
scenarios for research-grade statistics.

Engine profile:

```sh
./build/project3_engine_profile benchmark-results/engine-profile.json
```

Records events/s, paths/s serial+parallel, CPU speedup, Hawkes
events/s, optimization iterations/s, GPU speedup (0 without CUDA), peak
RSS, backend agreement errors, and per-stage timings.

Other examples (each self-contained and seeded):

```sh
./build/project3_demo
./build/avellaneda_stoikov_demo
./build/hawkes_policy_comparison
./build/reduced_state_convergence
./build/differentiable_surrogate_demo   # writes surrogate-policy-*.csv/txt
./build/policy_optimization_demo        # writes policy-gradient-*.csv/txt
./build/market_quality_stress
```

## Reproducibility notes

- All randomness is seed-driven; identical seeds + configuration produce
  identical episodes (verified by integration tests).
- Parallel reductions (benchmark episodes, surrogate evaluation, batched
  Hawkes) are deterministic for a fixed worker count; surrogate parallel
  mode may differ from serial in the last floating-point bit only.
- `worker_count=0` means hardware concurrency; set it explicitly for
  cross-machine bit-comparison.
- Benchmark manifests record policy `model_version`/`configuration`
  strings, market and broker configuration, split labels, Hawkes
  parameters, and seeds. Learned weights must be serialized into
  `configuration` by the caller for full reproducibility.
- Committed artifacts under `benchmark-results/` are examples generated on
  one machine; regenerate them rather than treating the numbers as
ground truth.
