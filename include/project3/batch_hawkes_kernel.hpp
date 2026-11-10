#pragma once

#include <cstdint>
#include <math.h>

#ifdef __CUDACC__
#define PROJECT3_HD __host__ __device__
#else
#define PROJECT3_HD
#endif

namespace project3::detail {

/// SplitMix64: independent counter streams for CPU/GPU agreement tests.
PROJECT3_HD inline std::uint64_t splitmix64(std::uint64_t& state) {
  std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31U);
}

PROJECT3_HD inline double uniform_open(std::uint64_t& state) {
  const auto bits = splitmix64(state) >> 11U;
  return (static_cast<double>(bits) + 0.5) * 0x1.0p-53;
}

PROJECT3_HD inline std::uint64_t path_rng_state(const std::uint64_t seed,
                                                const std::uint64_t path) {
  std::uint64_t state = seed ^ 0xD1B54A32D192ED03ULL;
  state += path * 0x9e3779b97f4a7c15ULL;
  return splitmix64(state);
}

struct PackedHawkesParameters final {
  double mu[6]{};
  double alpha[36]{};
  double beta[36]{};
};

PROJECT3_HD inline void decay_state(double* state,
                                    const PackedHawkesParameters& p,
                                    const double dt) {
  for (int child = 0; child < 6; ++child) {
    for (int parent = 0; parent < 6; ++parent) {
      const int index = child * 6 + parent;
      state[index] *= exp(-p.beta[index] * dt);
    }
  }
}

PROJECT3_HD inline void intensities_from_state(
    const double* state,
    const PackedHawkesParameters& p,
    double* out) {
  for (int child = 0; child < 6; ++child) {
    double value = p.mu[child];
    for (int parent = 0; parent < 6; ++parent) {
      value += state[child * 6 + parent];
    }
    out[child] = value;
  }
}

/// Decays every excitation state in place and writes the post-decay
/// intensities in a single pass, replacing separate decay + intensity sweeps.
PROJECT3_HD inline void decay_state_and_intensities(
    double* state,
    const PackedHawkesParameters& p,
    const double dt,
    double* out) {
  for (int child = 0; child < 6; ++child) {
    double value = p.mu[child];
    for (int parent = 0; parent < 6; ++parent) {
      const int index = child * 6 + parent;
      const double decayed = state[index] * exp(-p.beta[index] * dt);
      state[index] = decayed;
      value += decayed;
    }
    out[child] = value;
  }
}

PROJECT3_HD inline double sum6(const double* values) {
  return values[0] + values[1] + values[2] + values[3] + values[4] + values[5];
}

struct PackedHawkesPathResult final {
  unsigned long long event_count{};
  double last_event_time{};
  double terminal_intensities[6]{};
};

PROJECT3_HD inline PackedHawkesPathResult simulate_one_hawkes_path(
    const PackedHawkesParameters& parameters,
    const double horizon,
    const std::uint64_t seed,
    const std::uint64_t path_index) {
  PackedHawkesPathResult result{};
  double state[36]{};
  double time = 0.0;
  std::uint64_t rng = path_rng_state(seed, path_index);
  double lambda[6];
  intensities_from_state(state, parameters, lambda);
  // `bound` is the post-jump intensity at `time`, an upper bound on the
  // intensity until the next accepted event (Ogata thinning).
  double bound = sum6(lambda);
  while (time < horizon) {
    const double candidate = time - log(uniform_open(rng)) / bound;
    if (candidate > horizon) {
      decay_state(state, parameters, horizon - time);
      time = horizon;
      break;
    }
    decay_state_and_intensities(state, parameters, candidate - time, lambda);
    time = candidate;
    const double actual = sum6(lambda);
    if (uniform_open(rng) * bound > actual) {
      bound = actual;
      continue;
    }
    double draw = uniform_open(rng) * actual;
    int type = 0;
    for (; type < 5; ++type) {
      if (draw < lambda[type]) break;
      draw -= lambda[type];
    }
    for (int child = 0; child < 6; ++child) {
      const int index = child * 6 + type;
      const double jump = parameters.alpha[index] * parameters.beta[index];
      state[index] += jump;
      lambda[child] += jump;
    }
    bound = sum6(lambda);
    ++result.event_count;
    result.last_event_time = time;
  }
  intensities_from_state(state, parameters, result.terminal_intensities);
  return result;
}

}  // namespace project3::detail
