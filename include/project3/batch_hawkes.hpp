#pragma once

#include "lob/stochastic/hawkes_parameters.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace project3 {

inline constexpr std::size_t kHawkesDimension = 6;
inline constexpr std::size_t kHawkesStateSize =
    kHawkesDimension * kHawkesDimension;

enum class HawkesBackend {
  CpuSerial,
  CpuParallel,
  Cuda,
};

struct BatchedHawkesConfig final {
  lob::stochastic::HawkesParameters parameters;
  double horizon_seconds{2.0};
  std::uint64_t seed{1};
  std::size_t path_count{64};
  std::size_t worker_count{0};
};

struct BatchedHawkesPathSummary final {
  std::size_t event_count{};
  double last_event_time_seconds{};
  std::array<double, kHawkesDimension> terminal_intensities{};
};

struct BatchedHawkesResult final {
  HawkesBackend backend{HawkesBackend::CpuSerial};
  std::string backend_name;
  std::size_t path_count{};
  std::size_t total_events{};
  double wall_seconds{};
  std::vector<BatchedHawkesPathSummary> paths;
};

[[nodiscard]] const char* to_string(HawkesBackend backend) noexcept;

/// Independent Ogata-thinned exponential Hawkes paths with counter-based RNG.
/// This is a Monte Carlo generator for profiling and GPU comparison; the
/// exchange still uses `lob::stochastic::ExponentialHawkesProcess`.
[[nodiscard]] BatchedHawkesResult simulate_batched_hawkes(
    const BatchedHawkesConfig& config,
    HawkesBackend backend = HawkesBackend::CpuSerial);

[[nodiscard]] bool cuda_batched_hawkes_available() noexcept;

/// Maximum absolute intensity difference across matched paths.
[[nodiscard]] double batched_hawkes_max_intensity_error(
    const BatchedHawkesResult& left,
    const BatchedHawkesResult& right);

}  // namespace project3
