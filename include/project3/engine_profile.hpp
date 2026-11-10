#pragma once

#include "project3/batch_hawkes.hpp"
#include "project3/differentiable_surrogate.hpp"

#include "lob/stochastic/hawkes_market_adapter.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace project3 {

struct EngineProfileConfig final {
  lob::stochastic::SyntheticMarketConfig market;
  std::vector<std::uint64_t> seeds{1, 2, 3, 4};
  std::size_t worker_count{0};
  std::size_t surrogate_paths{32};
  std::size_t surrogate_steps{8};
  std::size_t surrogate_iterations{25};
  std::size_t hawkes_paths{64};
  std::string output_json_path;
};

struct TimedStage final {
  std::string name;
  double wall_seconds{};
  double items_per_second{};
  std::size_t item_count{};
};

struct EngineProfileReport final {
  std::string git_revision;
  std::string library_version;
  std::size_t worker_count{};
  std::size_t peak_resident_set_bytes{};
  double events_per_second{};
  double simulation_paths_per_second{};
  double sequential_paths_per_second{};
  double parallel_paths_per_second{};
  double cpu_path_speedup{};
  double hawkes_events_per_second{};
  double optimization_iterations_per_second{};
  double surrogate_iterations_per_second{};
  double gpu_hawkes_speedup{};
  bool cuda_available{};
  double cuda_cpu_max_intensity_error{};
  double cpu_serial_parallel_max_intensity_error{};
  std::vector<TimedStage> stages;
  std::string json_path;
};

/// Profiles Hawkes generation, exchange episodes, parallel Monte Carlo,
/// surrogate optimization, and the optional CUDA Hawkes backend.
/// The discrete matching engine remains the CPU reference.
[[nodiscard]] EngineProfileReport profile_project3_engine(
    const EngineProfileConfig& config);

}  // namespace project3
