#include "project3/batch_hawkes.hpp"
#include "project3/batch_hawkes_kernel.hpp"
#include "project3/detail/parallel.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

#if defined(PROJECT3_HAS_CUDA)
extern "C" bool project3_cuda_batched_hawkes(
    const project3::detail::PackedHawkesParameters& parameters,
    double horizon,
    std::uint64_t seed,
    std::size_t path_count,
    project3::detail::PackedHawkesPathResult* out);
#endif

namespace project3 {
namespace {

detail::PackedHawkesParameters pack_parameters(
    const lob::stochastic::HawkesParameters& parameters) {
  if (parameters.baseline_events_per_second.size() != kHawkesDimension ||
      parameters.alpha_branching.size() != kHawkesDimension ||
      parameters.beta_per_second.size() != kHawkesDimension) {
    throw std::invalid_argument{
        "batched Hawkes generator requires the six-channel market process"};
  }
  detail::PackedHawkesParameters packed;
  for (std::size_t i = 0; i < kHawkesDimension; ++i) {
    packed.mu[i] = parameters.baseline_events_per_second[i];
    if (parameters.alpha_branching[i].size() != kHawkesDimension ||
        parameters.beta_per_second[i].size() != kHawkesDimension) {
      throw std::invalid_argument{"Hawkes matrices must be 6x6"};
    }
    for (std::size_t j = 0; j < kHawkesDimension; ++j) {
      packed.alpha[i * kHawkesDimension + j] = parameters.alpha_branching[i][j];
      packed.beta[i * kHawkesDimension + j] = parameters.beta_per_second[i][j];
    }
  }
  return packed;
}

BatchedHawkesPathSummary from_packed(
    const detail::PackedHawkesPathResult& packed) {
  BatchedHawkesPathSummary summary;
  summary.event_count = static_cast<std::size_t>(packed.event_count);
  summary.last_event_time_seconds = packed.last_event_time;
  for (std::size_t i = 0; i < kHawkesDimension; ++i) {
    summary.terminal_intensities[i] = packed.terminal_intensities[i];
  }
  return summary;
}

BatchedHawkesResult run_cpu(const BatchedHawkesConfig& config,
                            const HawkesBackend backend) {
  lob::stochastic::validate_stationary(config.parameters);
  if (!std::isfinite(config.horizon_seconds) || config.horizon_seconds <= 0.0 ||
      config.path_count == 0) {
    throw std::invalid_argument{"invalid batched Hawkes configuration"};
  }
  const auto packed = pack_parameters(config.parameters);
  const auto workers = backend == HawkesBackend::CpuSerial
                           ? std::size_t{1}
                           : config.worker_count;
  std::vector<detail::PackedHawkesPathResult> packed_paths(config.path_count);
  const auto started = std::chrono::steady_clock::now();
  detail::parallel_for(config.path_count, workers, [&](const std::size_t path) {
    packed_paths[path] = detail::simulate_one_hawkes_path(
        packed, config.horizon_seconds, config.seed,
        static_cast<std::uint64_t>(path));
  });
  const auto elapsed = std::chrono::steady_clock::now() - started;
  BatchedHawkesResult result;
  result.backend = backend;
  result.backend_name = to_string(backend);
  result.path_count = config.path_count;
  result.wall_seconds =
      std::chrono::duration<double>(elapsed).count();
  result.paths.reserve(config.path_count);
  for (const auto& path : packed_paths) {
    auto summary = from_packed(path);
    result.total_events += summary.event_count;
    result.paths.push_back(summary);
  }
  return result;
}

}  // namespace

const char* to_string(const HawkesBackend backend) noexcept {
  switch (backend) {
    case HawkesBackend::CpuSerial: return "cpu-serial";
    case HawkesBackend::CpuParallel: return "cpu-parallel";
    case HawkesBackend::Cuda: return "cuda";
  }
  return "unknown";
}

bool cuda_batched_hawkes_available() noexcept {
#if defined(PROJECT3_HAS_CUDA)
  return true;
#else
  return false;
#endif
}

BatchedHawkesResult simulate_batched_hawkes(const BatchedHawkesConfig& config,
                                            const HawkesBackend backend) {
  if (backend == HawkesBackend::Cuda) {
#if defined(PROJECT3_HAS_CUDA)
    lob::stochastic::validate_stationary(config.parameters);
    const auto packed = pack_parameters(config.parameters);
    std::vector<detail::PackedHawkesPathResult> packed_paths(config.path_count);
    const auto started = std::chrono::steady_clock::now();
    if (!project3_cuda_batched_hawkes(packed, config.horizon_seconds,
                                      config.seed, config.path_count,
                                      packed_paths.data())) {
      throw std::runtime_error{"CUDA Hawkes batch failed"};
    }
    const auto elapsed = std::chrono::steady_clock::now() - started;
    BatchedHawkesResult result;
    result.backend = HawkesBackend::Cuda;
    result.backend_name = to_string(HawkesBackend::Cuda);
    result.path_count = config.path_count;
    result.wall_seconds = std::chrono::duration<double>(elapsed).count();
    for (const auto& path : packed_paths) {
      auto summary = from_packed(path);
      result.total_events += summary.event_count;
      result.paths.push_back(summary);
    }
    return result;
#else
    throw std::runtime_error{
        "CUDA Hawkes backend was not compiled; CPU remains the reference"};
#endif
  }
  return run_cpu(config, backend);
}

double batched_hawkes_max_intensity_error(const BatchedHawkesResult& left,
                                          const BatchedHawkesResult& right) {
  if (left.paths.size() != right.paths.size()) {
    throw std::invalid_argument{"Hawkes comparison requires equal path counts"};
  }
  double error = 0.0;
  for (std::size_t p = 0; p < left.paths.size(); ++p) {
    if (left.paths[p].event_count != right.paths[p].event_count) {
      error = std::max(error, 1.0);
    }
    for (std::size_t i = 0; i < kHawkesDimension; ++i) {
      error = std::max(
          error, std::abs(left.paths[p].terminal_intensities[i] -
                          right.paths[p].terminal_intensities[i]));
    }
  }
  return error;
}

}  // namespace project3
