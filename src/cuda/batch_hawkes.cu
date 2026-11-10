#include "project3/batch_hawkes_kernel.hpp"

#include <cuda_runtime.h>

namespace {

__global__ void hawkes_kernel(
    project3::detail::PackedHawkesParameters parameters,
    double horizon,
    unsigned long long seed,
    unsigned long long path_count,
    project3::detail::PackedHawkesPathResult* out) {
  const unsigned long long path =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (path >= path_count) return;
  out[path] = project3::detail::simulate_one_hawkes_path(
      parameters, horizon, seed, path);
}

}  // namespace

extern "C" bool project3_cuda_batched_hawkes(
    const project3::detail::PackedHawkesParameters& parameters,
    double horizon,
    unsigned long long seed,
    std::size_t path_count,
    project3::detail::PackedHawkesPathResult* host_out) {
  if (path_count == 0 || host_out == nullptr) return false;
  project3::detail::PackedHawkesPathResult* device_out = nullptr;
  const auto bytes =
      path_count * sizeof(project3::detail::PackedHawkesPathResult);
  if (cudaMalloc(&device_out, bytes) != cudaSuccess) return false;
  constexpr int threads = 64;
  const int blocks = static_cast<int>(
      (path_count + static_cast<std::size_t>(threads) - 1) /
      static_cast<std::size_t>(threads));
  hawkes_kernel<<<blocks, threads>>>(parameters, horizon, seed,
                                     static_cast<unsigned long long>(path_count),
                                     device_out);
  const auto launch = cudaDeviceSynchronize();
  const auto copy = cudaMemcpy(host_out, device_out, bytes,
                               cudaMemcpyDeviceToHost);
  cudaFree(device_out);
  return launch == cudaSuccess && copy == cudaSuccess;
}
