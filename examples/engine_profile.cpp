#include "project3/engine_profile.hpp"

#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  try {
    project3::EngineProfileConfig config;
    config.market.horizon_seconds = 30.0;
    config.market.hawkes.baseline_events_per_second = {8.0, 6.0, 2.8, 2.8, 4.8, 4.8};
    config.market.hawkes.alpha_branching.assign(6, std::vector<double>(6, 0.02));
    config.market.hawkes.beta_per_second.assign(6, std::vector<double>(6, 3.0));
    config.market.initial_book.best_bid_ticks = 1000;
    config.market.initial_book.best_ask_ticks = 1001;
    config.market.initial_book.levels_per_side = 5;
    config.market.initial_book.orders_per_level = 4;
    config.seeds = {11, 12, 13, 14, 15, 16, 17, 18};
    config.hawkes_paths = 512;
    config.surrogate_paths = 64;
    config.surrogate_steps = 12;
    config.surrogate_iterations = 100;
    config.output_json_path =
        argc > 1 ? argv[1] : "benchmark-results/engine-profile.json";
    const auto report = project3::profile_project3_engine(config);
    std::cout << "Project 3 engine profile\n"
              << "  git_revision=" << report.git_revision << '\n'
              << "  workers=" << report.worker_count << '\n'
              << "  events/s=" << report.events_per_second << '\n'
              << "  sequential paths/s=" << report.sequential_paths_per_second
              << '\n'
              << "  parallel paths/s=" << report.parallel_paths_per_second
              << '\n'
              << "  CPU path speedup=" << report.cpu_path_speedup << '\n'
              << "  Hawkes events/s=" << report.hawkes_events_per_second << '\n'
              << "  surrogate iterations/s="
              << report.surrogate_iterations_per_second << '\n'
              << "  peak RSS bytes=" << report.peak_resident_set_bytes << '\n'
              << "  CUDA available=" << (report.cuda_available ? "yes" : "no")
              << '\n'
              << "  CPU serial/parallel Hawkes intensity error="
              << report.cpu_serial_parallel_max_intensity_error << '\n';
    if (report.cuda_available) {
      std::cout << "  GPU Hawkes speedup=" << report.gpu_hawkes_speedup
                << "\n  CUDA vs CPU intensity error="
                << report.cuda_cpu_max_intensity_error << '\n';
    }
    std::cout << "  json=" << report.json_path << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Engine profile failed: " << error.what() << '\n';
    return 1;
  }
}
