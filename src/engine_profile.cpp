#include "project3/engine_profile.hpp"

#include "project3/avellaneda_stoikov.hpp"
#include "project3/detail/parallel.hpp"
#include "project3/exchange_adapter.hpp"
#include "project3/version.hpp"

#include "lob/stochastic/hawkes_process.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <sys/resource.h>
#include <vector>

namespace project3 {
namespace {

class QuietMaker final : public TradingAgent {
 public:
  [[nodiscard]] std::string name() const override { return "profile-maker"; }

  void on_start(const MarketView& view, OrderGateway& gateway) override {
    quote(view, gateway);
  }

  void on_market_event(const MarketView& view, OrderGateway& gateway) override {
    quote(view, gateway);
  }

  void on_timer(const MarketView& view, OrderGateway& gateway) override {
    quote(view, gateway);
  }

 private:
  void quote(const MarketView& view, OrderGateway& gateway) {
    if (!view.best_bid || !view.best_ask) return;
    if (view.open_orders.empty()) {
      if (view.inventory_lots < 5) {
        static_cast<void>(
            gateway.submit_limit(lob::core::Side::Buy, *view.best_bid, 1));
      }
      if (view.inventory_lots > -5) {
        static_cast<void>(
            gateway.submit_limit(lob::core::Side::Sell, *view.best_ask, 1));
      }
    }
  }
};

struct CountedRun final {
  std::size_t events{};
  double seconds{};
};

double now_seconds() {
  using Clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

std::size_t peak_rss_bytes() {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#if defined(__APPLE__)
  return static_cast<std::size_t>(usage.ru_maxrss);
#else
  return static_cast<std::size_t>(usage.ru_maxrss) * 1024U;
#endif
}

CountedRun run_exchange(const lob::stochastic::SyntheticMarketConfig& market,
                        const std::uint64_t seed) {
  auto configured = market;
  configured.seed = seed;
  QuietMaker agent;
  lob::strategy::StrategyEvaluatorConfig evaluation;
  evaluation.timer_interval_ns = 250'000'000;
  const auto started = std::chrono::steady_clock::now();
  const auto report = ExchangeAdapter{}.run(agent, configured, evaluation);
  const auto elapsed = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - started)
                           .count();
  return {.events = report.execution_history.size(), .seconds = elapsed};
}

std::size_t hawkes_event_count(
    const lob::stochastic::SyntheticMarketConfig& market,
    const std::uint64_t seed) {
  lob::stochastic::ExponentialHawkesProcess process{market.hawkes, seed};
  return process.simulate_until(market.horizon_seconds).size();
}

void write_json(const std::string& path, const EngineProfileReport& report) {
  std::ofstream output{path};
  if (!output) throw std::runtime_error{"cannot write engine profile JSON"};
  output << std::setprecision(17) << "{\n  \"git_revision\": \""
         << report.git_revision << "\",\n  \"library_version\": \""
         << report.library_version << "\",\n  \"worker_count\": "
         << report.worker_count << ",\n  \"peak_resident_set_bytes\": "
         << report.peak_resident_set_bytes
         << ",\n  \"events_per_second\": " << report.events_per_second
         << ",\n  \"simulation_paths_per_second\": "
         << report.simulation_paths_per_second
         << ",\n  \"sequential_paths_per_second\": "
         << report.sequential_paths_per_second
         << ",\n  \"parallel_paths_per_second\": "
         << report.parallel_paths_per_second
         << ",\n  \"cpu_path_speedup\": " << report.cpu_path_speedup
         << ",\n  \"hawkes_events_per_second\": "
         << report.hawkes_events_per_second
         << ",\n  \"optimization_iterations_per_second\": "
         << report.optimization_iterations_per_second
         << ",\n  \"surrogate_iterations_per_second\": "
         << report.surrogate_iterations_per_second
         << ",\n  \"gpu_hawkes_speedup\": " << report.gpu_hawkes_speedup
         << ",\n  \"cuda_available\": "
         << (report.cuda_available ? "true" : "false")
         << ",\n  \"cuda_cpu_max_intensity_error\": "
         << report.cuda_cpu_max_intensity_error
         << ",\n  \"cpu_serial_parallel_max_intensity_error\": "
         << report.cpu_serial_parallel_max_intensity_error
         << ",\n  \"stages\": [";
  for (std::size_t i = 0; i < report.stages.size(); ++i) {
    if (i > 0) output << ',';
    output << "\n    {\"name\": \"" << report.stages[i].name
           << "\", \"wall_seconds\": " << report.stages[i].wall_seconds
           << ", \"items_per_second\": " << report.stages[i].items_per_second
           << ", \"item_count\": " << report.stages[i].item_count << '}';
  }
  output << "\n  ]\n}\n";
}

TimedStage stage(const std::string& name,
                 const double seconds,
                 const std::size_t items) {
  return {.name = name,
          .wall_seconds = seconds,
          .items_per_second = seconds > 0.0
                                  ? static_cast<double>(items) / seconds
                                  : 0.0,
          .item_count = items};
}

}  // namespace

EngineProfileReport profile_project3_engine(const EngineProfileConfig& config) {
  if (config.seeds.empty() || config.hawkes_paths == 0 ||
      config.surrogate_paths == 0 || config.surrogate_steps == 0 ||
      config.surrogate_iterations == 0) {
    throw std::invalid_argument{"engine profile configuration is incomplete"};
  }
  lob::stochastic::validate_stationary(config.market.hawkes);

  EngineProfileReport report;
  report.git_revision = git_revision();
  report.library_version = library_version();
  report.worker_count = detail::resolve_worker_count(config.worker_count);
  report.cuda_available = cuda_batched_hawkes_available();

  std::size_t hawkes_events = 0;
  const auto hawkes_started = now_seconds();
  for (const auto seed : config.seeds) {
    hawkes_events += hawkes_event_count(config.market, seed);
  }
  const auto hawkes_seconds = now_seconds() - hawkes_started;
  report.hawkes_events_per_second =
      hawkes_seconds > 0.0
          ? static_cast<double>(hawkes_events) / hawkes_seconds
          : 0.0;
  report.stages.push_back(
      stage("hawkes-generation-cpu-reference", hawkes_seconds, hawkes_events));

  double sequential_seconds = 0.0;
  std::size_t sequential_events = 0;
  const auto seq_started = now_seconds();
  for (const auto seed : config.seeds) {
    const auto run = run_exchange(config.market, seed);
    sequential_events += run.events;
    sequential_seconds += run.seconds;
  }
  const auto seq_wall = now_seconds() - seq_started;
  report.sequential_paths_per_second =
      seq_wall > 0.0 ? static_cast<double>(config.seeds.size()) / seq_wall
                     : 0.0;
  report.events_per_second =
      seq_wall > 0.0 ? static_cast<double>(hawkes_events) / seq_wall : 0.0;
  report.stages.push_back(
      stage("exchange-episodes-serial", seq_wall, config.seeds.size()));

  std::vector<CountedRun> parallel_runs(config.seeds.size());
  const auto par_started = now_seconds();
  detail::parallel_for(
      config.seeds.size(), config.worker_count, [&](const std::size_t i) {
        parallel_runs[i] = run_exchange(config.market, config.seeds[i]);
      });
  const auto par_wall = now_seconds() - par_started;
  report.parallel_paths_per_second =
      par_wall > 0.0 ? static_cast<double>(config.seeds.size()) / par_wall
                     : 0.0;
  report.cpu_path_speedup =
      par_wall > 0.0 ? seq_wall / par_wall : 0.0;
  report.simulation_paths_per_second = report.parallel_paths_per_second;
  report.stages.push_back(
      stage("exchange-episodes-parallel", par_wall, config.seeds.size()));
  static_cast<void>(sequential_events);
  static_cast<void>(sequential_seconds);

  BatchedHawkesConfig hawkes_batch;
  hawkes_batch.parameters = config.market.hawkes;
  hawkes_batch.horizon_seconds = config.market.horizon_seconds;
  hawkes_batch.seed = config.seeds.front();
  hawkes_batch.path_count = config.hawkes_paths;
  hawkes_batch.worker_count = config.worker_count;
  const auto serial_batch =
      simulate_batched_hawkes(hawkes_batch, HawkesBackend::CpuSerial);
  const auto parallel_batch =
      simulate_batched_hawkes(hawkes_batch, HawkesBackend::CpuParallel);
  report.cpu_serial_parallel_max_intensity_error =
      batched_hawkes_max_intensity_error(serial_batch, parallel_batch);
  report.stages.push_back(stage("batched-hawkes-cpu-serial",
                                serial_batch.wall_seconds,
                                serial_batch.total_events));
  report.stages.push_back(stage("batched-hawkes-cpu-parallel",
                                parallel_batch.wall_seconds,
                                parallel_batch.total_events));

  if (report.cuda_available) {
    const auto cuda_batch =
        simulate_batched_hawkes(hawkes_batch, HawkesBackend::Cuda);
    report.cuda_cpu_max_intensity_error =
        batched_hawkes_max_intensity_error(serial_batch, cuda_batch);
    report.gpu_hawkes_speedup =
        cuda_batch.wall_seconds > 0.0
            ? serial_batch.wall_seconds / cuda_batch.wall_seconds
            : 0.0;
    report.stages.push_back(stage("batched-hawkes-cuda", cuda_batch.wall_seconds,
                                  cuda_batch.total_events));
  } else {
    report.gpu_hawkes_speedup = 0.0;
    report.cuda_cpu_max_intensity_error = 0.0;
  }

  const auto surrogate_batch = make_surrogate_training_batch(
      config.seeds.front(), config.surrogate_paths, config.surrogate_steps);
  SurrogateParameters parameters{};
  SurrogateOptimizerConfig optimizer;
  optimizer.iterations = config.surrogate_iterations;
  optimizer.learning_rate = 0.02;
  optimizer.worker_count = report.worker_count;
  const auto opt_started = now_seconds();
  static_cast<void>(optimize_differentiable_surrogate(
      parameters, surrogate_batch, {}, optimizer));
  const auto opt_seconds = now_seconds() - opt_started;
  report.surrogate_iterations_per_second =
      opt_seconds > 0.0
          ? static_cast<double>(config.surrogate_iterations) / opt_seconds
          : 0.0;
  report.optimization_iterations_per_second =
      report.surrogate_iterations_per_second;
  report.stages.push_back(stage("surrogate-optimization", opt_seconds,
                                config.surrogate_iterations));

  AvellanedaStoikovConfig as_config;
  as_config.horizon_seconds = config.market.horizon_seconds;
  as_config.inventory_limit_lots = 5;
  as_config.max_quote_size_lots = 1;
  auto as_market = config.market;
  as_market.seed = config.seeds.front();
  const auto as_started = now_seconds();
  static_cast<void>(evaluate_avellaneda_stoikov(as_config, as_market));
  report.stages.push_back(stage("avellaneda-stoikov-episode",
                                now_seconds() - as_started, 1));

  report.peak_resident_set_bytes = peak_rss_bytes();
  if (!config.output_json_path.empty()) {
    const std::filesystem::path prefix{config.output_json_path};
    if (!prefix.parent_path().empty()) {
      std::filesystem::create_directories(prefix.parent_path());
    }
    write_json(config.output_json_path, report);
    report.json_path = config.output_json_path;
  }
  return report;
}

}  // namespace project3
