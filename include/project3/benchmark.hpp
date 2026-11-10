#pragma once

#include "project3/market_quality.hpp"

#include <functional>
#include <string>
#include <vector>

namespace project3 {

enum class BenchmarkSplit {
  Training,
  Validation,
  FinalTest,
};

enum class AblationKind {
  WithoutHawkesInformation,
  WithoutQueueInformation,
  WithoutInventoryPenalties,
  WithoutRegimeInformation,
  WithoutDynamicQuoteAdjustment,
};

struct BenchmarkScenario final {
  std::string label;
  /// Descriptive labels; volatility is a realized-flow proxy in this simulator.
  std::string volatility_regime;
  std::string liquidity_regime;
  lob::stochastic::SyntheticMarketConfig market;
  std::vector<std::uint64_t> seeds;
  bool parameter_perturbation{};
  bool unseen_configuration{};
};

struct BenchmarkPolicy final {
  std::string name;
  std::string model_version;
  /// Human-readable serialized policy settings, recorded in the manifest.
  std::string configuration;
  std::function<std::unique_ptr<TradingAgent>()> make_agent;
};

struct BenchmarkAblation final {
  AblationKind kind{AblationKind::WithoutHawkesInformation};
  std::string baseline_policy_name;
  BenchmarkPolicy ablated_policy;
};

struct BenchmarkExperimentConfig final {
  std::string experiment_id;
  std::vector<BenchmarkPolicy> policies;
  std::vector<BenchmarkAblation> ablations;
  std::vector<BenchmarkScenario> training_scenarios;
  std::vector<BenchmarkScenario> validation_scenarios;
  std::vector<BenchmarkScenario> final_test_scenarios;
  MarketQualityConfig market_quality;
  double confidence_level{0.95};
  double primary_adverse_selection_horizon_seconds{0.5};
  /// Zero selects hardware concurrency for independent episode evaluation.
  std::size_t worker_count{0};
  /// Recorded in the manifest; defaults to the compiled git revision.
  std::string git_revision;
  /// Writes manifest, per-run, aggregate-summary, and paired-comparison CSVs.
  std::string output_prefix;
};

struct BenchmarkMetricSummary final {
  BenchmarkSplit split{BenchmarkSplit::Training};
  std::string scenario_label{"ALL"};
  bool parameter_perturbation{};
  bool unseen_configuration{};
  std::string policy_name;
  std::string model_version;
  std::size_t sample_count{};
  double mean_pnl_currency{};
  double pnl_standard_deviation_currency{};
  double pnl_confidence_interval_low_currency{};
  double pnl_confidence_interval_high_currency{};
  double mean_sharpe_ratio{};
  double mean_maximum_drawdown_currency{};
  double worst_maximum_drawdown_currency{};
  double value_at_risk_loss_currency{};
  double expected_shortfall_loss_currency{};
  double fill_rate{};
  double spread_capture_currency{};
  double adverse_selection_ticks{};
  double inventory_risk_lot_squared_seconds{};
  double inventory_concentration_ratio{};
  double turnover_currency{};
  double transaction_costs_currency{};
};

struct PairedBenchmarkComparison final {
  BenchmarkSplit split{BenchmarkSplit::Training};
  std::string scenario_label;
  bool parameter_perturbation{};
  bool unseen_configuration{};
  std::string reference_policy;
  std::string comparison_policy;
  std::size_t paired_samples{};
  double mean_pnl_difference_currency{};
  double pnl_difference_standard_deviation_currency{};
  double difference_confidence_interval_low_currency{};
  double difference_confidence_interval_high_currency{};
};

struct BenchmarkExperimentResult final {
  std::string experiment_id;
  std::vector<BenchmarkMetricSummary> summaries;
  std::vector<PairedBenchmarkComparison> paired_comparisons;
  std::string manifest_path;
  std::string runs_csv_path;
  std::string paired_csv_path;
  std::string summaries_csv_path;
};

/// Evaluation-only runner: policy factories must be frozen before invocation.
/// Final-test outputs are reported once and are never fed back into a trainer.
/// Seeds must be disjoint across splits. Ablation factories are caller-owned
/// and must actually disable the named feature.
[[nodiscard]] BenchmarkExperimentResult run_market_making_benchmark(
    const BenchmarkExperimentConfig& config);

[[nodiscard]] const char* to_string(BenchmarkSplit split) noexcept;
[[nodiscard]] const char* to_string(AblationKind kind) noexcept;

}  // namespace project3
