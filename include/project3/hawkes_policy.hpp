#pragma once

#include "project3/avellaneda_stoikov.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace project3 {

struct HawkesDirectionFitConfig final {
  double forecast_horizon_seconds{0.25};
  std::size_t training_epochs{500};
  double learning_rate{0.05};
  double l2_regularization{0.01};
};

struct HawkesDirectionPrediction final {
  double probability_up{};
  double probability_down{};
  double probability_unchanged{};
  double mean_up_move_ticks{};
  double mean_down_move_ticks{};
  double expected_up_move_ticks() const noexcept {
    return probability_up * mean_up_move_ticks;
  }
  double expected_down_move_ticks() const noexcept {
    return probability_down * mean_down_move_ticks;
  }
};

/// Empirical three-class softmax model trained on Hawkes synthetic runs.
/// Features are log intensities for all six event types plus normalized OFI.
class HawkesDirectionModel final {
 public:
  [[nodiscard]] static HawkesDirectionModel train(
      const std::vector<lob::stochastic::SyntheticMarketConfig>& training_runs,
      const HawkesDirectionFitConfig& fit_config = {});

  [[nodiscard]] HawkesDirectionPrediction predict(
      const HawkesFlowState& state) const;
  [[nodiscard]] double forecast_horizon_seconds() const noexcept {
    return forecast_horizon_seconds_;
  }
  [[nodiscard]] std::size_t training_observations() const noexcept {
    return training_observations_;
  }
  [[nodiscard]] double training_log_loss() const noexcept {
    return training_log_loss_;
  }
  /// In model class order: down, unchanged, up.
  [[nodiscard]] const std::array<double, 3>& training_class_probabilities() const noexcept {
    return training_class_probabilities_;
  }

 private:
  static constexpr std::size_t kFeatureCount = 7;
  static constexpr std::size_t kClassCount = 3;
  std::array<double, kFeatureCount> feature_means_{};
  std::array<double, kFeatureCount> feature_scales_{};
  std::array<std::array<double, kFeatureCount + 1>, kClassCount> weights_{};
  std::array<double, kClassCount> training_class_probabilities_{};
  double mean_up_move_ticks_{};
  double mean_down_move_ticks_{};
  double forecast_horizon_seconds_{};
  std::size_t training_observations_{};
  double training_log_loss_{};
};

struct HawkesAwareConfig final {
  double mean_market_order_size_lots{1.0};
};

struct SymmetricMarketMakerConfig final {
  lob::core::PriceTicks quote_offset_ticks{1};
  lob::core::QuantityLots quote_size_lots{1};
  lob::core::QuantityLots inventory_limit_lots{10};
};

struct PolicyRunSummary final {
  std::size_t runs{};
  double mean_total_pnl{};
  double std_total_pnl{};
  double mean_fill_rate{};
  double mean_max_drawdown{};
  double mean_turnover{};
  double mean_max_inventory{};
};

struct PolicyComparisonScenario final {
  std::string label;
  lob::stochastic::SyntheticMarketConfig market;
  std::vector<std::uint64_t> evaluation_seeds;
};

struct PolicyComparisonResult final {
  std::string scenario_label;
  std::vector<lob::strategy::StrategyEvaluationReport> classical_as;
  std::vector<lob::strategy::StrategyEvaluationReport> hawkes_aware;
  std::vector<lob::strategy::StrategyEvaluationReport> symmetric;
  PolicyRunSummary classical_summary;
  PolicyRunSummary hawkes_summary;
  PolicyRunSummary symmetric_summary;
  double direction_log_loss{};
  double direction_accuracy{};
  double direction_base_rate_log_loss{};
  double direction_base_rate_accuracy{};
  std::size_t direction_observations{};
};

/// Trains only on training_runs, then evaluates all three policies on paired
/// seeds and Hawkes parameter configurations not present in the training set.
[[nodiscard]] std::vector<PolicyComparisonResult> compare_market_making_policies(
    const std::vector<lob::stochastic::SyntheticMarketConfig>& training_runs,
    const std::vector<PolicyComparisonScenario>& evaluation_scenarios,
    const HawkesDirectionFitConfig& fit_config,
    const AvellanedaStoikovConfig& avellaneda_config,
    const HawkesAwareConfig& hawkes_config,
    const SymmetricMarketMakerConfig& symmetric_config,
    lob::strategy::StrategyEvaluatorConfig evaluation_config = {});

}  // namespace project3
