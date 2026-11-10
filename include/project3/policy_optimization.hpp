#pragma once

#include "project3/avellaneda_stoikov.hpp"
#include "project3/differentiable_surrogate.hpp"
#include "project3/reduced_state_solver.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace project3 {

inline constexpr std::size_t kPolicyFeatureCount = 18;
inline constexpr std::size_t kPolicyActionCount = 8;
inline constexpr std::size_t kPolicyGradientParameterCount =
    2 * kPolicyActionCount * kPolicyFeatureCount;

struct PolicyGradientParameters final {
  std::array<double, kPolicyGradientParameterCount> values{};
};

struct PolicyOptimizationScenario final {
  std::string label;
  lob::stochastic::SyntheticMarketConfig market;
  std::vector<std::uint64_t> seeds;
};

struct PolicyGradientConfig final {
  std::size_t epochs{30};
  std::size_t episodes_per_epoch{8};
  double learning_rate{0.003};
  double minimum_learning_rate{0.0003};
  LearningRateSchedule learning_rate_schedule{
      LearningRateSchedule::LinearDecay};
  double gradient_clip_norm{5.0};
  double reward_normalization_scale_currency{100.0};
  double baseline_decay{0.9};
  double adam_beta1{0.9};
  double adam_beta2{0.999};
  double adam_epsilon{1.0e-8};
  double inventory_limit_lots{10.0};
  double running_inventory_penalty_currency_per_second_lot_squared{0.0};
  double terminal_inventory_penalty_currency_per_lot_squared{0.0};
  double volatility_feature_scale{1.0};
  double queue_feature_scale_lots{10.0};
  double depth_feature_scale_lots{10.0};
  double intensity_feature_scale{5.0};
  double decision_interval_seconds{0.25};
  std::array<double, 3> quote_offsets_ticks{1.0, 2.0, 3.0};
  std::array<lob::core::QuantityLots, 2> quote_sizes_lots{1, 2};
  std::size_t validation_patience{6};
  std::uint64_t seed{1};
  /// Zero selects hardware concurrency for independent training episodes.
  std::size_t worker_count{0};
  std::string checkpoint_path;
  std::string learning_curve_csv_path;
  lob::strategy::StrategyEvaluatorConfig evaluation;
};

struct PolicyLearningCurvePoint final {
  std::size_t epoch{};
  double training_reward_mean_currency{};
  double training_reward_std_currency{};
  double validation_reward_mean_currency{};
  double validation_reward_std_currency{};
  double normalized_advantage_mean{};
  double gradient_norm{};
  double learning_rate{};
};

struct PolicyGradientTrainingResult final {
  PolicyGradientParameters parameters;
  PolicyGradientConfig hyperparameters;
  std::vector<PolicyLearningCurvePoint> learning_curve;
  std::size_t best_epoch{};
  double best_validation_reward_currency{};
  double training_reward_currency{};
  std::size_t completed_epochs{};
  std::uint64_t seed{};
};

struct SurrogateTrainingConfig final {
  std::size_t epochs{100};
  double learning_rate{0.02};
  double minimum_learning_rate{0.002};
  LearningRateSchedule learning_rate_schedule{LearningRateSchedule::CosineDecay};
  double gradient_clip_norm{5.0};
  double reward_normalization_scale_currency{10.0};
  double adam_beta1{0.9};
  double adam_beta2{0.999};
  double adam_epsilon{1.0e-8};
  std::size_t validation_patience{15};
  std::uint64_t seed{2};
  std::string checkpoint_path;
  std::string learning_curve_csv_path;
  DifferentiableSurrogateConfig surrogate;
};

struct SurrogateLearningCurvePoint final {
  std::size_t epoch{};
  double training_reward_mean_currency{};
  double validation_reward_mean_currency{};
  double gradient_norm{};
  double learning_rate{};
};

struct SurrogateTrainingResult final {
  SurrogateParameters parameters;
  SurrogateTrainingConfig hyperparameters;
  std::vector<SurrogateLearningCurvePoint> learning_curve;
  std::size_t best_epoch{};
  double best_validation_reward_currency{};
  double training_reward_currency{};
  double training_exact_market_reward_currency{};
  double validation_exact_market_reward_currency{};
  std::uint64_t seed{};
};

struct PolicyEvaluationSummary final {
  std::string scenario_label;
  std::string policy_name;
  std::size_t episodes{};
  double mean_net_pnl_currency{};
  double std_net_pnl_currency{};
  double mean_objective_reward_currency{};
  double mean_max_drawdown_currency{};
  double mean_fill_rate{};
  double mean_max_inventory_lots{};
};

struct PolicyOptimizationStudyConfig final {
  std::vector<PolicyOptimizationScenario> training_scenarios;
  std::vector<PolicyOptimizationScenario> validation_scenarios;
  std::vector<PolicyOptimizationScenario> out_of_sample_scenarios;
  PolicyGradientConfig policy_gradient;
  SurrogateTrainingConfig surrogate;
  AvellanedaStoikovConfig avellaneda_stoikov;
  ReducedStateConfig dynamic_programming;
};

struct PolicyOptimizationStudyResult final {
  PolicyGradientTrainingResult policy_gradient;
  SurrogateTrainingResult surrogate;
  std::vector<PolicyEvaluationSummary> out_of_sample;
};

/// Trains a masked categorical actor against the actual discrete simulator
/// using episode-return REINFORCE; no gradient is propagated through exchange
/// events. Training and validation scenarios must be disjoint from OOS seeds.
[[nodiscard]] PolicyGradientTrainingResult train_simulator_policy_gradient(
    const std::vector<PolicyOptimizationScenario>& training_scenarios,
    const std::vector<PolicyOptimizationScenario>& validation_scenarios,
    const PolicyGradientConfig& config = {},
    PolicyGradientParameters initial_parameters = {});

/// Collects causal, no-agent market observations from the exact simulator and
/// performs early-stopped differentiable surrogate optimization.
[[nodiscard]] SurrogateTrainingResult train_differentiable_policy_surrogate(
    const std::vector<PolicyOptimizationScenario>& training_scenarios,
    const std::vector<PolicyOptimizationScenario>& validation_scenarios,
    const SurrogateTrainingConfig& config = {},
    SurrogateParameters initial_parameters = {});

/// Trains both approaches, then compares them on held-out exact simulator runs
/// alongside classical Avellaneda-Stoikov and reduced-state DP agents.
[[nodiscard]] PolicyOptimizationStudyResult run_policy_optimization_study(
    const PolicyOptimizationStudyConfig& config);

}  // namespace project3
