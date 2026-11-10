#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace project3 {

inline constexpr std::size_t kSurrogateParameterCount = 22;

struct SurrogateParameters final {
  // Bid/ask intercepts, inventory skew, queue sensitivity, intensity
  // sensitivity, and side-specific adverse-selection sensitivity.
  std::array<double, kSurrogateParameterCount> values{};
};

struct SurrogateMarketStep final {
  double duration_seconds{1.0};
  double market_buy_intensity_per_second{1.0};
  double market_sell_intensity_per_second{1.0};
  double bid_queue_ahead_lots{};
  double ask_queue_ahead_lots{};
  double bid_adverse_selection_ticks{};
  double ask_adverse_selection_ticks{};
  double midprice_return_ticks{};
  double midprice_feature{};
  double spread_ticks{1.0};
  double bid_depth_lots{};
  double ask_depth_lots{};
  std::array<double, 6> hawkes_intensities{};
  double volatility_ticks_per_sqrt_second{};
  double time_remaining_fraction{1.0};
};

struct SurrogatePath final {
  double initial_inventory_lots{};
  std::vector<SurrogateMarketStep> steps;
};

struct DifferentiableSurrogateConfig final {
  double min_quote_offset_ticks{0.0};
  double max_quote_offset_ticks{5.0};
  double inventory_limit_lots{10.0};
  double queue_feature_scale_lots{5.0};
  double intensity_feature_scale{1.0};
  double depth_feature_scale{5.0};
  double volatility_feature_scale{1.0};
  double fill_logit_intercept{-0.5};
  double fill_offset_coefficient{-0.8};
  double fill_queue_coefficient{-1.0};
  double fill_log_intensity_coefficient{0.5};
  double maximum_executed_quantity_lots{1.0};
  double smooth_inventory_temperature_lots{0.5};
  double running_inventory_penalty_currency_per_second_lot_squared{0.01};
  double terminal_inventory_penalty_currency_per_lot_squared{0.1};
  double fee_currency_per_lot{0.0};
  double currency_per_tick{1.0};
  double parameter_absolute_limit{20.0};
  /// Independent path batches are evaluated on contiguous chunks; values
  /// above one enable deterministic-order parallel reduction. The default
  /// preserves single-threaded bit-exact results.
  std::size_t evaluation_worker_count{1};
};

struct SurrogateEvaluation final {
  double expected_reward_currency{};
  double spread_capture_currency{};
  double inventory_pnl_currency{};
  double adverse_selection_cost_currency{};
  double fees_currency{};
  double mean_fill_probability{};
  double expected_executed_quantity_lots{};
  double mean_terminal_inventory_lots{};
  std::size_t decision_count{};
};

struct SurrogateDecision final {
  double bid_offset_ticks{};
  double ask_offset_ticks{};
  double bid_fill_probability{};
  double ask_fill_probability{};
  double expected_bid_quantity_lots{};
  double expected_ask_quantity_lots{};
  double suggested_bid_size_lots{};
  double suggested_ask_size_lots{};
  double bid_cancel_probability{};
  double ask_cancel_probability{};
  double next_inventory_lots{};
};

struct SurrogateGradient final {
  double expected_reward_currency{};
  std::array<double, kSurrogateParameterCount> gradient{};
};

struct GradientCheck final {
  std::array<double, kSurrogateParameterCount> analytic{};
  std::array<double, kSurrogateParameterCount> finite_difference{};
  std::array<double, kSurrogateParameterCount> relative_error{};
  double maximum_relative_error{};
};

enum class SurrogateOptimizer {
  GradientDescent,
  Adam,
};

enum class LearningRateSchedule {
  Constant,
  LinearDecay,
  CosineDecay,
};

struct SurrogateOptimizerConfig final {
  SurrogateOptimizer optimizer{SurrogateOptimizer::Adam};
  LearningRateSchedule learning_rate_schedule{LearningRateSchedule::CosineDecay};
  std::size_t iterations{500};
  double learning_rate{0.03};
  double minimum_learning_rate{0.001};
  double gradient_clip_norm{5.0};
  double adam_beta1{0.9};
  double adam_beta2{0.999};
  double adam_epsilon{1.0e-8};
  // Experiment identifier; stochasticity is confined to the fixed input batch.
  std::uint64_t seed{1};
  /// Zero keeps the serial reference evaluation; values above one enable
  /// deterministic chunked parallel path evaluation.
  std::size_t worker_count{0};
};

struct SurrogateOptimizationStep final {
  std::size_t iteration{};
  double expected_reward_currency{};
  double gradient_norm{};
  double learning_rate{};
};

struct SurrogateOptimizationResult final {
  SurrogateParameters parameters;
  double initial_reward_currency{};
  double final_reward_currency{};
  std::uint64_t seed{};
  std::vector<SurrogateOptimizationStep> trace;
};

/// Evaluates a smooth sample-average reward; it is not the event-driven
/// exchange simulator's reward or exact execution model.
[[nodiscard]] SurrogateEvaluation evaluate_differentiable_surrogate(
    const SurrogateParameters& parameters,
    const std::vector<SurrogatePath>& paths,
    const DifferentiableSurrogateConfig& config = {});

/// Applies learned parameters to one causal observation. This returns smooth
/// predictions, not exchange orders or fill guarantees.
[[nodiscard]] SurrogateDecision smooth_surrogate_decision(
    const SurrogateParameters& parameters,
    double inventory_lots,
    const SurrogateMarketStep& observation,
    const DifferentiableSurrogateConfig& config = {});

/// Forward-mode automatic differentiation through the smooth surrogate.
[[nodiscard]] SurrogateGradient differentiable_surrogate_gradient(
    const SurrogateParameters& parameters,
    const std::vector<SurrogatePath>& paths,
    const DifferentiableSurrogateConfig& config = {});

[[nodiscard]] std::array<double, kSurrogateParameterCount>
finite_difference_surrogate_gradient(
    const SurrogateParameters& parameters,
    const std::vector<SurrogatePath>& paths,
    const DifferentiableSurrogateConfig& config = {},
    double relative_step = 1.0e-5);

[[nodiscard]] GradientCheck check_surrogate_gradient(
    const SurrogateParameters& parameters,
    const std::vector<SurrogatePath>& paths,
    const DifferentiableSurrogateConfig& config = {},
    double relative_step = 1.0e-5,
    double denominator_epsilon = 1.0e-12);

/// Creates a fixed, reproducible exogenous batch. Optimization is deterministic
/// for a fixed batch, configuration, and seed; the seed is retained in results.
[[nodiscard]] std::vector<SurrogatePath> make_surrogate_training_batch(
    std::uint64_t seed,
    std::size_t path_count,
    std::size_t steps_per_path);

/// Maximizes the sample-average reward using gradient ascent (equivalent to
/// descent on negative reward), optional global-norm clipping, and Adam or SGD.
[[nodiscard]] SurrogateOptimizationResult optimize_differentiable_surrogate(
    SurrogateParameters initial_parameters,
    const std::vector<SurrogatePath>& paths,
    const DifferentiableSurrogateConfig& surrogate_config = {},
    const SurrogateOptimizerConfig& optimizer_config = {});

}  // namespace project3
