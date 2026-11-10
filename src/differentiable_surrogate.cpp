#include "project3/differentiable_surrogate.hpp"

#include "project3/detail/parallel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
#include <type_traits>

namespace project3 {
namespace {

using Derivatives = std::array<double, kSurrogateParameterCount>;

struct Dual final {
  double value{};
  Derivatives derivative{};

  static Dual constant(const double value) { return {.value = value}; }
  static Dual variable(const double value, const std::size_t index) {
    Dual result{.value = value};
    result.derivative[index] = 1.0;
    return result;
  }
};

Dual operator+(const Dual& left, const Dual& right) {
  Dual result{.value = left.value + right.value};
  for (std::size_t i = 0; i < kSurrogateParameterCount; ++i) {
    result.derivative[i] = left.derivative[i] + right.derivative[i];
  }
  return result;
}

Dual operator+(const double left, const Dual& right) {
  Dual result = right;
  result.value += left;
  return result;
}

Dual operator+(const Dual& left, const double right) {
  Dual result = left;
  result.value += right;
  return result;
}

Dual operator-(const Dual& left, const Dual& right) {
  Dual result{.value = left.value - right.value};
  for (std::size_t i = 0; i < kSurrogateParameterCount; ++i) {
    result.derivative[i] = left.derivative[i] - right.derivative[i];
  }
  return result;
}

Dual operator-(const double left, const Dual& right) {
  Dual result{.value = left - right.value};
  for (std::size_t i = 0; i < kSurrogateParameterCount; ++i) {
    result.derivative[i] = -right.derivative[i];
  }
  return result;
}

Dual operator*(const Dual& left, const Dual& right) {
  Dual result{.value = left.value * right.value};
  for (std::size_t i = 0; i < kSurrogateParameterCount; ++i) {
    result.derivative[i] =
        left.derivative[i] * right.value +
        left.value * right.derivative[i];
  }
  return result;
}

Dual operator*(const Dual& left, const double right) {
  Dual result{.value = left.value * right};
  for (std::size_t i = 0; i < kSurrogateParameterCount; ++i) {
    result.derivative[i] = left.derivative[i] * right;
  }
  return result;
}

Dual operator*(const double left, const Dual& right) { return right * left; }

Dual operator/(const Dual& left, const double right) {
  Dual result{.value = left.value / right};
  for (std::size_t i = 0; i < kSurrogateParameterCount; ++i) {
    result.derivative[i] = left.derivative[i] / right;
  }
  return result;
}

Dual sigmoid(const Dual& input) {
  double result_value = 0.0;
  if (input.value >= 0.0) {
    const double exp_negative = std::exp(-input.value);
    result_value = 1.0 / (1.0 + exp_negative);
  } else {
    const double exp_positive = std::exp(input.value);
    result_value = exp_positive / (1.0 + exp_positive);
  }
  Dual result{.value = result_value};
  const double slope = result_value * (1.0 - result_value);
  for (std::size_t i = 0; i < kSurrogateParameterCount; ++i) {
    result.derivative[i] = slope * input.derivative[i];
  }
  return result;
}

Dual square(const Dual& value) { return value * value; }

template <typename Scalar>
struct GenericEvaluation final {
  Scalar reward{};
  Scalar spread_capture{};
  Scalar inventory_pnl{};
  Scalar adverse_selection_cost{};
  Scalar fees{};
  Scalar fill_probability_sum{};
  Scalar executed_quantity_sum{};
  Scalar terminal_inventory_sum{};
  std::size_t decision_count{};
  std::size_t path_count{};
};

void validate_config(const DifferentiableSurrogateConfig& config) {
  const auto finite_nonnegative = [](const double value) {
    return std::isfinite(value) && value >= 0.0;
  };
  if (!std::isfinite(config.min_quote_offset_ticks) ||
      !std::isfinite(config.max_quote_offset_ticks) ||
      config.min_quote_offset_ticks < 0.0 ||
      config.max_quote_offset_ticks <= config.min_quote_offset_ticks ||
      !finite_nonnegative(config.inventory_limit_lots) ||
      config.inventory_limit_lots <= 0.0 ||
      !std::isfinite(config.queue_feature_scale_lots) ||
      config.queue_feature_scale_lots <= 0.0 ||
      !std::isfinite(config.intensity_feature_scale) ||
      config.intensity_feature_scale <= 0.0 ||
      !std::isfinite(config.depth_feature_scale) ||
      config.depth_feature_scale <= 0.0 ||
      !std::isfinite(config.volatility_feature_scale) ||
      config.volatility_feature_scale <= 0.0 ||
      !std::isfinite(config.fill_logit_intercept) ||
      !std::isfinite(config.fill_offset_coefficient) ||
      !std::isfinite(config.fill_queue_coefficient) ||
      !std::isfinite(config.fill_log_intensity_coefficient) ||
      !std::isfinite(config.maximum_executed_quantity_lots) ||
      config.maximum_executed_quantity_lots <= 0.0 ||
      !std::isfinite(config.smooth_inventory_temperature_lots) ||
      config.smooth_inventory_temperature_lots <= 0.0 ||
      !finite_nonnegative(
          config.running_inventory_penalty_currency_per_second_lot_squared) ||
      !finite_nonnegative(
          config.terminal_inventory_penalty_currency_per_lot_squared) ||
      !finite_nonnegative(config.fee_currency_per_lot) ||
      !std::isfinite(config.currency_per_tick) ||
      config.currency_per_tick <= 0.0 ||
      !std::isfinite(config.parameter_absolute_limit) ||
      config.parameter_absolute_limit <= 0.0) {
    throw std::invalid_argument{"invalid differentiable-surrogate configuration"};
  }
}

void validate_input(const SurrogateParameters& parameters,
                    const std::vector<SurrogatePath>& paths,
                    const DifferentiableSurrogateConfig& config) {
  validate_config(config);
  if (paths.empty()) {
    throw std::invalid_argument{"surrogate evaluation requires at least one path"};
  }
  for (const double parameter : parameters.values) {
    if (!std::isfinite(parameter)) {
      throw std::invalid_argument{"surrogate policy parameter is not finite"};
    }
  }
  for (const auto& path : paths) {
    if (path.steps.empty() || !std::isfinite(path.initial_inventory_lots) ||
        std::abs(path.initial_inventory_lots) > config.inventory_limit_lots) {
      throw std::invalid_argument{"surrogate path is empty or has invalid inventory"};
    }
    for (const auto& step : path.steps) {
      if (!std::isfinite(step.duration_seconds) ||
          step.duration_seconds <= 0.0 ||
          !std::isfinite(step.market_buy_intensity_per_second) ||
          step.market_buy_intensity_per_second < 0.0 ||
          !std::isfinite(step.market_sell_intensity_per_second) ||
          step.market_sell_intensity_per_second < 0.0 ||
          !std::isfinite(step.bid_queue_ahead_lots) ||
          step.bid_queue_ahead_lots < 0.0 ||
          !std::isfinite(step.ask_queue_ahead_lots) ||
          step.ask_queue_ahead_lots < 0.0 ||
          !std::isfinite(step.bid_adverse_selection_ticks) ||
          step.bid_adverse_selection_ticks < 0.0 ||
          !std::isfinite(step.ask_adverse_selection_ticks) ||
          step.ask_adverse_selection_ticks < 0.0 ||
          !std::isfinite(step.midprice_return_ticks) ||
          !std::isfinite(step.midprice_feature) ||
          !std::isfinite(step.spread_ticks) || step.spread_ticks < 0.0 ||
          !std::isfinite(step.bid_depth_lots) || step.bid_depth_lots < 0.0 ||
          !std::isfinite(step.ask_depth_lots) || step.ask_depth_lots < 0.0 ||
          !std::isfinite(step.volatility_ticks_per_sqrt_second) ||
          step.volatility_ticks_per_sqrt_second < 0.0 ||
          !std::isfinite(step.time_remaining_fraction) ||
          step.time_remaining_fraction < 0.0 ||
          step.time_remaining_fraction > 1.0) {
        throw std::invalid_argument{"surrogate path contains an invalid market step"};
      }
      for (const double intensity : step.hawkes_intensities) {
        if (!std::isfinite(intensity) || intensity < 0.0) {
          throw std::invalid_argument{
              "surrogate path contains an invalid Hawkes intensity"};
        }
      }
    }
  }
}

template <typename Scalar>
Scalar parameter(const SurrogateParameters& parameters,
                 const std::size_t index) {
  if constexpr (std::is_same_v<Scalar, Dual>) {
    return Dual::variable(parameters.values[index], index);
  } else {
    return parameters.values[index];
  }
}

template <typename Scalar>
Scalar logistic(const Scalar& input) {
  if constexpr (std::is_same_v<Scalar, Dual>) {
    return sigmoid(input);
  } else {
    if (input >= 0.0) {
      return 1.0 / (1.0 + std::exp(-input));
    }
    const double positive = std::exp(input);
    return positive / (1.0 + positive);
  }
}

template <typename Scalar>
Scalar scalar_square(const Scalar& input) {
  if constexpr (std::is_same_v<Scalar, Dual>) {
    return square(input);
  } else {
    return input * input;
  }
}

template <typename Scalar>
struct Decision final {
  Scalar bid_offset;
  Scalar ask_offset;
  Scalar bid_probability;
  Scalar ask_probability;
  Scalar bid_quantity;
  Scalar ask_quantity;
  Scalar bid_size;
  Scalar ask_size;
  Scalar bid_cancel_probability;
  Scalar ask_cancel_probability;
  Scalar next_inventory;
};

template <typename Scalar>
Decision<Scalar> make_decision(const SurrogateParameters& parameters,
                               const Scalar& inventory,
                               const SurrogateMarketStep& step,
                               const DifferentiableSurrogateConfig& config) {
  const Scalar theta_bid = parameter<Scalar>(parameters, 0);
  const Scalar theta_ask = parameter<Scalar>(parameters, 1);
  const Scalar theta_inventory = parameter<Scalar>(parameters, 2);
  const Scalar theta_queue = parameter<Scalar>(parameters, 3);
  const Scalar theta_intensity = parameter<Scalar>(parameters, 4);
  const Scalar theta_adverse = parameter<Scalar>(parameters, 5);
  std::array<Scalar, 6> theta_hawkes{};
  for (std::size_t i = 0; i < theta_hawkes.size(); ++i) {
    theta_hawkes[i] = parameter<Scalar>(parameters, 6 + i);
  }
  const Scalar theta_spread = parameter<Scalar>(parameters, 12);
  const Scalar theta_bid_depth = parameter<Scalar>(parameters, 13);
  const Scalar theta_ask_depth = parameter<Scalar>(parameters, 14);
  const Scalar theta_volatility = parameter<Scalar>(parameters, 15);
  const Scalar theta_time = parameter<Scalar>(parameters, 16);
  const Scalar theta_bid_size = parameter<Scalar>(parameters, 17);
  const Scalar theta_ask_size = parameter<Scalar>(parameters, 18);
  const Scalar theta_bid_cancel = parameter<Scalar>(parameters, 19);
  const Scalar theta_ask_cancel = parameter<Scalar>(parameters, 20);
  const Scalar theta_midprice = parameter<Scalar>(parameters, 21);
  const Scalar inventory_normalized =
      inventory / config.inventory_limit_lots;
  const double bid_queue =
      step.bid_queue_ahead_lots / config.queue_feature_scale_lots;
  const double ask_queue =
      step.ask_queue_ahead_lots / config.queue_feature_scale_lots;
  const double bid_log_intensity =
      std::log1p(step.market_sell_intensity_per_second) /
      config.intensity_feature_scale;
  const double ask_log_intensity =
      std::log1p(step.market_buy_intensity_per_second) /
      config.intensity_feature_scale;
  const double adverse_scale =
      std::max(1.0, config.max_quote_offset_ticks -
                        config.min_quote_offset_ticks);
  const double bid_adverse =
      step.bid_adverse_selection_ticks / adverse_scale;
  const double ask_adverse =
      step.ask_adverse_selection_ticks / adverse_scale;
  const double spread_feature =
      step.spread_ticks / adverse_scale;
  const double bid_depth_feature =
      std::log1p(step.bid_depth_lots) / config.depth_feature_scale;
  const double ask_depth_feature =
      std::log1p(step.ask_depth_lots) / config.depth_feature_scale;
  const double volatility_feature =
      std::log1p(step.volatility_ticks_per_sqrt_second) /
      config.volatility_feature_scale;
  Scalar hawkes_feature_sum{};
  for (std::size_t i = 0; i < theta_hawkes.size(); ++i) {
    hawkes_feature_sum =
        hawkes_feature_sum +
        theta_hawkes[i] *
            (std::log1p(step.hawkes_intensities[i]) /
             config.intensity_feature_scale);
  }
  const Scalar bid_latent =
      theta_bid + theta_inventory * inventory_normalized +
      theta_queue * bid_queue + theta_intensity * bid_log_intensity +
      theta_adverse * bid_adverse + hawkes_feature_sum +
      theta_spread * spread_feature +
      theta_bid_depth * bid_depth_feature +
      theta_volatility * volatility_feature +
      theta_time * step.time_remaining_fraction +
      theta_midprice * step.midprice_feature;
  const Scalar ask_latent =
      theta_ask - theta_inventory * inventory_normalized +
      theta_queue * ask_queue + theta_intensity * ask_log_intensity +
      theta_adverse * ask_adverse + hawkes_feature_sum +
      theta_spread * spread_feature +
      theta_ask_depth * ask_depth_feature +
      theta_volatility * volatility_feature +
      theta_time * step.time_remaining_fraction +
      theta_midprice * step.midprice_feature;
  const Scalar bid_offset =
      config.min_quote_offset_ticks +
      (config.max_quote_offset_ticks - config.min_quote_offset_ticks) *
          logistic(bid_latent);
  const Scalar ask_offset =
      config.min_quote_offset_ticks +
      (config.max_quote_offset_ticks - config.min_quote_offset_ticks) *
          logistic(ask_latent);
  const double bid_fill_logit =
      config.fill_logit_intercept +
      config.fill_queue_coefficient * bid_queue +
      config.fill_log_intensity_coefficient * bid_log_intensity;
  const double ask_fill_logit =
      config.fill_logit_intercept +
      config.fill_queue_coefficient * ask_queue +
      config.fill_log_intensity_coefficient * ask_log_intensity;
  const Scalar bid_probability =
      logistic(config.fill_offset_coefficient * bid_offset + bid_fill_logit);
  const Scalar ask_probability =
      logistic(config.fill_offset_coefficient * ask_offset + ask_fill_logit);
  const Scalar bid_capacity = logistic(
      (config.inventory_limit_lots - inventory) /
      config.smooth_inventory_temperature_lots);
  const Scalar ask_capacity = logistic(
      (config.inventory_limit_lots + inventory) /
      config.smooth_inventory_temperature_lots);
  const Scalar bid_size =
      config.maximum_executed_quantity_lots * logistic(theta_bid_size);
  const Scalar ask_size =
      config.maximum_executed_quantity_lots * logistic(theta_ask_size);
  const Scalar bid_cancel_probability = logistic(theta_bid_cancel);
  const Scalar ask_cancel_probability = logistic(theta_ask_cancel);
  const Scalar bid_quantity =
      bid_size * bid_probability * bid_capacity *
      (1.0 - bid_cancel_probability);
  const Scalar ask_quantity =
      ask_size * ask_probability * ask_capacity *
      (1.0 - ask_cancel_probability);
  return {
      .bid_offset = bid_offset,
      .ask_offset = ask_offset,
      .bid_probability = bid_probability,
      .ask_probability = ask_probability,
      .bid_quantity = bid_quantity,
      .ask_quantity = ask_quantity,
      .bid_size = bid_size,
      .ask_size = ask_size,
      .bid_cancel_probability = bid_cancel_probability,
      .ask_cancel_probability = ask_cancel_probability,
      .next_inventory = inventory + bid_quantity - ask_quantity,
  };
}

template <typename Scalar>
GenericEvaluation<Scalar> merge_evaluations(
    const std::vector<GenericEvaluation<Scalar>>& parts) {
  GenericEvaluation<Scalar> totals;
  for (const auto& part : parts) {
    totals.reward = totals.reward + part.reward;
    totals.spread_capture = totals.spread_capture + part.spread_capture;
    totals.inventory_pnl = totals.inventory_pnl + part.inventory_pnl;
    totals.adverse_selection_cost =
        totals.adverse_selection_cost + part.adverse_selection_cost;
    totals.fees = totals.fees + part.fees;
    totals.fill_probability_sum =
        totals.fill_probability_sum + part.fill_probability_sum;
    totals.executed_quantity_sum =
        totals.executed_quantity_sum + part.executed_quantity_sum;
    totals.terminal_inventory_sum =
        totals.terminal_inventory_sum + part.terminal_inventory_sum;
    totals.decision_count += part.decision_count;
    totals.path_count += part.path_count;
  }
  return totals;
}

template <typename Scalar>
void normalize_evaluation(GenericEvaluation<Scalar>& totals) {
  const double path_count = static_cast<double>(totals.path_count);
  totals.reward = totals.reward / path_count;
  totals.spread_capture = totals.spread_capture / path_count;
  totals.inventory_pnl = totals.inventory_pnl / path_count;
  totals.adverse_selection_cost = totals.adverse_selection_cost / path_count;
  totals.fees = totals.fees / path_count;
  totals.terminal_inventory_sum = totals.terminal_inventory_sum / path_count;
  totals.fill_probability_sum =
      totals.fill_probability_sum /
      static_cast<double>(totals.decision_count);
  totals.executed_quantity_sum = totals.executed_quantity_sum / path_count;
}

template <typename Scalar>
GenericEvaluation<Scalar> evaluate_paths(
    const SurrogateParameters& parameters,
    const SurrogatePath* begin,
    const SurrogatePath* end,
    const DifferentiableSurrogateConfig& config) {
  GenericEvaluation<Scalar> totals;
  for (const SurrogatePath* path = begin; path != end; ++path) {
    Scalar inventory = [&] {
      if constexpr (std::is_same_v<Scalar, Dual>) {
        return Dual::constant(path->initial_inventory_lots);
      } else {
        return path->initial_inventory_lots;
      }
    }();
    Scalar path_reward = [&] {
      if constexpr (std::is_same_v<Scalar, Dual>) {
        return Dual::constant(0.0);
      } else {
        return 0.0;
      }
    }();

    for (const auto& step : path->steps) {
      const auto decision =
          make_decision<Scalar>(parameters, inventory, step, config);
      const Scalar spread_capture =
          (decision.bid_quantity * decision.bid_offset +
           decision.ask_quantity * decision.ask_offset) *
          config.currency_per_tick;
      const Scalar inventory_pnl =
          inventory * (step.midprice_return_ticks * config.currency_per_tick);
      const Scalar adverse_cost =
          (decision.bid_quantity * step.bid_adverse_selection_ticks +
           decision.ask_quantity * step.ask_adverse_selection_ticks) *
          config.currency_per_tick;
      const Scalar fees =
          (decision.bid_quantity + decision.ask_quantity) *
          config.fee_currency_per_lot;
      const Scalar running_penalty =
          scalar_square(decision.next_inventory) *
          (config.running_inventory_penalty_currency_per_second_lot_squared *
           step.duration_seconds);
      path_reward = path_reward + spread_capture + inventory_pnl - adverse_cost -
                    fees - running_penalty;
      inventory = decision.next_inventory;

      totals.spread_capture = totals.spread_capture + spread_capture;
      totals.inventory_pnl = totals.inventory_pnl + inventory_pnl;
      totals.adverse_selection_cost =
          totals.adverse_selection_cost + adverse_cost;
      totals.fees = totals.fees + fees;
      totals.fill_probability_sum =
          totals.fill_probability_sum + decision.bid_probability +
          decision.ask_probability;
      totals.executed_quantity_sum =
          totals.executed_quantity_sum + decision.bid_quantity +
          decision.ask_quantity;
      totals.decision_count += 2;
    }
    path_reward =
        path_reward -
        scalar_square(inventory) *
            config.terminal_inventory_penalty_currency_per_lot_squared;
    totals.reward = totals.reward + path_reward;
    totals.terminal_inventory_sum =
        totals.terminal_inventory_sum + inventory;
    ++totals.path_count;
  }
  return totals;
}

template <typename Scalar>
GenericEvaluation<Scalar> evaluate(
    const SurrogateParameters& parameters,
    const std::vector<SurrogatePath>& paths,
    const DifferentiableSurrogateConfig& config) {
  const auto requested = config.evaluation_worker_count;
  std::size_t total_steps = 0;
  for (const auto& path : paths) total_steps += path.steps.size();
  // Very small batches keep the serial reference path; the persistent pool
  // makes larger batches cheap to dispatch.
  const auto workers =
      requested <= 1 || total_steps < 2048
          ? std::size_t{1}
          : std::min(detail::resolve_worker_count(requested), paths.size());
  if (workers <= 1) {
    auto totals = evaluate_paths<Scalar>(parameters, paths.data(),
                                         paths.data() + paths.size(), config);
    normalize_evaluation(totals);
    return totals;
  }
  // Contiguous chunks keep the reduction order deterministic for a fixed
  // worker count; only the last-bit summation order differs from serial.
  std::vector<GenericEvaluation<Scalar>> parts(workers);
  const std::size_t chunk =
      (paths.size() + workers - 1) / workers;
  detail::ChunkedPool::instance().run(workers, workers,
                                      [&](const std::size_t index) {
    const std::size_t first = index * chunk;
    if (first >= paths.size()) {
      parts[index] = GenericEvaluation<Scalar>{};
      return;
    }
    const std::size_t last = std::min(paths.size(), first + chunk);
    parts[index] = evaluate_paths<Scalar>(
        parameters, paths.data() + first, paths.data() + last, config);
  });
  auto totals = merge_evaluations(parts);
  normalize_evaluation(totals);
  return totals;
}

void validate_parameters(const SurrogateParameters& parameters,
                         const double limit) {
  for (const double value : parameters.values) {
    if (!std::isfinite(value) || std::abs(value) > limit) {
      throw std::invalid_argument{
          "surrogate parameter is non-finite or outside configured bounds"};
    }
  }
}

double norm(const std::array<double, kSurrogateParameterCount>& values) {
  double result = 0.0;
  for (const double value : values) result = std::hypot(result, value);
  return result;
}

double scheduled_rate(const SurrogateOptimizerConfig& config,
                      const std::size_t iteration) {
  if (config.iterations <= 1 ||
      config.learning_rate_schedule == LearningRateSchedule::Constant) {
    return config.learning_rate;
  }
  const double fraction =
      static_cast<double>(iteration) /
      static_cast<double>(config.iterations - 1);
  if (config.learning_rate_schedule == LearningRateSchedule::LinearDecay) {
    return config.learning_rate +
           (config.minimum_learning_rate - config.learning_rate) * fraction;
  }
  const double weight = 0.5 * (1.0 + std::cos(std::acos(-1.0) * fraction));
  return config.minimum_learning_rate +
         (config.learning_rate - config.minimum_learning_rate) * weight;
}

}  // namespace

SurrogateEvaluation evaluate_differentiable_surrogate(
    const SurrogateParameters& parameters,
    const std::vector<SurrogatePath>& paths,
    const DifferentiableSurrogateConfig& config) {
  validate_input(parameters, paths, config);
  validate_parameters(parameters, config.parameter_absolute_limit);
  const auto result = evaluate<double>(parameters, paths, config);
  if (!std::isfinite(result.reward) ||
      !std::isfinite(result.spread_capture) ||
      !std::isfinite(result.inventory_pnl) ||
      !std::isfinite(result.adverse_selection_cost) ||
      !std::isfinite(result.fees) ||
      !std::isfinite(result.fill_probability_sum) ||
      !std::isfinite(result.executed_quantity_sum) ||
      !std::isfinite(result.terminal_inventory_sum)) {
    throw std::overflow_error{"surrogate evaluation produced a non-finite result"};
  }
  return {
      .expected_reward_currency = result.reward,
      .spread_capture_currency = result.spread_capture,
      .inventory_pnl_currency = result.inventory_pnl,
      .adverse_selection_cost_currency = result.adverse_selection_cost,
      .fees_currency = result.fees,
      .mean_fill_probability = result.fill_probability_sum,
      .expected_executed_quantity_lots = result.executed_quantity_sum,
      .mean_terminal_inventory_lots = result.terminal_inventory_sum,
      .decision_count = result.decision_count,
  };
}

SurrogateDecision smooth_surrogate_decision(
    const SurrogateParameters& parameters,
    const double inventory_lots,
    const SurrogateMarketStep& observation,
    const DifferentiableSurrogateConfig& config) {
  const std::vector<SurrogatePath> one_step_path{
      SurrogatePath{.initial_inventory_lots = inventory_lots,
                    .steps = {observation}}};
  validate_input(parameters, one_step_path, config);
  validate_parameters(parameters, config.parameter_absolute_limit);
  const auto decision = make_decision<double>(
      parameters, inventory_lots, observation, config);
  if (!std::isfinite(decision.bid_offset) ||
      !std::isfinite(decision.ask_offset) ||
      !std::isfinite(decision.bid_probability) ||
      !std::isfinite(decision.ask_probability) ||
      !std::isfinite(decision.bid_quantity) ||
      !std::isfinite(decision.ask_quantity) ||
      !std::isfinite(decision.bid_size) ||
      !std::isfinite(decision.ask_size) ||
      !std::isfinite(decision.bid_cancel_probability) ||
      !std::isfinite(decision.ask_cancel_probability) ||
      !std::isfinite(decision.next_inventory)) {
    throw std::overflow_error{"surrogate decision produced a non-finite result"};
  }
  return {
      .bid_offset_ticks = decision.bid_offset,
      .ask_offset_ticks = decision.ask_offset,
      .bid_fill_probability = decision.bid_probability,
      .ask_fill_probability = decision.ask_probability,
      .expected_bid_quantity_lots = decision.bid_quantity,
      .expected_ask_quantity_lots = decision.ask_quantity,
      .suggested_bid_size_lots = decision.bid_size,
      .suggested_ask_size_lots = decision.ask_size,
      .bid_cancel_probability = decision.bid_cancel_probability,
      .ask_cancel_probability = decision.ask_cancel_probability,
      .next_inventory_lots = decision.next_inventory,
  };
}

SurrogateGradient differentiable_surrogate_gradient(
    const SurrogateParameters& parameters,
    const std::vector<SurrogatePath>& paths,
    const DifferentiableSurrogateConfig& config) {
  validate_input(parameters, paths, config);
  validate_parameters(parameters, config.parameter_absolute_limit);
  const auto result = evaluate<Dual>(parameters, paths, config).reward;
  if (!std::isfinite(result.value)) {
    throw std::overflow_error{"surrogate gradient objective is not finite"};
  }
  for (const double component : result.derivative) {
    if (!std::isfinite(component)) {
      throw std::overflow_error{"surrogate gradient is not finite"};
    }
  }
  return {.expected_reward_currency = result.value,
          .gradient = result.derivative};
}

std::array<double, kSurrogateParameterCount>
finite_difference_surrogate_gradient(
    const SurrogateParameters& parameters,
    const std::vector<SurrogatePath>& paths,
    const DifferentiableSurrogateConfig& config,
    const double relative_step) {
  validate_input(parameters, paths, config);
  validate_parameters(parameters, config.parameter_absolute_limit);
  if (!std::isfinite(relative_step) || relative_step <= 0.0) {
    throw std::invalid_argument{"finite-difference step must be positive and finite"};
  }
  std::array<double, kSurrogateParameterCount> gradient{};
  for (std::size_t i = 0; i < kSurrogateParameterCount; ++i) {
    const double step =
        relative_step * std::max(1.0, std::abs(parameters.values[i]));
    auto plus = parameters;
    auto minus = parameters;
    plus.values[i] = std::min(config.parameter_absolute_limit,
                              parameters.values[i] + step);
    minus.values[i] = std::max(-config.parameter_absolute_limit,
                               parameters.values[i] - step);
    const double interval = plus.values[i] - minus.values[i];
    if (interval <= 0.0) {
      throw std::invalid_argument{
          "finite-difference interval is zero at the parameter constraint"};
    }
    const double reward_plus =
        evaluate_differentiable_surrogate(plus, paths, config)
            .expected_reward_currency;
    const double reward_minus =
        evaluate_differentiable_surrogate(minus, paths, config)
            .expected_reward_currency;
    gradient[i] = (reward_plus - reward_minus) / interval;
  }
  return gradient;
}

GradientCheck check_surrogate_gradient(
    const SurrogateParameters& parameters,
    const std::vector<SurrogatePath>& paths,
    const DifferentiableSurrogateConfig& config,
    const double relative_step,
    const double denominator_epsilon) {
  if (!std::isfinite(denominator_epsilon) || denominator_epsilon <= 0.0) {
    throw std::invalid_argument{"gradient-check epsilon must be positive and finite"};
  }
  const auto analytic =
      differentiable_surrogate_gradient(parameters, paths, config).gradient;
  const auto finite =
      finite_difference_surrogate_gradient(parameters, paths, config,
                                           relative_step);
  GradientCheck result;
  result.analytic = analytic;
  result.finite_difference = finite;
  for (std::size_t i = 0; i < kSurrogateParameterCount; ++i) {
    const double denominator =
        std::max({denominator_epsilon, std::abs(analytic[i]),
                  std::abs(finite[i])});
    result.relative_error[i] = std::abs(analytic[i] - finite[i]) / denominator;
    result.maximum_relative_error =
        std::max(result.maximum_relative_error, result.relative_error[i]);
  }
  return result;
}

std::vector<SurrogatePath> make_surrogate_training_batch(
    const std::uint64_t seed,
    const std::size_t path_count,
    const std::size_t steps_per_path) {
  if (path_count == 0 || steps_per_path == 0) {
    throw std::invalid_argument{"surrogate batch dimensions must be nonzero"};
  }
  std::mt19937_64 generator{seed};
  std::uniform_real_distribution<double> intensity_log{-1.0, 2.0};
  std::uniform_real_distribution<double> queue{0.0, 10.0};
  std::uniform_real_distribution<double> adverse{0.0, 1.5};
  std::normal_distribution<double> mid_return{0.0, 0.25};
  std::vector<SurrogatePath> paths(path_count);
  for (auto& path : paths) {
    path.steps.reserve(steps_per_path);
    for (std::size_t i = 0; i < steps_per_path; ++i) {
      path.steps.push_back({
          .duration_seconds = 1.0,
          .market_buy_intensity_per_second = std::exp(intensity_log(generator)),
          .market_sell_intensity_per_second = std::exp(intensity_log(generator)),
          .bid_queue_ahead_lots = queue(generator),
          .ask_queue_ahead_lots = queue(generator),
          .bid_adverse_selection_ticks = adverse(generator),
          .ask_adverse_selection_ticks = adverse(generator),
          .midprice_return_ticks = mid_return(generator),
      });
    }
  }
  return paths;
}

SurrogateOptimizationResult optimize_differentiable_surrogate(
    SurrogateParameters initial_parameters,
    const std::vector<SurrogatePath>& paths,
    const DifferentiableSurrogateConfig& surrogate_config,
    const SurrogateOptimizerConfig& optimizer_config) {
  validate_input(initial_parameters, paths, surrogate_config);
  if (optimizer_config.iterations == 0 ||
      !std::isfinite(optimizer_config.learning_rate) ||
      optimizer_config.learning_rate <= 0.0 ||
      !std::isfinite(optimizer_config.minimum_learning_rate) ||
      optimizer_config.minimum_learning_rate < 0.0 ||
      optimizer_config.minimum_learning_rate > optimizer_config.learning_rate ||
      !std::isfinite(optimizer_config.gradient_clip_norm) ||
      optimizer_config.gradient_clip_norm <= 0.0 ||
      !std::isfinite(optimizer_config.adam_beta1) ||
      optimizer_config.adam_beta1 < 0.0 || optimizer_config.adam_beta1 >= 1.0 ||
      !std::isfinite(optimizer_config.adam_beta2) ||
      optimizer_config.adam_beta2 < 0.0 || optimizer_config.adam_beta2 >= 1.0 ||
      !std::isfinite(optimizer_config.adam_epsilon) ||
      optimizer_config.adam_epsilon <= 0.0) {
    throw std::invalid_argument{"invalid surrogate optimizer configuration"};
  }
  for (double& value : initial_parameters.values) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument{"initial surrogate parameter is not finite"};
    }
    value = std::clamp(value, -surrogate_config.parameter_absolute_limit,
                       surrogate_config.parameter_absolute_limit);
  }
  SurrogateOptimizationResult result;
  result.parameters = initial_parameters;
  result.seed = optimizer_config.seed;
  auto evaluation_config = surrogate_config;
  if (optimizer_config.worker_count != 0) {
    evaluation_config.evaluation_worker_count =
        optimizer_config.worker_count;
  }
  result.initial_reward_currency =
      evaluate_differentiable_surrogate(result.parameters, paths,
                                        evaluation_config)
          .expected_reward_currency;
  result.trace.reserve(optimizer_config.iterations);
  Derivatives first_moment{};
  Derivatives second_moment{};

  for (std::size_t iteration = 0; iteration < optimizer_config.iterations;
       ++iteration) {
    const auto evaluation = differentiable_surrogate_gradient(
        result.parameters, paths, evaluation_config);
    auto gradient = evaluation.gradient;
    const double gradient_norm = norm(gradient);
    if (!std::isfinite(gradient_norm)) {
      throw std::overflow_error{"surrogate gradient norm is not finite"};
    }
    if (gradient_norm > optimizer_config.gradient_clip_norm) {
      const double scale = optimizer_config.gradient_clip_norm / gradient_norm;
      for (double& component : gradient) component *= scale;
    }
    const double learning_rate =
        scheduled_rate(optimizer_config, iteration);
    result.trace.push_back({
        .iteration = iteration,
        .expected_reward_currency = evaluation.expected_reward_currency,
        .gradient_norm = gradient_norm,
        .learning_rate = learning_rate,
    });

    for (std::size_t i = 0; i < kSurrogateParameterCount; ++i) {
      double update = gradient[i];
      if (optimizer_config.optimizer == SurrogateOptimizer::Adam) {
        first_moment[i] =
            optimizer_config.adam_beta1 * first_moment[i] +
            (1.0 - optimizer_config.adam_beta1) * gradient[i];
        second_moment[i] =
            optimizer_config.adam_beta2 * second_moment[i] +
            (1.0 - optimizer_config.adam_beta2) * gradient[i] * gradient[i];
        const double step = static_cast<double>(iteration + 1);
        const double corrected_first =
            first_moment[i] /
            (1.0 - std::pow(optimizer_config.adam_beta1, step));
        const double corrected_second =
            second_moment[i] /
            (1.0 - std::pow(optimizer_config.adam_beta2, step));
        update = corrected_first /
                 (std::sqrt(corrected_second) + optimizer_config.adam_epsilon);
      }
      result.parameters.values[i] = std::clamp(
          result.parameters.values[i] + learning_rate * update,
          -surrogate_config.parameter_absolute_limit,
          surrogate_config.parameter_absolute_limit);
    }
  }
  result.final_reward_currency =
      evaluate_differentiable_surrogate(result.parameters, paths,
                                        evaluation_config)
          .expected_reward_currency;
  return result;
}

}  // namespace project3
