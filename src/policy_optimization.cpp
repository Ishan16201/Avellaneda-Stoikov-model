#include "project3/policy_optimization.hpp"

#include "project3/detail/parallel.hpp"
#include "project3/exchange_adapter.hpp"
#include "project3/market_agent.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace project3 {
namespace {

constexpr std::size_t kRetainAction = 0;
constexpr std::size_t kCancelAction = 1;
constexpr std::size_t kFirstQuoteAction = 2;
constexpr std::size_t kQuoteSizeCount = 2;
constexpr double kNanosecondsPerSecond = 1.0e9;
constexpr std::uint64_t kPolicySeedSalt = 0x9e3779b97f4a7c15ULL;

using Features = std::array<double, kPolicyFeatureCount>;
using PolicyGradient = std::array<double, kPolicyGradientParameterCount>;
using ActionProbabilities = std::array<double, kPolicyActionCount>;

struct EpisodeResult final {
  double pnl{};
  double reward{};
  double drawdown{};
  double fill_rate{};
  double max_inventory{};
  PolicyGradient score_gradient{};
};

struct MarketObservationState final {
  bool initialized{};
  double initial_midprice{};
  double previous_midprice{};
  lob::core::TimestampNs previous_timestamp_ns{};
  double variance_rate{};
  bool has_previous_midprice{};

  void update(const MarketView& view) {
    if (view.midprice_ticks.has_value()) {
      const double midprice = *view.midprice_ticks;
      if (!initialized) {
        initialized = true;
        initial_midprice = midprice;
      }
      if (has_previous_midprice &&
          view.timestamp_ns > previous_timestamp_ns) {
        const double seconds =
            static_cast<double>(view.timestamp_ns - previous_timestamp_ns) /
            kNanosecondsPerSecond;
        const double normalized_return =
            (midprice - previous_midprice) / std::sqrt(seconds);
        variance_rate =
            0.95 * variance_rate + 0.05 * normalized_return * normalized_return;
      }
      previous_midprice = midprice;
      previous_timestamp_ns = view.timestamp_ns;
      has_previous_midprice = true;
    }
  }

  [[nodiscard]] double volatility() const noexcept {
    return std::sqrt(std::max(0.0, variance_rate));
  }
};

double displayed_depth(const std::vector<PriceLevel>& levels) {
  double result = 0.0;
  for (const auto& level : levels) {
    result += static_cast<double>(level.quantity);
  }
  return result;
}

double estimated_queue_ahead(const MarketView& view,
                             const lob::core::Side side) {
  double result = 0.0;
  bool found = false;
  for (const auto& queue : view.queue_positions) {
    if (queue.side == side) {
      result += static_cast<double>(queue.quantity_ahead);
      found = true;
    }
  }
  if (found) return result;
  const auto& levels =
      side == lob::core::Side::Buy ? view.bids : view.asks;
  return levels.empty() ? 0.0 : static_cast<double>(levels.front().quantity);
}

bool has_open_order(const MarketView& view, const lob::core::Side side) {
  return std::any_of(view.open_orders.begin(), view.open_orders.end(),
                     [side](const auto& order) { return order.side == side; });
}

Features policy_features(const MarketView& view,
                         const MarketObservationState& state,
                         const double elapsed_seconds,
                         const double horizon_seconds,
                         const PolicyGradientConfig& config) {
  Features features{};
  const double limit = config.inventory_limit_lots;
  const double reference_mid =
      std::max(1.0, std::abs(state.initialized ? state.initial_midprice : 1.0));
  features[0] = 1.0;
  features[1] = std::clamp(static_cast<double>(view.inventory_lots) / limit,
                           -2.0, 2.0);
  features[2] =
      view.midprice_ticks.has_value()
          ? std::clamp((*view.midprice_ticks -
                        (state.initialized ? state.initial_midprice : 0.0)) /
                           reference_mid,
                       -2.0, 2.0)
          : 0.0;
  features[3] = std::clamp(
      static_cast<double>(view.spread_ticks.value_or(1)) / 10.0, 0.0, 5.0);
  features[4] =
      std::log1p(displayed_depth(view.bids)) / config.depth_feature_scale_lots;
  features[5] =
      std::log1p(displayed_depth(view.asks)) / config.depth_feature_scale_lots;
  for (std::size_t i = 0; i < lob::stochastic::kEventTypeCount; ++i) {
    const double intensity =
        i < view.hawkes_intensities.size() ? view.hawkes_intensities[i] : 0.0;
    features[6 + i] =
        std::log1p(intensity) / config.intensity_feature_scale;
  }
  features[12] = state.volatility() / config.volatility_feature_scale;
  features[13] =
      std::log1p(estimated_queue_ahead(view, lob::core::Side::Buy)) /
      config.queue_feature_scale_lots;
  features[14] =
      std::log1p(estimated_queue_ahead(view, lob::core::Side::Sell)) /
      config.queue_feature_scale_lots;
  features[15] = std::clamp(
      (horizon_seconds - elapsed_seconds) / horizon_seconds, 0.0, 1.0);
  features[16] =
      has_open_order(view, lob::core::Side::Buy) ? 1.0 : 0.0;
  features[17] =
      has_open_order(view, lob::core::Side::Sell) ? 1.0 : 0.0;
  for (std::size_t i = 1; i < features.size(); ++i) {
    features[i] = std::clamp(features[i], -5.0, 5.0);
  }
  return features;
}

std::size_t parameter_index(const std::size_t side,
                            const std::size_t action,
                            const std::size_t feature) {
  return (side * kPolicyActionCount + action) * kPolicyFeatureCount + feature;
}

ActionProbabilities action_distribution(
    const PolicyGradientParameters& parameters,
    const std::size_t side,
    const Features& features,
    const bool order_is_open,
    const bool side_has_inventory_capacity,
    const PolicyGradientConfig& config) {
  ActionProbabilities logits{};
  std::array<bool, kPolicyActionCount> valid{};
  valid[kRetainAction] = true;
  if (order_is_open) {
    valid[kCancelAction] = true;
  } else if (side_has_inventory_capacity) {
    for (std::size_t action = kFirstQuoteAction;
         action < kPolicyActionCount; ++action) {
      const std::size_t size_index =
          (action - kFirstQuoteAction) % kQuoteSizeCount;
      valid[action] =
          config.quote_sizes_lots[size_index] > 0;
    }
  }

  double max_logit = -std::numeric_limits<double>::infinity();
  for (std::size_t action = 0; action < kPolicyActionCount; ++action) {
    if (!valid[action]) continue;
    double logit = 0.0;
    for (std::size_t feature = 0; feature < kPolicyFeatureCount; ++feature) {
      logit += parameters.values[parameter_index(side, action, feature)] *
               features[feature];
    }
    logits[action] = logit;
    max_logit = std::max(max_logit, logit);
  }
  double normalizer = 0.0;
  for (std::size_t action = 0; action < kPolicyActionCount; ++action) {
    if (valid[action]) {
      logits[action] = std::exp(logits[action] - max_logit);
      normalizer += logits[action];
    }
  }
  if (!std::isfinite(normalizer) || normalizer <= 0.0) {
    throw std::overflow_error{"policy action softmax normalization failed"};
  }
  for (std::size_t action = 0; action < kPolicyActionCount; ++action) {
    logits[action] = valid[action] ? logits[action] / normalizer : 0.0;
  }
  return logits;
}

std::size_t sample_action(const ActionProbabilities& probabilities,
                          std::mt19937_64& random) {
  std::uniform_real_distribution<double> uniform{0.0, 1.0};
  const double draw = uniform(random);
  double cumulative = 0.0;
  for (std::size_t action = 0; action < probabilities.size(); ++action) {
    cumulative += probabilities[action];
    if (draw < cumulative || action + 1 == probabilities.size()) {
      if (probabilities[action] > 0.0) return action;
    }
  }
  throw std::logic_error{"sampled policy action has zero probability"};
}

void accumulate_score_gradient(PolicyGradient& score,
                               const std::size_t side,
                               const std::size_t chosen_action,
                               const ActionProbabilities& probabilities,
                               const Features& features) {
  for (std::size_t action = 0; action < kPolicyActionCount; ++action) {
    const double residual =
        (action == chosen_action ? 1.0 : 0.0) - probabilities[action];
    if (residual == 0.0) continue;
    for (std::size_t feature = 0; feature < kPolicyFeatureCount; ++feature) {
      score[parameter_index(side, action, feature)] +=
          residual * features[feature];
    }
  }
}

double scheduled_learning_rate(const PolicyGradientConfig& config,
                               const std::size_t epoch) {
  if (config.epochs <= 1 ||
      config.learning_rate_schedule == LearningRateSchedule::Constant) {
    return config.learning_rate;
  }
  const double fraction =
      static_cast<double>(epoch) / static_cast<double>(config.epochs - 1);
  if (config.learning_rate_schedule == LearningRateSchedule::LinearDecay) {
    return config.learning_rate +
           (config.minimum_learning_rate - config.learning_rate) * fraction;
  }
  const double weight =
      0.5 * (1.0 + std::cos(std::acos(-1.0) * fraction));
  return config.minimum_learning_rate +
         (config.learning_rate - config.minimum_learning_rate) * weight;
}

double scheduled_learning_rate(const SurrogateTrainingConfig& config,
                               const std::size_t epoch) {
  if (config.epochs <= 1 ||
      config.learning_rate_schedule == LearningRateSchedule::Constant) {
    return config.learning_rate;
  }
  const double fraction =
      static_cast<double>(epoch) / static_cast<double>(config.epochs - 1);
  if (config.learning_rate_schedule == LearningRateSchedule::LinearDecay) {
    return config.learning_rate +
           (config.minimum_learning_rate - config.learning_rate) * fraction;
  }
  const double weight =
      0.5 * (1.0 + std::cos(std::acos(-1.0) * fraction));
  return config.minimum_learning_rate +
         (config.learning_rate - config.minimum_learning_rate) * weight;
}

double vector_norm(const PolicyGradient& values) {
  double result = 0.0;
  for (const double value : values) result = std::hypot(result, value);
  return result;
}

double vector_norm(const std::array<double, kSurrogateParameterCount>& values) {
  double result = 0.0;
  for (const double value : values) result = std::hypot(result, value);
  return result;
}

std::pair<double, double> mean_and_std(const std::vector<double>& values) {
  if (values.empty()) {
    throw std::invalid_argument{"cannot summarize an empty reward sample"};
  }
  const double mean =
      std::accumulate(values.begin(), values.end(), 0.0) /
      static_cast<double>(values.size());
  double squared = 0.0;
  for (const double value : values) squared += (value - mean) * (value - mean);
  return {mean,
          std::sqrt(squared / static_cast<double>(values.size()))};
}

void validate_scenarios(
    const std::vector<PolicyOptimizationScenario>& scenarios,
    const char* name) {
  if (scenarios.empty()) {
    throw std::invalid_argument{std::string{name} + " scenarios are empty"};
  }
  for (const auto& scenario : scenarios) {
    if (scenario.seeds.empty()) {
      throw std::invalid_argument{
          std::string{name} + " scenario has no random seeds"};
    }
  }
}

void ensure_seed_partitions_disjoint(
    const std::vector<PolicyOptimizationScenario>& training,
    const std::vector<PolicyOptimizationScenario>& validation,
    const std::vector<PolicyOptimizationScenario>* out_of_sample = nullptr) {
  std::vector<std::uint64_t> training_seeds;
  std::vector<std::uint64_t> validation_seeds;
  std::vector<std::uint64_t> evaluation_seeds;
  for (const auto& scenario : training) {
    training_seeds.insert(training_seeds.end(), scenario.seeds.begin(),
                          scenario.seeds.end());
  }
  for (const auto& scenario : validation) {
    validation_seeds.insert(validation_seeds.end(), scenario.seeds.begin(),
                            scenario.seeds.end());
  }
  if (out_of_sample != nullptr) {
    for (const auto& scenario : *out_of_sample) {
      evaluation_seeds.insert(evaluation_seeds.end(), scenario.seeds.begin(),
                              scenario.seeds.end());
    }
  }
  const auto overlaps = [](const auto& left, const auto& right) {
    for (const auto seed : left) {
      if (std::find(right.begin(), right.end(), seed) != right.end()) return true;
    }
    return false;
  };
  if (overlaps(training_seeds, validation_seeds) ||
      overlaps(training_seeds, evaluation_seeds) ||
      overlaps(validation_seeds, evaluation_seeds)) {
    throw std::invalid_argument{
        "training, validation, and out-of-sample seeds must be disjoint"};
  }
}

void validate_policy_gradient_config(const PolicyGradientConfig& config) {
  const auto positive_finite = [](const double value) {
    return std::isfinite(value) && value > 0.0;
  };
  if (config.epochs == 0 || config.episodes_per_epoch == 0 ||
      !positive_finite(config.learning_rate) ||
      !std::isfinite(config.minimum_learning_rate) ||
      config.minimum_learning_rate < 0.0 ||
      config.minimum_learning_rate > config.learning_rate ||
      !positive_finite(config.gradient_clip_norm) ||
      !positive_finite(config.reward_normalization_scale_currency) ||
      !std::isfinite(config.baseline_decay) || config.baseline_decay < 0.0 ||
      config.baseline_decay >= 1.0 ||
      !std::isfinite(config.adam_beta1) || config.adam_beta1 < 0.0 ||
      config.adam_beta1 >= 1.0 || !std::isfinite(config.adam_beta2) ||
      config.adam_beta2 < 0.0 || config.adam_beta2 >= 1.0 ||
      !positive_finite(config.adam_epsilon) ||
      !positive_finite(config.inventory_limit_lots) ||
      !std::isfinite(
          config.running_inventory_penalty_currency_per_second_lot_squared) ||
      config.running_inventory_penalty_currency_per_second_lot_squared < 0.0 ||
      !std::isfinite(config.terminal_inventory_penalty_currency_per_lot_squared) ||
      config.terminal_inventory_penalty_currency_per_lot_squared < 0.0 ||
      !positive_finite(config.volatility_feature_scale) ||
      !positive_finite(config.queue_feature_scale_lots) ||
      !positive_finite(config.depth_feature_scale_lots) ||
      !positive_finite(config.intensity_feature_scale) ||
      !positive_finite(config.decision_interval_seconds) ||
      config.quote_sizes_lots[0] <= 0 ||
      config.quote_sizes_lots[1] < config.quote_sizes_lots[0] ||
      config.quote_sizes_lots[1] >
          config.evaluation.broker_config.risk.max_order_lots) {
    throw std::invalid_argument{"invalid simulator policy-gradient configuration"};
  }
  if (config.evaluation.timer_interval_ns <= 0) {
    throw std::invalid_argument{"policy timer interval must be positive"};
  }
  for (std::size_t i = 0; i < config.quote_offsets_ticks.size(); ++i) {
    if (!std::isfinite(config.quote_offsets_ticks[i]) ||
        config.quote_offsets_ticks[i] <= 0.0 ||
        (i > 0 && config.quote_offsets_ticks[i] <=
                      config.quote_offsets_ticks[i - 1])) {
      throw std::invalid_argument{
          "policy quote offsets must be positive and strictly increasing"};
    }
  }
}

void validate_surrogate_training_config(const SurrogateTrainingConfig& config) {
  if (config.epochs == 0 || !std::isfinite(config.learning_rate) ||
      config.learning_rate <= 0.0 ||
      !std::isfinite(config.minimum_learning_rate) ||
      config.minimum_learning_rate < 0.0 ||
      config.minimum_learning_rate > config.learning_rate ||
      !std::isfinite(config.gradient_clip_norm) ||
      config.gradient_clip_norm <= 0.0 ||
      !std::isfinite(config.reward_normalization_scale_currency) ||
      config.reward_normalization_scale_currency <= 0.0 ||
      !std::isfinite(config.adam_beta1) || config.adam_beta1 < 0.0 ||
      config.adam_beta1 >= 1.0 || !std::isfinite(config.adam_beta2) ||
      config.adam_beta2 < 0.0 || config.adam_beta2 >= 1.0 ||
      !std::isfinite(config.adam_epsilon) || config.adam_epsilon <= 0.0) {
    throw std::invalid_argument{"invalid surrogate policy-training configuration"};
  }
}

void write_checkpoint(const std::string& path,
                      const std::string& algorithm,
                      const std::uint64_t seed,
                      const std::size_t epoch,
                      const double validation_reward,
                      const std::vector<double>& parameters,
                      const std::vector<double>& optimizer_state,
                      const std::vector<double>& hyperparameters) {
  if (path.empty()) return;
  const std::filesystem::path destination{path};
  if (!destination.parent_path().empty()) {
    std::filesystem::create_directories(destination.parent_path());
  }
  std::ofstream output{destination, std::ios::trunc};
  if (!output) {
    throw std::runtime_error{"failed to open policy checkpoint: " + path};
  }
  output.precision(17);
  output << "PROJECT3_POLICY_CHECKPOINT 1\n"
         << algorithm << '\n'
         << seed << ' ' << epoch << ' ' << validation_reward << '\n'
         << "hyperparameters " << hyperparameters.size() << '\n';
  for (const double value : hyperparameters) output << value << ' ';
  output << "\nparameters " << parameters.size() << '\n';
  for (const double value : parameters) output << value << ' ';
  output << "\noptimizer_state " << optimizer_state.size() << '\n';
  for (const double value : optimizer_state) output << value << ' ';
  output << '\n';
  output.flush();
  if (!output) {
    throw std::runtime_error{"failed to write policy checkpoint: " + path};
  }
}

void write_learning_curve(
    const std::string& path,
    const std::vector<PolicyLearningCurvePoint>& curve) {
  if (path.empty()) return;
  const std::filesystem::path destination{path};
  if (!destination.parent_path().empty()) {
    std::filesystem::create_directories(destination.parent_path());
  }
  std::ofstream output{destination, std::ios::trunc};
  if (!output) {
    throw std::runtime_error{"failed to open policy learning curve: " + path};
  }
  output << "epoch,training_mean,training_std,validation_mean,validation_std,"
            "normalized_advantage_mean,gradient_norm,learning_rate\n";
  output.precision(17);
  for (const auto& point : curve) {
    output << point.epoch << ',' << point.training_reward_mean_currency << ','
           << point.training_reward_std_currency << ','
           << point.validation_reward_mean_currency << ','
           << point.validation_reward_std_currency << ','
           << point.normalized_advantage_mean << ',' << point.gradient_norm
           << ',' << point.learning_rate << '\n';
  }
  if (!output) {
    throw std::runtime_error{"failed to write policy learning curve: " + path};
  }
}

void write_learning_curve(
    const std::string& path,
    const std::vector<SurrogateLearningCurvePoint>& curve) {
  if (path.empty()) return;
  const std::filesystem::path destination{path};
  if (!destination.parent_path().empty()) {
    std::filesystem::create_directories(destination.parent_path());
  }
  std::ofstream output{destination, std::ios::trunc};
  if (!output) {
    throw std::runtime_error{"failed to open surrogate learning curve: " + path};
  }
  output << "epoch,training_mean,validation_mean,gradient_norm,learning_rate\n";
  output.precision(17);
  for (const auto& point : curve) {
    output << point.epoch << ',' << point.training_reward_mean_currency << ','
           << point.validation_reward_mean_currency << ','
           << point.gradient_norm << ',' << point.learning_rate << '\n';
  }
  if (!output) {
    throw std::runtime_error{"failed to write surrogate learning curve: " + path};
  }
}

class CategoricalPolicyAgent final : public TradingAgent {
 public:
  CategoricalPolicyAgent(PolicyGradientParameters parameters,
                         PolicyGradientConfig config,
                         const double horizon_seconds,
                         const std::uint64_t seed)
      : parameters_(std::move(parameters)),
        config_(std::move(config)),
        horizon_seconds_(horizon_seconds),
        random_(seed) {}

  [[nodiscard]] std::string name() const override {
    return "SimulatorPolicyGradient";
  }
  void on_start(const MarketView& view, OrderGateway& gateway) override {
    observe(view);
    act_if_due(view, gateway, true);
  }
  void on_market_event(const MarketView& view, OrderGateway& gateway) override {
    observe(view);
    act_if_due(view, gateway, false);
  }
  void on_timer(const MarketView& view, OrderGateway& gateway) override {
    observe(view);
    act_if_due(view, gateway, false);
  }
  void on_trade(const lob::strategy::StrategyMarketTrade&,
                const MarketView& view) override {
    observe(view);
  }
  void on_execution(const ExecutionNotification&,
                    const MarketView& view,
                    OrderGateway&) override {
    observe(view);
  }
  void on_stop(const MarketView& view, OrderGateway&) override {
    observe(view);
    const double net_pnl =
        view.market_maker.account.net_total_pnl_currency();
    const double terminal_inventory =
        static_cast<double>(view.inventory_lots);
    objective_reward_ =
        net_pnl -
        config_.running_inventory_penalty_currency_per_second_lot_squared *
            running_inventory_square_seconds_ -
        config_.terminal_inventory_penalty_currency_per_lot_squared *
            terminal_inventory * terminal_inventory;
  }

  [[nodiscard]] const PolicyGradient& score_gradient() const noexcept {
    return score_gradient_;
  }
  [[nodiscard]] double objective_reward() const noexcept {
    return objective_reward_;
  }

 private:
  void observe(const MarketView& view) {
    const double timestamp =
        static_cast<double>(view.timestamp_ns) / kNanosecondsPerSecond;
    if (has_last_observation_ && timestamp > last_observation_seconds_) {
      const double elapsed = timestamp - last_observation_seconds_;
      const double inventory = static_cast<double>(last_inventory_lots_);
      running_inventory_square_seconds_ += elapsed * inventory * inventory;
    }
    if (view.timestamp_ns < last_timestamp_ns_) {
      throw std::logic_error{"market observation time moved backwards"};
    }
    last_observation_seconds_ = timestamp;
    last_timestamp_ns_ = view.timestamp_ns;
    last_inventory_lots_ = view.inventory_lots;
    has_last_observation_ = true;
    market_state_.update(view);
  }

  void act_if_due(const MarketView& view,
                  OrderGateway& gateway,
                  const bool force) {
    if (!view.midprice_ticks.has_value()) return;
    if (!force && view.timestamp_ns - last_decision_ns_ <
                      static_cast<lob::core::TimestampNs>(
                          config_.decision_interval_seconds *
                          kNanosecondsPerSecond)) {
      return;
    }
    last_decision_ns_ = view.timestamp_ns;
    const double elapsed =
        static_cast<double>(view.timestamp_ns) / kNanosecondsPerSecond;
    const auto features = policy_features(
        view, market_state_, elapsed, horizon_seconds_, config_);
    act_side(view, gateway, lob::core::Side::Buy, 0, features);
    act_side(view, gateway, lob::core::Side::Sell, 1, features);
  }

  void act_side(const MarketView& view,
                OrderGateway& gateway,
                const lob::core::Side side_value,
                const std::size_t side_index,
                const Features& features) {
    const bool open = has_open_order(view, side_value);
    const bool has_capacity =
        side_value == lob::core::Side::Buy
            ? static_cast<double>(view.inventory_lots) <
                  config_.inventory_limit_lots
            : static_cast<double>(view.inventory_lots) >
                  -config_.inventory_limit_lots;
    const auto probabilities = action_distribution(
        parameters_, side_index, features, open, has_capacity, config_);
    const auto action = sample_action(probabilities, random_);
    accumulate_score_gradient(score_gradient_, side_index, action,
                              probabilities, features);
    if (action == kRetainAction) return;
    if (action == kCancelAction) {
      for (const auto& order : view.open_orders) {
        if (order.side == side_value) gateway.cancel(order.client_order_id);
      }
      return;
    }

    const std::size_t quote_index = action - kFirstQuoteAction;
    const std::size_t offset_index = quote_index / kQuoteSizeCount;
    const std::size_t size_index = quote_index % kQuoteSizeCount;
    const auto capacity = side_value == lob::core::Side::Buy
                              ? config_.inventory_limit_lots -
                                    static_cast<double>(view.inventory_lots)
                              : config_.inventory_limit_lots +
                                    static_cast<double>(view.inventory_lots);
    const auto quantity = static_cast<lob::core::QuantityLots>(
        std::min<double>(static_cast<double>(config_.quote_sizes_lots[size_index]),
                         static_cast<double>(capacity)));
    if (quantity <= 0) return;
    const double mid = *view.midprice_ticks;
    const double offset = config_.quote_offsets_ticks[offset_index];
    double quote_price =
        side_value == lob::core::Side::Buy ? std::floor(mid - offset)
                                           : std::ceil(mid + offset);
    if (side_value == lob::core::Side::Buy && view.best_ask.has_value()) {
      quote_price = std::min(
          quote_price, static_cast<double>(*view.best_ask - 1));
    }
    if (side_value == lob::core::Side::Sell && view.best_bid.has_value()) {
      quote_price = std::max(
          quote_price, static_cast<double>(*view.best_bid + 1));
    }
    if (!std::isfinite(quote_price) ||
        quote_price < static_cast<double>(
                          std::numeric_limits<lob::core::PriceTicks>::min()) ||
        quote_price > static_cast<double>(
                          std::numeric_limits<lob::core::PriceTicks>::max())) {
      throw std::overflow_error{"policy-generated quote is out of price range"};
    }
    static_cast<void>(gateway.submit_limit(
        side_value, static_cast<lob::core::PriceTicks>(quote_price), quantity));
  }

  PolicyGradientParameters parameters_;
  PolicyGradientConfig config_;
  double horizon_seconds_{};
  std::mt19937_64 random_;
  MarketObservationState market_state_;
  PolicyGradient score_gradient_{};
  lob::core::TimestampNs last_timestamp_ns_{};
  lob::core::TimestampNs last_decision_ns_{};
  double last_observation_seconds_{};
  double running_inventory_square_seconds_{};
  std::int64_t last_inventory_lots_{};
  double objective_reward_{};
  bool has_last_observation_{};
};

class SurrogateMarketCollector final : public TradingAgent {
 public:
  explicit SurrogateMarketCollector(const double horizon_seconds)
      : horizon_seconds_(horizon_seconds) {}

  [[nodiscard]] std::string name() const override {
    return "SurrogateMarketCollector";
  }
  void on_start(const MarketView& view, OrderGateway&) override {
    state_.update(view);
    previous_timestamp_ns_ = view.timestamp_ns;
    previous_midprice_ = view.midprice_ticks.value_or(0.0);
    path_.initial_inventory_lots = static_cast<double>(view.inventory_lots);
    initialized_ = true;
  }
  void on_market_event(const MarketView& view, OrderGateway&) override {
    collect(view);
  }
  void on_timer(const MarketView& view, OrderGateway&) override {
    collect(view);
  }
  void on_trade(const lob::strategy::StrategyMarketTrade&,
                const MarketView& view) override {
    collect(view);
  }
  void on_execution(const ExecutionNotification&,
                    const MarketView& view,
                    OrderGateway&) override {
    collect(view);
  }
  void on_stop(const MarketView& view, OrderGateway&) override {
    collect(view);
    if (path_.steps.empty()) {
      append(view, 0.001, 0.0);
    }
  }
  [[nodiscard]] const SurrogatePath& path() const noexcept { return path_; }

 private:
  void collect(const MarketView& view) {
    if (!initialized_) {
      state_.update(view);
      previous_timestamp_ns_ = view.timestamp_ns;
      previous_midprice_ = view.midprice_ticks.value_or(0.0);
      path_.initial_inventory_lots = static_cast<double>(view.inventory_lots);
      initialized_ = true;
      return;
    }
    if (view.timestamp_ns <= previous_timestamp_ns_) return;
    const double seconds =
        static_cast<double>(view.timestamp_ns - previous_timestamp_ns_) /
        kNanosecondsPerSecond;
    state_.update(view);
    const double mid = view.midprice_ticks.value_or(previous_midprice_);
    append(view, seconds, mid - previous_midprice_);
    previous_timestamp_ns_ = view.timestamp_ns;
    previous_midprice_ = mid;
  }

  void append(const MarketView& view,
              const double duration,
              const double observed_return) {
    SurrogateMarketStep step;
    step.duration_seconds = std::max(duration, 1.0e-6);
    step.market_buy_intensity_per_second = view.hawkes_flow.market_buy;
    step.market_sell_intensity_per_second = view.hawkes_flow.market_sell;
    step.bid_queue_ahead_lots =
        estimated_queue_ahead(view, lob::core::Side::Buy);
    step.ask_queue_ahead_lots =
        estimated_queue_ahead(view, lob::core::Side::Sell);
    step.bid_adverse_selection_ticks = std::max(0.0, -observed_return);
    step.ask_adverse_selection_ticks = std::max(0.0, observed_return);
    step.midprice_return_ticks = observed_return;
    const double reference =
        std::max(1.0, std::abs(state_.initial_midprice));
    step.midprice_feature =
        view.midprice_ticks.has_value()
            ? (*view.midprice_ticks - state_.initial_midprice) / reference
            : 0.0;
    step.spread_ticks =
        static_cast<double>(view.spread_ticks.value_or(1));
    step.bid_depth_lots = displayed_depth(view.bids);
    step.ask_depth_lots = displayed_depth(view.asks);
    for (std::size_t i = 0;
         i < step.hawkes_intensities.size() &&
         i < view.hawkes_intensities.size();
         ++i) {
      step.hawkes_intensities[i] = view.hawkes_intensities[i];
    }
    step.volatility_ticks_per_sqrt_second = state_.volatility();
    const double elapsed =
        static_cast<double>(view.timestamp_ns) / kNanosecondsPerSecond;
    step.time_remaining_fraction =
        std::clamp((horizon_seconds_ - elapsed) / horizon_seconds_, 0.0, 1.0);
    path_.steps.push_back(step);
  }

  double horizon_seconds_{};
  MarketObservationState state_;
  SurrogatePath path_;
  lob::core::TimestampNs previous_timestamp_ns_{};
  double previous_midprice_{};
  bool initialized_{};
};

SurrogateMarketStep make_surrogate_step(
    const MarketView& view,
    const MarketObservationState& state,
    const double observed_return,
    const double duration_seconds,
    const double horizon_seconds) {
  SurrogateMarketStep step;
  step.duration_seconds = std::max(duration_seconds, 1.0e-6);
  step.market_buy_intensity_per_second = view.hawkes_flow.market_buy;
  step.market_sell_intensity_per_second = view.hawkes_flow.market_sell;
  step.bid_queue_ahead_lots =
      estimated_queue_ahead(view, lob::core::Side::Buy);
  step.ask_queue_ahead_lots =
      estimated_queue_ahead(view, lob::core::Side::Sell);
  step.bid_adverse_selection_ticks = std::max(0.0, -observed_return);
  step.ask_adverse_selection_ticks = std::max(0.0, observed_return);
  step.midprice_return_ticks = observed_return;
  const double reference =
      std::max(1.0, std::abs(state.initial_midprice));
  step.midprice_feature =
      view.midprice_ticks.has_value()
          ? (*view.midprice_ticks - state.initial_midprice) / reference
          : 0.0;
  step.spread_ticks = static_cast<double>(view.spread_ticks.value_or(1));
  step.bid_depth_lots = displayed_depth(view.bids);
  step.ask_depth_lots = displayed_depth(view.asks);
  for (std::size_t i = 0;
       i < step.hawkes_intensities.size() &&
       i < view.hawkes_intensities.size();
       ++i) {
    step.hawkes_intensities[i] = view.hawkes_intensities[i];
  }
  step.volatility_ticks_per_sqrt_second = state.volatility();
  const double elapsed =
      static_cast<double>(view.timestamp_ns) / kNanosecondsPerSecond;
  step.time_remaining_fraction =
      std::clamp((horizon_seconds - elapsed) / horizon_seconds, 0.0, 1.0);
  return step;
}

class SurrogatePolicyAgent final : public TradingAgent {
 public:
  SurrogatePolicyAgent(SurrogateParameters parameters,
                       DifferentiableSurrogateConfig config,
                       const double horizon_seconds,
                       const double decision_interval_seconds,
                       const std::uint64_t seed)
      : parameters_(std::move(parameters)),
        config_(std::move(config)),
        horizon_seconds_(horizon_seconds),
        decision_interval_seconds_(decision_interval_seconds),
        random_(seed) {}

  [[nodiscard]] std::string name() const override {
    return "DifferentiableSurrogatePolicy";
  }
  void on_start(const MarketView& view, OrderGateway& gateway) override {
    state_.update(view);
    previous_timestamp_ns_ = view.timestamp_ns;
    previous_midprice_ = view.midprice_ticks.value_or(0.0);
    act(view, gateway, true, 0.0, 1.0e-6);
  }
  void on_market_event(const MarketView& view, OrderGateway& gateway) override {
    act_observed(view, gateway);
  }
  void on_timer(const MarketView& view, OrderGateway& gateway) override {
    act_observed(view, gateway);
  }
  void on_trade(const lob::strategy::StrategyMarketTrade&,
                const MarketView& view) override {
    update_observation(view);
  }
  void on_execution(const ExecutionNotification&,
                    const MarketView& view,
                    OrderGateway&) override {
    update_observation(view);
  }

 private:
  void update_observation(const MarketView& view) {
    if (view.timestamp_ns <= previous_timestamp_ns_) return;
    const double seconds =
        static_cast<double>(view.timestamp_ns - previous_timestamp_ns_) /
        kNanosecondsPerSecond;
    const double mid = view.midprice_ticks.value_or(previous_midprice_);
    observed_return_ = mid - previous_midprice_;
    last_duration_seconds_ = seconds;
    state_.update(view);
    previous_timestamp_ns_ = view.timestamp_ns;
    previous_midprice_ = mid;
  }

  void act_observed(const MarketView& view, OrderGateway& gateway) {
    update_observation(view);
    if (view.timestamp_ns - last_decision_ns_ <
        static_cast<lob::core::TimestampNs>(decision_interval_seconds_ *
                                            kNanosecondsPerSecond)) {
      return;
    }
    act(view, gateway, false, observed_return_, last_duration_seconds_);
  }

  void act(const MarketView& view,
           OrderGateway& gateway,
           const bool force,
           const double observed_return,
           const double duration_seconds) {
    if (!view.midprice_ticks.has_value()) return;
    if (!force && view.timestamp_ns == last_decision_ns_) return;
    last_decision_ns_ = view.timestamp_ns;
    const auto observation = make_surrogate_step(
        view, state_, observed_return, duration_seconds, horizon_seconds_);
    const auto decision = smooth_surrogate_decision(
        parameters_, static_cast<double>(view.inventory_lots), observation,
        config_);
    submit_or_manage_side(view, gateway, lob::core::Side::Buy,
                          decision.bid_cancel_probability,
                          decision.suggested_bid_size_lots,
                          decision.bid_offset_ticks);
    submit_or_manage_side(view, gateway, lob::core::Side::Sell,
                          decision.ask_cancel_probability,
                          decision.suggested_ask_size_lots,
                          decision.ask_offset_ticks);
  }

  void submit_or_manage_side(const MarketView& view,
                             OrderGateway& gateway,
                             const lob::core::Side side,
                             const double cancel_probability,
                             const double suggested_size,
                             const double offset) {
    std::vector<lob::strategy::OpenStrategyOrder> current;
    for (const auto& order : view.open_orders) {
      if (order.side == side) current.push_back(order);
    }
    std::uniform_real_distribution<double> uniform{0.0, 1.0};
    if (!current.empty()) {
      if (uniform(random_) < cancel_probability) {
        for (const auto& order : current) gateway.cancel(order.client_order_id);
      }
      return;
    }
    if (uniform(random_) < cancel_probability) return;
    const double inventory = static_cast<double>(view.inventory_lots);
    const double capacity =
        side == lob::core::Side::Buy
            ? config_.inventory_limit_lots - inventory
            : config_.inventory_limit_lots + inventory;
    const auto quantity = static_cast<lob::core::QuantityLots>(
        std::min(std::max(0.0, std::round(suggested_size)),
                 std::floor(capacity)));
    if (quantity <= 0) return;
    double price =
        side == lob::core::Side::Buy
            ? std::floor(*view.midprice_ticks - offset)
            : std::ceil(*view.midprice_ticks + offset);
    if (side == lob::core::Side::Buy && view.best_ask.has_value()) {
      price = std::min(price, static_cast<double>(*view.best_ask - 1));
    }
    if (side == lob::core::Side::Sell && view.best_bid.has_value()) {
      price = std::max(price, static_cast<double>(*view.best_bid + 1));
    }
    if (price < static_cast<double>(
                    std::numeric_limits<lob::core::PriceTicks>::min()) ||
        price > static_cast<double>(
                    std::numeric_limits<lob::core::PriceTicks>::max())) {
      throw std::overflow_error{"surrogate policy quote is out of price range"};
    }
    static_cast<void>(gateway.submit_limit(
        side, static_cast<lob::core::PriceTicks>(price), quantity));
  }

  SurrogateParameters parameters_;
  DifferentiableSurrogateConfig config_;
  double horizon_seconds_{};
  double decision_interval_seconds_{};
  std::mt19937_64 random_;
  MarketObservationState state_;
  lob::core::TimestampNs previous_timestamp_ns_{};
  lob::core::TimestampNs last_decision_ns_{};
  double previous_midprice_{};
  double observed_return_{};
  double last_duration_seconds_{1.0e-6};
};

class DynamicProgrammingAgent final : public TradingAgent {
 public:
  DynamicProgrammingAgent(ReducedStateSolution solution,
                          const lob::core::QuantityLots quote_size)
      : solution_(std::move(solution)), quote_size_(quote_size) {}

  [[nodiscard]] std::string name() const override {
    return "ReducedStateDynamicProgramming";
  }
  void on_start(const MarketView& view, OrderGateway& gateway) override {
    act(view, gateway);
  }
  void on_market_event(const MarketView& view, OrderGateway& gateway) override {
    act(view, gateway);
  }
  void on_timer(const MarketView& view, OrderGateway& gateway) override {
    act(view, gateway);
  }

 private:
  void act(const MarketView& view, OrderGateway& gateway) {
    if (!view.midprice_ticks.has_value()) return;
    const bool bid_open = has_open_order(view, lob::core::Side::Buy);
    const bool ask_open = has_open_order(view, lob::core::Side::Sell);
    if (bid_open && ask_open) return;
    const double elapsed =
        static_cast<double>(view.timestamp_ns) / kNanosecondsPerSecond;
    const auto action = solution_.policy(
        elapsed, static_cast<double>(view.inventory_lots));
    if (!bid_open && action.bid_enabled &&
        view.inventory_lots < solution_.config().inventory_limit_lots) {
      double price = std::floor(*view.midprice_ticks - action.bid_offset_ticks);
      if (view.best_ask.has_value()) {
        price = std::min(price, static_cast<double>(*view.best_ask - 1));
      }
      static_cast<void>(gateway.submit_limit(
          lob::core::Side::Buy, static_cast<lob::core::PriceTicks>(price),
          std::min(quote_size_, static_cast<lob::core::QuantityLots>(
                                    solution_.config().inventory_limit_lots -
                                    view.inventory_lots))));
    }
    if (!ask_open && action.ask_enabled &&
        view.inventory_lots > -solution_.config().inventory_limit_lots) {
      double price = std::ceil(*view.midprice_ticks + action.ask_offset_ticks);
      if (view.best_bid.has_value()) {
        price = std::max(price, static_cast<double>(*view.best_bid + 1));
      }
      static_cast<void>(gateway.submit_limit(
          lob::core::Side::Sell, static_cast<lob::core::PriceTicks>(price),
          std::min(quote_size_, static_cast<lob::core::QuantityLots>(
                                    solution_.config().inventory_limit_lots +
                                    view.inventory_lots))));
    }
  }

  ReducedStateSolution solution_;
  lob::core::QuantityLots quote_size_{};
};

std::uint64_t mix_seed(std::uint64_t value) {
  value += kPolicySeedSalt;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

EpisodeResult run_policy_gradient_episode(
    const PolicyGradientParameters& parameters,
    const PolicyOptimizationScenario& scenario,
    const std::uint64_t market_seed,
    const std::uint64_t action_seed,
    const PolicyGradientConfig& config) {
  auto market = scenario.market;
  market.seed = market_seed;
  CategoricalPolicyAgent agent{parameters, config, market.horizon_seconds,
                               action_seed};
  const auto report =
      ExchangeAdapter{}.run(agent, market, config.evaluation);
  return {
      .pnl = report.final_portfolio.net_total_pnl_currency(),
      .reward = agent.objective_reward(),
      .drawdown = report.market_making.max_drawdown,
      .fill_rate = report.market_making.order_fill_rate,
      .max_inventory = report.market_making.max_inventory_exposure,
      .score_gradient = agent.score_gradient(),
  };
}

std::vector<double> scenario_rewards(
    const PolicyGradientParameters& parameters,
    const std::vector<PolicyOptimizationScenario>& scenarios,
    const PolicyGradientConfig& config) {
  std::vector<double> rewards;
  for (std::size_t scenario_index = 0; scenario_index < scenarios.size();
       ++scenario_index) {
    const auto& scenario = scenarios[scenario_index];
    for (std::size_t seed_index = 0; seed_index < scenario.seeds.size();
         ++seed_index) {
      const auto market_seed = scenario.seeds[seed_index];
      const auto action_seed =
          mix_seed(config.seed ^ market_seed ^
                   static_cast<std::uint64_t>(scenario_index));
      rewards.push_back(run_policy_gradient_episode(
                            parameters, scenario, market_seed, action_seed,
                            config)
                            .reward);
    }
  }
  return rewards;
}

std::vector<SurrogatePath> collect_surrogate_paths(
    const std::vector<PolicyOptimizationScenario>& scenarios,
    const lob::strategy::StrategyEvaluatorConfig& evaluation) {
  std::vector<SurrogatePath> paths;
  for (const auto& scenario : scenarios) {
    for (const auto seed : scenario.seeds) {
      auto market = scenario.market;
      market.seed = seed;
      SurrogateMarketCollector collector{market.horizon_seconds};
      static_cast<void>(ExchangeAdapter{}.run(collector, market, evaluation));
      paths.push_back(collector.path());
    }
  }
  return paths;
}

void validate_surrogate_parameters(
    const SurrogateParameters& parameters,
    const DifferentiableSurrogateConfig& config) {
  for (const double value : parameters.values) {
    if (!std::isfinite(value) ||
        std::abs(value) > config.parameter_absolute_limit) {
      throw std::invalid_argument{
          "initial surrogate parameter is outside configured bounds"};
    }
  }
}

void validate_policy_parameters(const PolicyGradientParameters& parameters) {
  for (const double value : parameters.values) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument{"initial policy-gradient weight is not finite"};
    }
  }
}

PolicyEvaluationSummary evaluate_policy_on_scenarios(
    const std::string& policy_name,
    const PolicyGradientParameters* categorical_parameters,
    const SurrogateParameters* surrogate_parameters,
    const std::vector<PolicyOptimizationScenario>& scenarios,
    const PolicyGradientConfig& policy_config,
    const DifferentiableSurrogateConfig& surrogate_config,
    const AvellanedaStoikovConfig& avellaneda_config,
    const ReducedStateConfig& dynamic_programming_config) {
  std::vector<double> pnls;
  std::vector<double> rewards;
  std::vector<double> drawdowns;
  std::vector<double> fill_rates;
  std::vector<double> max_inventories;
  std::size_t episode_count = 0;
  std::string scenario_label;

  for (const auto& scenario : scenarios) {
    if (scenario_label.empty()) scenario_label = scenario.label;
    for (const auto seed : scenario.seeds) {
      auto market = scenario.market;
      market.seed = seed;
      lob::strategy::StrategyEvaluationReport report;
      double objective = 0.0;
      if (categorical_parameters != nullptr) {
        const auto action_seed = mix_seed(policy_config.seed ^ seed);
        CategoricalPolicyAgent agent{*categorical_parameters, policy_config,
                                     market.horizon_seconds, action_seed};
        report =
            ExchangeAdapter{}.run(agent, market, policy_config.evaluation);
        objective = agent.objective_reward();
      } else if (surrogate_parameters != nullptr) {
        const auto action_seed = mix_seed(policy_config.seed ^ seed);
        SurrogatePolicyAgent agent{
            *surrogate_parameters, surrogate_config, market.horizon_seconds,
            policy_config.decision_interval_seconds, action_seed};
        report =
            ExchangeAdapter{}.run(agent, market, policy_config.evaluation);
        objective = report.final_portfolio.net_total_pnl_currency();
      } else if (policy_name == "AvellanedaStoikov") {
        auto config = avellaneda_config;
        config.horizon_seconds = market.horizon_seconds;
        AvellanedaStoikovAgent agent{config};
        report = ExchangeAdapter{}.run(agent, market, policy_config.evaluation);
        objective = report.final_portfolio.net_total_pnl_currency();
      } else if (policy_name == "ReducedStateDynamicProgramming") {
        auto config = dynamic_programming_config;
        config.horizon_seconds = market.horizon_seconds;
        const auto solution = ReducedStateSolver::solve(config);
        DynamicProgrammingAgent agent{
            solution,
            std::min(policy_config.quote_sizes_lots.front(),
                     policy_config.evaluation.broker_config.risk.max_order_lots)};
        report = ExchangeAdapter{}.run(agent, market, policy_config.evaluation);
        objective = report.final_portfolio.net_total_pnl_currency();
      } else {
        throw std::logic_error{"unknown policy requested for evaluation"};
      }
      pnls.push_back(report.final_portfolio.net_total_pnl_currency());
      rewards.push_back(objective);
      drawdowns.push_back(report.market_making.max_drawdown);
      fill_rates.push_back(report.market_making.order_fill_rate);
      max_inventories.push_back(
          report.market_making.max_inventory_exposure);
      ++episode_count;
    }
  }

  const auto [mean_pnl, std_pnl] = mean_and_std(pnls);
  const auto mean = [](const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
  };
  return {
      .scenario_label = scenario_label,
      .policy_name = policy_name,
      .episodes = episode_count,
      .mean_net_pnl_currency = mean_pnl,
      .std_net_pnl_currency = std_pnl,
      .mean_objective_reward_currency = mean(rewards),
      .mean_max_drawdown_currency = mean(drawdowns),
      .mean_fill_rate = mean(fill_rates),
      .mean_max_inventory_lots = mean(max_inventories),
  };
}

void append_summaries(std::vector<PolicyEvaluationSummary>& destination,
                      const std::string& policy_name,
                      const PolicyGradientParameters* categorical_parameters,
                      const SurrogateParameters* surrogate_parameters,
                      const std::vector<PolicyOptimizationScenario>& scenarios,
                      const PolicyGradientConfig& policy_config,
                      const DifferentiableSurrogateConfig& surrogate_config,
                      const AvellanedaStoikovConfig& avellaneda_config,
                      const ReducedStateConfig& dynamic_programming_config) {
  for (const auto& scenario : scenarios) {
    destination.push_back(evaluate_policy_on_scenarios(
        policy_name, categorical_parameters, surrogate_parameters, {scenario},
        policy_config, surrogate_config, avellaneda_config,
        dynamic_programming_config));
  }
}

}  // namespace

PolicyGradientTrainingResult train_simulator_policy_gradient(
    const std::vector<PolicyOptimizationScenario>& training_scenarios,
    const std::vector<PolicyOptimizationScenario>& validation_scenarios,
    const PolicyGradientConfig& config,
    PolicyGradientParameters initial_parameters) {
  validate_scenarios(training_scenarios, "training");
  validate_scenarios(validation_scenarios, "validation");
  ensure_seed_partitions_disjoint(training_scenarios, validation_scenarios);
  validate_policy_gradient_config(config);
  validate_policy_parameters(initial_parameters);

  PolicyGradientTrainingResult result;
  result.parameters = initial_parameters;
  result.hyperparameters = config;
  result.seed = config.seed;
  result.learning_curve.reserve(config.epochs);
  PolicyGradient best_parameters = initial_parameters.values;
  PolicyGradient first_moment{};
  PolicyGradient second_moment{};
  double baseline = 0.0;
  bool has_baseline = false;
  double best_validation = -std::numeric_limits<double>::infinity();
  std::size_t epochs_without_improvement = 0;

  for (std::size_t epoch = 0; epoch < config.epochs; ++epoch) {
    std::vector<EpisodeResult> episodes(config.episodes_per_epoch);
    std::vector<double> training_rewards(config.episodes_per_epoch);
    detail::parallel_for(
        config.episodes_per_epoch, config.worker_count,
        [&](const std::size_t episode) {
          const std::size_t flat_index =
              epoch * config.episodes_per_epoch + episode;
          const auto& scenario =
              training_scenarios[flat_index % training_scenarios.size()];
          const std::size_t seed_index =
              (flat_index / training_scenarios.size()) % scenario.seeds.size();
          const auto market_seed = scenario.seeds[seed_index];
          const auto action_seed =
              mix_seed(config.seed ^ market_seed ^
                       (static_cast<std::uint64_t>(epoch) << 32U) ^
                       static_cast<std::uint64_t>(episode));
          episodes[episode] = run_policy_gradient_episode(
              result.parameters, scenario, market_seed, action_seed, config);
          training_rewards[episode] = episodes[episode].reward;
        });
    const auto [training_mean, training_std] = mean_and_std(training_rewards);
    PolicyGradient batch_gradient{};
    double normalized_advantage_mean = 0.0;
    for (const auto& episode : episodes) {
      const double advantage =
          (episode.reward - (has_baseline ? baseline : 0.0)) /
          config.reward_normalization_scale_currency;
      normalized_advantage_mean += advantage;
      for (std::size_t i = 0; i < batch_gradient.size(); ++i) {
        batch_gradient[i] +=
            advantage * episode.score_gradient[i] /
            static_cast<double>(episodes.size());
      }
    }
    normalized_advantage_mean /= static_cast<double>(episodes.size());
    const double gradient_norm = vector_norm(batch_gradient);
    if (!std::isfinite(gradient_norm)) {
      throw std::overflow_error{"REINFORCE batch gradient is not finite"};
    }
    if (gradient_norm > config.gradient_clip_norm) {
      const double scale = config.gradient_clip_norm / gradient_norm;
      for (double& value : batch_gradient) value *= scale;
    }

    const double learning_rate = scheduled_learning_rate(config, epoch);
    for (std::size_t i = 0; i < result.parameters.values.size(); ++i) {
      first_moment[i] =
          config.adam_beta1 * first_moment[i] +
          (1.0 - config.adam_beta1) * batch_gradient[i];
      second_moment[i] =
          config.adam_beta2 * second_moment[i] +
          (1.0 - config.adam_beta2) * batch_gradient[i] * batch_gradient[i];
    }

    const auto validation_rewards =
        scenario_rewards(result.parameters, validation_scenarios, config);
    const auto [validation_mean, validation_std] =
        mean_and_std(validation_rewards);
    result.learning_curve.push_back({
        .epoch = epoch,
        .training_reward_mean_currency = training_mean,
        .training_reward_std_currency = training_std,
        .validation_reward_mean_currency = validation_mean,
        .validation_reward_std_currency = validation_std,
        .normalized_advantage_mean = normalized_advantage_mean,
        .gradient_norm = gradient_norm,
        .learning_rate = learning_rate,
    });
    ++result.completed_epochs;

    const double batch_baseline =
        std::accumulate(training_rewards.begin(), training_rewards.end(), 0.0) /
        static_cast<double>(training_rewards.size());
    baseline = has_baseline
                   ? config.baseline_decay * baseline +
                         (1.0 - config.baseline_decay) * batch_baseline
                   : batch_baseline;
    has_baseline = true;

    if (validation_mean > best_validation) {
      best_validation = validation_mean;
      best_parameters = result.parameters.values;
      result.best_epoch = epoch;
      result.best_validation_reward_currency = validation_mean;
      epochs_without_improvement = 0;
      std::vector<double> optimizer_state;
      optimizer_state.reserve(first_moment.size() + second_moment.size() + 1);
      optimizer_state.insert(optimizer_state.end(), first_moment.begin(),
                             first_moment.end());
      optimizer_state.insert(optimizer_state.end(), second_moment.begin(),
                             second_moment.end());
      optimizer_state.push_back(baseline);
      write_checkpoint(
          config.checkpoint_path, "simulator_reinforce", config.seed, epoch,
          validation_mean,
          std::vector<double>(best_parameters.begin(), best_parameters.end()),
          optimizer_state,
          {static_cast<double>(config.epochs),
           static_cast<double>(config.episodes_per_epoch), config.learning_rate,
           config.minimum_learning_rate, config.gradient_clip_norm,
           config.reward_normalization_scale_currency, config.baseline_decay,
           config.inventory_limit_lots, config.decision_interval_seconds});
    } else {
      ++epochs_without_improvement;
    }
    if (config.validation_patience > 0 &&
        epochs_without_improvement >= config.validation_patience) {
      break;
    }
    for (std::size_t i = 0; i < result.parameters.values.size(); ++i) {
      const double step = static_cast<double>(epoch + 1);
      const double corrected_first =
          first_moment[i] / (1.0 - std::pow(config.adam_beta1, step));
      const double corrected_second =
          second_moment[i] / (1.0 - std::pow(config.adam_beta2, step));
      result.parameters.values[i] = std::clamp(
          result.parameters.values[i] +
              learning_rate * corrected_first /
                  (std::sqrt(corrected_second) + config.adam_epsilon),
          -20.0, 20.0);
    }
  }
  result.parameters.values = best_parameters;
  const auto final_training_rewards =
      scenario_rewards(result.parameters, training_scenarios, config);
  result.training_reward_currency =
      mean_and_std(final_training_rewards).first;
  write_learning_curve(config.learning_curve_csv_path, result.learning_curve);
  return result;
}

SurrogateTrainingResult train_differentiable_policy_surrogate(
    const std::vector<PolicyOptimizationScenario>& training_scenarios,
    const std::vector<PolicyOptimizationScenario>& validation_scenarios,
    const SurrogateTrainingConfig& config,
    SurrogateParameters initial_parameters) {
  validate_scenarios(training_scenarios, "surrogate training");
  validate_scenarios(validation_scenarios, "surrogate validation");
  ensure_seed_partitions_disjoint(training_scenarios, validation_scenarios);
  validate_surrogate_training_config(config);
  validate_surrogate_parameters(initial_parameters, config.surrogate);
  const auto training_paths =
      collect_surrogate_paths(training_scenarios, {});
  const auto validation_paths =
      collect_surrogate_paths(validation_scenarios, {});

  SurrogateTrainingResult result;
  result.parameters = initial_parameters;
  result.hyperparameters = config;
  result.seed = config.seed;
  result.learning_curve.reserve(config.epochs);
  auto best_parameters = initial_parameters.values;
  std::array<double, kSurrogateParameterCount> first_moment{};
  std::array<double, kSurrogateParameterCount> second_moment{};
  double best_validation = -std::numeric_limits<double>::infinity();
  std::size_t epochs_without_improvement = 0;

  for (std::size_t epoch = 0; epoch < config.epochs; ++epoch) {
    const auto training_evaluation = evaluate_differentiable_surrogate(
        result.parameters, training_paths, config.surrogate);
    const auto validation_evaluation = evaluate_differentiable_surrogate(
        result.parameters, validation_paths, config.surrogate);
    const auto objective_gradient = differentiable_surrogate_gradient(
        result.parameters, training_paths, config.surrogate);
    auto gradient = objective_gradient.gradient;
    for (double& value : gradient) {
      value /= config.reward_normalization_scale_currency;
    }
    const double gradient_norm = vector_norm(gradient);
    if (!std::isfinite(gradient_norm)) {
      throw std::overflow_error{"surrogate policy gradient is not finite"};
    }
    if (gradient_norm > config.gradient_clip_norm) {
      const double scale = config.gradient_clip_norm / gradient_norm;
      for (double& value : gradient) value *= scale;
    }
    const double learning_rate = scheduled_learning_rate(config, epoch);
    for (std::size_t i = 0; i < result.parameters.values.size(); ++i) {
      first_moment[i] =
          config.adam_beta1 * first_moment[i] +
          (1.0 - config.adam_beta1) * gradient[i];
      second_moment[i] =
          config.adam_beta2 * second_moment[i] +
          (1.0 - config.adam_beta2) * gradient[i] * gradient[i];
    }
    result.learning_curve.push_back({
        .epoch = epoch,
        .training_reward_mean_currency =
            training_evaluation.expected_reward_currency,
        .validation_reward_mean_currency =
            validation_evaluation.expected_reward_currency,
        .gradient_norm = gradient_norm,
        .learning_rate = learning_rate,
    });

    if (validation_evaluation.expected_reward_currency > best_validation) {
      best_validation = validation_evaluation.expected_reward_currency;
      best_parameters = result.parameters.values;
      result.best_epoch = epoch;
      result.best_validation_reward_currency = best_validation;
      epochs_without_improvement = 0;
      std::vector<double> optimizer_state;
      optimizer_state.reserve(first_moment.size() + second_moment.size());
      optimizer_state.insert(optimizer_state.end(), first_moment.begin(),
                             first_moment.end());
      optimizer_state.insert(optimizer_state.end(), second_moment.begin(),
                             second_moment.end());
      write_checkpoint(
          config.checkpoint_path, "differentiable_surrogate", config.seed,
          epoch, best_validation,
          std::vector<double>(best_parameters.begin(), best_parameters.end()),
          optimizer_state,
          {static_cast<double>(config.epochs), config.learning_rate,
           config.minimum_learning_rate, config.gradient_clip_norm,
           config.reward_normalization_scale_currency,
           config.surrogate.inventory_limit_lots,
           config.surrogate.maximum_executed_quantity_lots,
           config.surrogate.terminal_inventory_penalty_currency_per_lot_squared,
           config.surrogate.running_inventory_penalty_currency_per_second_lot_squared});
    } else {
      ++epochs_without_improvement;
    }
    if (config.validation_patience > 0 &&
        epochs_without_improvement >= config.validation_patience) {
      break;
    }
    for (std::size_t i = 0; i < result.parameters.values.size(); ++i) {
      const double step = static_cast<double>(epoch + 1);
      const double corrected_first =
          first_moment[i] / (1.0 - std::pow(config.adam_beta1, step));
      const double corrected_second =
          second_moment[i] / (1.0 - std::pow(config.adam_beta2, step));
      result.parameters.values[i] = std::clamp(
          result.parameters.values[i] +
              learning_rate * corrected_first /
                  (std::sqrt(corrected_second) + config.adam_epsilon),
          -config.surrogate.parameter_absolute_limit,
          config.surrogate.parameter_absolute_limit);
    }
  }
  result.parameters.values = best_parameters;
  result.training_reward_currency =
      evaluate_differentiable_surrogate(result.parameters, training_paths,
                                         config.surrogate)
          .expected_reward_currency;
  write_learning_curve(config.learning_curve_csv_path, result.learning_curve);
  return result;
}

PolicyOptimizationStudyResult run_policy_optimization_study(
    const PolicyOptimizationStudyConfig& config) {
  validate_scenarios(config.training_scenarios, "training");
  validate_scenarios(config.validation_scenarios, "validation");
  validate_scenarios(config.out_of_sample_scenarios, "out-of-sample");
  ensure_seed_partitions_disjoint(
      config.training_scenarios, config.validation_scenarios,
      &config.out_of_sample_scenarios);
  validate_policy_gradient_config(config.policy_gradient);
  validate_surrogate_training_config(config.surrogate);

  PolicyOptimizationStudyResult result;
  result.policy_gradient = train_simulator_policy_gradient(
      config.training_scenarios, config.validation_scenarios,
      config.policy_gradient);
  result.surrogate = train_differentiable_policy_surrogate(
      config.training_scenarios, config.validation_scenarios,
      config.surrogate);
  result.surrogate.training_exact_market_reward_currency =
      evaluate_policy_on_scenarios(
          "DifferentiableSurrogatePolicy", nullptr,
          &result.surrogate.parameters, config.training_scenarios,
          config.policy_gradient, config.surrogate.surrogate,
          config.avellaneda_stoikov, config.dynamic_programming)
          .mean_objective_reward_currency;
  result.surrogate.validation_exact_market_reward_currency =
      evaluate_policy_on_scenarios(
          "DifferentiableSurrogatePolicy", nullptr,
          &result.surrogate.parameters, config.validation_scenarios,
          config.policy_gradient, config.surrogate.surrogate,
          config.avellaneda_stoikov, config.dynamic_programming)
          .mean_objective_reward_currency;

  append_summaries(
      result.out_of_sample, "DifferentiableSurrogatePolicy", nullptr,
      &result.surrogate.parameters, config.out_of_sample_scenarios,
      config.policy_gradient, config.surrogate.surrogate,
      config.avellaneda_stoikov, config.dynamic_programming);
  append_summaries(
      result.out_of_sample, "SimulatorPolicyGradient",
      &result.policy_gradient.parameters, nullptr,
      config.out_of_sample_scenarios, config.policy_gradient,
      config.surrogate.surrogate, config.avellaneda_stoikov,
      config.dynamic_programming);
  append_summaries(
      result.out_of_sample, "AvellanedaStoikov", nullptr, nullptr,
      config.out_of_sample_scenarios, config.policy_gradient,
      config.surrogate.surrogate, config.avellaneda_stoikov,
      config.dynamic_programming);
  append_summaries(
      result.out_of_sample, "ReducedStateDynamicProgramming", nullptr, nullptr,
      config.out_of_sample_scenarios, config.policy_gradient,
      config.surrogate.surrogate, config.avellaneda_stoikov,
      config.dynamic_programming);
  write_learning_curve(config.policy_gradient.learning_curve_csv_path,
                       result.policy_gradient.learning_curve);
  return result;
}

}  // namespace project3
