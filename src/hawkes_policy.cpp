#include "project3/hawkes_policy.hpp"

#include "lob/stochastic/event_types.hpp"
#include "lob/stochastic/hawkes_market_adapter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace project3 {
namespace {

using Features = std::array<double, 7>;
using Weights = std::array<std::array<double, 8>, 3>;

struct TrainingObservation final {
  Features features{};
  std::size_t label{};
  double price_change_ticks{};
};

struct DirectionScore final {
  double log_loss{};
  double accuracy{};
  double base_rate_log_loss{};
  double base_rate_accuracy{};
  std::size_t observations{};
};

HawkesFlowState flow_from_intensities(const std::vector<double>& intensities) {
  if (intensities.size() != lob::stochastic::kEventTypeCount) {
    return {};
  }
  for (const double intensity : intensities) {
    if (!std::isfinite(intensity) || intensity < 0.0) {
      throw std::invalid_argument{"invalid Hawkes intensity in training data"};
    }
  }
  return {
      .limit_buy = intensities[0],
      .limit_sell = intensities[1],
      .cancel_buy = intensities[2],
      .cancel_sell = intensities[3],
      .market_buy = intensities[4],
      .market_sell = intensities[5],
      .available = true,
  };
}

std::vector<double> post_event_intensities(
    const lob::stochastic::MarketEventRecord& event,
    const lob::stochastic::HawkesParameters& parameters) {
  auto result = event.pre_event_intensities;
  const auto parent = lob::stochastic::event_type_index(event.type);
  if (result.size() != parameters.baseline_events_per_second.size()) {
    throw std::invalid_argument{"training event has an invalid intensity vector"};
  }
  for (std::size_t child = 0; child < result.size(); ++child) {
    result[child] += parameters.alpha_branching[child][parent] *
                     parameters.beta_per_second[child][parent];
  }
  return result;
}

Features make_features(const HawkesFlowState& state) {
  if (!state.available) {
    throw std::invalid_argument{"Hawkes-aware policy requires six intensities"};
  }
  const std::array<double, 6> intensities{
      state.limit_buy, state.limit_sell, state.cancel_buy,
      state.cancel_sell, state.market_buy, state.market_sell};
  Features features{};
  for (std::size_t i = 0; i < intensities.size(); ++i) {
    if (!std::isfinite(intensities[i]) || intensities[i] < 0.0) {
      throw std::invalid_argument{"invalid Hawkes intensity"};
    }
    features[i] = std::log1p(intensities[i]);
  }
  features[6] = state.order_flow_imbalance();
  return features;
}

std::optional<double> midprice(
    const lob::exchange::BookSnapshot& snapshot) {
  if (snapshot.bids.empty() || snapshot.asks.empty()) return std::nullopt;
  return (static_cast<double>(snapshot.bids.front().price_ticks) +
          static_cast<double>(snapshot.asks.front().price_ticks)) *
         0.5;
}

std::optional<std::size_t> future_snapshot_index(
    const lob::stochastic::SyntheticMarketResult& simulation,
    const double target_seconds,
    const std::size_t start_index) {
  const auto first = simulation.book_snapshots.begin() +
                     static_cast<std::ptrdiff_t>(start_index);
  const auto it = std::lower_bound(
      first, simulation.book_snapshots.end(), target_seconds,
      [](const lob::exchange::BookSnapshot& snapshot, const double seconds) {
        return static_cast<double>(snapshot.timestamp_ns) * 1.0e-9 < seconds;
      });
  if (it == simulation.book_snapshots.end()) return std::nullopt;
  return static_cast<std::size_t>(
      std::distance(simulation.book_snapshots.begin(), it));
}

void append_training_observations(
    const lob::stochastic::SyntheticMarketConfig& market,
    const double forecast_horizon_seconds,
    std::vector<TrainingObservation>& observations) {
  auto simulation = lob::stochastic::HawkesMarketAdapter{}.run(market);
  for (std::size_t i = 0; i < simulation.events.size(); ++i) {
    const auto& event = simulation.events[i];
    const double event_seconds =
        static_cast<double>(simulation.book_snapshots[i + 1].timestamp_ns) *
        1.0e-9;
    const auto end_index = future_snapshot_index(
        simulation, event_seconds + forecast_horizon_seconds, i + 1);
    if (!end_index) continue;
    const auto current_mid = midprice(simulation.book_snapshots[i + 1]);
    const auto future_mid = midprice(simulation.book_snapshots[*end_index]);
    if (!current_mid || !future_mid) continue;
    const double change = *future_mid - *current_mid;
    const std::size_t label = change > 0.0 ? 2U : (change < 0.0 ? 0U : 1U);
    observations.push_back({
        .features = make_features(flow_from_intensities(
            post_event_intensities(event, market.hawkes))),
        .label = label,
        .price_change_ticks = change,
    });
  }
}

std::array<double, 3> class_probabilities(
    const Features& raw_features,
    const std::array<double, 7>& means,
    const std::array<double, 7>& scales,
    const Weights& weights) {
  std::array<double, 3> logits{};
  for (std::size_t c = 0; c < logits.size(); ++c) {
    logits[c] = weights[c][7];
    for (std::size_t f = 0; f < raw_features.size(); ++f) {
      logits[c] +=
          weights[c][f] * (raw_features[f] - means[f]) / scales[f];
    }
    if (!std::isfinite(logits[c])) {
      throw std::runtime_error{"Hawkes direction classifier produced a non-finite logit"};
    }
  }
  const double max_logit = *std::max_element(logits.begin(), logits.end());
  double denominator = 0.0;
  for (auto& logit : logits) {
    logit = std::exp(logit - max_logit);
    denominator += logit;
  }
  if (!std::isfinite(denominator) || denominator <= 0.0) {
    throw std::runtime_error{"Hawkes direction classifier produced invalid probabilities"};
  }
  for (auto& probability : logits) probability /= denominator;
  return logits;
}

PolicyRunSummary summarize(
    const std::vector<lob::strategy::StrategyEvaluationReport>& reports) {
  PolicyRunSummary summary;
  summary.runs = reports.size();
  if (reports.empty()) return summary;
  std::vector<double> pnls;
  pnls.reserve(reports.size());
  for (const auto& report : reports) {
    const auto& metrics = report.market_making;
    pnls.push_back(metrics.total_net_pnl);
    summary.mean_fill_rate += metrics.order_fill_rate;
    summary.mean_max_drawdown += metrics.max_drawdown;
    summary.mean_turnover += metrics.gross_turnover;
    summary.mean_max_inventory += metrics.max_inventory_exposure;
  }
  summary.mean_total_pnl =
      std::accumulate(pnls.begin(), pnls.end(), 0.0) /
      static_cast<double>(pnls.size());
  for (const double pnl : pnls) {
    const double difference = pnl - summary.mean_total_pnl;
    summary.std_total_pnl += difference * difference;
  }
  if (pnls.size() > 1) {
    summary.std_total_pnl =
        std::sqrt(summary.std_total_pnl / static_cast<double>(pnls.size() - 1));
  } else {
    summary.std_total_pnl = 0.0;
  }
  const double count = static_cast<double>(reports.size());
  summary.mean_fill_rate /= count;
  summary.mean_max_drawdown /= count;
  summary.mean_turnover /= count;
  summary.mean_max_inventory /= count;
  return summary;
}

bool same_hawkes_parameters(const lob::stochastic::HawkesParameters& left,
                            const lob::stochastic::HawkesParameters& right) {
  return left.baseline_events_per_second == right.baseline_events_per_second &&
         left.alpha_branching == right.alpha_branching &&
         left.beta_per_second == right.beta_per_second;
}

lob::core::PriceTicks checked_price(const double price, const bool round_up) {
  if (!std::isfinite(price)) {
    throw std::overflow_error{"symmetric quote price is not finite"};
  }
  const long double rounded =
      round_up ? std::ceil(static_cast<long double>(price))
               : std::floor(static_cast<long double>(price));
  const long double min_price = static_cast<long double>(
      std::numeric_limits<lob::core::PriceTicks>::min());
  if (rounded < min_price || rounded >= -min_price) {
    throw std::overflow_error{"symmetric quote price is out of range"};
  }
  return static_cast<lob::core::PriceTicks>(rounded);
}

DirectionScore score_direction_model(
    const HawkesDirectionModel& model,
    const lob::stochastic::SyntheticMarketConfig& market) {
  const auto simulation = lob::stochastic::HawkesMarketAdapter{}.run(market);
  DirectionScore score;
  for (std::size_t i = 0; i < simulation.events.size(); ++i) {
    const auto& event = simulation.events[i];
    const double event_seconds =
        static_cast<double>(simulation.book_snapshots[i + 1].timestamp_ns) *
        1.0e-9;
    const auto end_index = future_snapshot_index(
        simulation, event_seconds + model.forecast_horizon_seconds(), i + 1);
    if (!end_index) continue;
    const auto current_mid = midprice(simulation.book_snapshots[i + 1]);
    const auto future_mid = midprice(simulation.book_snapshots[*end_index]);
    if (!current_mid || !future_mid) continue;
    const double change = *future_mid - *current_mid;
    const std::size_t label = change > 0.0 ? 2U : (change < 0.0 ? 0U : 1U);
    const auto state = flow_from_intensities(
        post_event_intensities(event, market.hawkes));
    const auto prediction = model.predict(state);
    const std::array<double, 3> probabilities{
        prediction.probability_down, prediction.probability_unchanged,
        prediction.probability_up};
    const auto& base_rates = model.training_class_probabilities();
    score.log_loss -= std::log(std::max(
        probabilities[label], std::numeric_limits<double>::min()));
    score.base_rate_log_loss -= std::log(std::max(
        base_rates[label], std::numeric_limits<double>::min()));
    const auto predicted_class = static_cast<std::size_t>(
        std::distance(probabilities.begin(),
                      std::max_element(probabilities.begin(), probabilities.end())));
    if (predicted_class == label) score.accuracy += 1.0;
    const auto base_rate_class = static_cast<std::size_t>(
        std::distance(base_rates.begin(),
                      std::max_element(base_rates.begin(), base_rates.end())));
    if (base_rate_class == label) score.base_rate_accuracy += 1.0;
    ++score.observations;
  }
  if (score.observations > 0) {
    score.log_loss /= static_cast<double>(score.observations);
    score.accuracy /= static_cast<double>(score.observations);
    score.base_rate_log_loss /= static_cast<double>(score.observations);
    score.base_rate_accuracy /= static_cast<double>(score.observations);
  }
  return score;
}

class SymmetricMarketMakerAgent final : public TradingAgent {
 public:
  explicit SymmetricMarketMakerAgent(SymmetricMarketMakerConfig config)
      : config_(config) {}

  std::string name() const override { return "SymmetricMarketMaker"; }

  void on_start(const MarketView& view, OrderGateway& gateway) override {
    refresh(view, gateway);
  }
  void on_market_event(const MarketView& view, OrderGateway& gateway) override {
    refresh(view, gateway);
  }
  void on_execution(const ExecutionNotification& report,
                    const MarketView& view,
                    OrderGateway&) override {
    const auto it = orders_.find(report.order_id);
    if (report.status == lob::strategy::StrategyOrderStatus::Filled ||
        report.status == lob::strategy::StrategyOrderStatus::Cancelled ||
        report.status == lob::strategy::StrategyOrderStatus::Rejected ||
        report.status == lob::strategy::StrategyOrderStatus::Expired) {
      if (it != orders_.end()) {
        sides_.erase(it->second.side);
        orders_.erase(it);
      }
    }
    static_cast<void>(view);
  }
  void on_timer(const MarketView& view, OrderGateway& gateway) override {
    refresh(view, gateway);
  }
  void on_stop(const MarketView&, OrderGateway& gateway) override {
    for (auto& [id, order] : orders_) {
      if (!order.cancel_requested) {
        gateway.cancel(id);
        order.cancel_requested = true;
      }
    }
  }

 private:
  struct Order final {
    lob::core::Side side{lob::core::Side::Buy};
    lob::core::PriceTicks price{};
    lob::core::QuantityLots remaining{};
    bool cancel_requested{};
  };

  void refresh(const MarketView& view, OrderGateway& gateway) {
    if (!view.midprice_ticks) return;
    auto bid = checked_price(
        *view.midprice_ticks -
            static_cast<double>(config_.quote_offset_ticks),
        false);
    auto ask = checked_price(
        *view.midprice_ticks +
            static_cast<double>(config_.quote_offset_ticks),
        true);
    if (view.best_ask && bid >= *view.best_ask) {
      if (*view.best_ask == std::numeric_limits<lob::core::PriceTicks>::min()) {
        throw std::overflow_error{"no non-crossing symmetric bid is representable"};
      }
      bid = *view.best_ask - 1;
    }
    if (view.best_bid && ask <= *view.best_bid) {
      if (*view.best_bid == std::numeric_limits<lob::core::PriceTicks>::max()) {
        throw std::overflow_error{"no non-crossing symmetric ask is representable"};
      }
      ask = *view.best_bid + 1;
    }
    reconcile_side(view, gateway, lob::core::Side::Buy, bid);
    reconcile_side(view, gateway, lob::core::Side::Sell, ask);
  }

  void reconcile_side(const MarketView& view,
                      OrderGateway& gateway,
                      const lob::core::Side side,
                      const lob::core::PriceTicks price) {
    const auto active = sides_.find(side);
    if (active != sides_.end()) {
      const auto order = orders_.find(active->second);
      if (order != orders_.end()) {
        if (order->second.cancel_requested) return;
        if (order->second.price == price) return;
        gateway.cancel(order->first);
        order->second.cancel_requested = true;
        return;
      }
      sides_.erase(active);
    }
    const auto limit = static_cast<std::int64_t>(config_.inventory_limit_lots);
    const auto position = view.inventory_lots;
    if ((side == lob::core::Side::Buy && position >= limit) ||
        (side == lob::core::Side::Sell && position <= -limit)) {
      return;
    }
    std::int64_t capacity =
        side == lob::core::Side::Buy ? limit - position : limit + position;
    for (const auto& [id, order] : orders_) {
      static_cast<void>(id);
      if (order.side == side) {
        capacity -= static_cast<std::int64_t>(order.remaining);
      }
    }
    if (capacity <= 0) return;
    const auto quantity = std::min(
        config_.quote_size_lots, static_cast<lob::core::QuantityLots>(capacity));
    const auto id = gateway.submit_limit(side, price, quantity);
    orders_.emplace(id, Order{side, price, quantity, false});
    sides_[side] = id;
  }

  SymmetricMarketMakerConfig config_;
  std::map<lob::core::OrderId, Order> orders_;
  std::map<lob::core::Side, lob::core::OrderId> sides_;
};

}  // namespace

HawkesDirectionModel HawkesDirectionModel::train(
    const std::vector<lob::stochastic::SyntheticMarketConfig>& training_runs,
    const HawkesDirectionFitConfig& fit_config) {
  if (training_runs.empty() ||
      !std::isfinite(fit_config.forecast_horizon_seconds) ||
      fit_config.forecast_horizon_seconds <= 0.0 ||
      fit_config.training_epochs == 0 ||
      !std::isfinite(fit_config.learning_rate) || fit_config.learning_rate <= 0.0 ||
      !std::isfinite(fit_config.l2_regularization) ||
      fit_config.l2_regularization < 0.0) {
    throw std::invalid_argument{"invalid Hawkes direction fit configuration"};
  }

  std::vector<TrainingObservation> observations;
  for (const auto& run : training_runs) {
    append_training_observations(run, fit_config.forecast_horizon_seconds,
                                 observations);
  }
  if (observations.size() < 3) {
    throw std::invalid_argument{
        "too few directional training observations; increase run horizons or runs"};
  }

  HawkesDirectionModel model;
  model.forecast_horizon_seconds_ = fit_config.forecast_horizon_seconds;
  model.training_observations_ = observations.size();
  for (const auto& observation : observations) {
    for (std::size_t f = 0; f < observation.features.size(); ++f) {
      model.feature_means_[f] += observation.features[f];
    }
  }
  for (auto& mean : model.feature_means_) {
    mean /= static_cast<double>(observations.size());
  }
  for (const auto& observation : observations) {
    for (std::size_t f = 0; f < observation.features.size(); ++f) {
      const double difference =
          observation.features[f] - model.feature_means_[f];
      model.feature_scales_[f] += difference * difference;
    }
    if (observation.label == 2U) {
      model.mean_up_move_ticks_ += observation.price_change_ticks;
    } else if (observation.label == 0U) {
      model.mean_down_move_ticks_ -= observation.price_change_ticks;
    }
  }
  std::size_t up_count = 0;
  std::size_t down_count = 0;
  std::array<std::size_t, 3> class_counts{};
  for (const auto& observation : observations) {
    ++class_counts[observation.label];
    if (observation.label == 2U) ++up_count;
    if (observation.label == 0U) ++down_count;
  }
  for (std::size_t c = 0; c < class_counts.size(); ++c) {
    model.training_class_probabilities_[c] =
        static_cast<double>(class_counts[c]) /
        static_cast<double>(observations.size());
  }
  if (up_count > 0) {
    model.mean_up_move_ticks_ /= static_cast<double>(up_count);
  }
  if (down_count > 0) {
    model.mean_down_move_ticks_ /= static_cast<double>(down_count);
  }
  for (auto& scale : model.feature_scales_) {
    scale = std::sqrt(scale / static_cast<double>(observations.size()));
    if (!std::isfinite(scale) || scale < 1.0e-9) scale = 1.0;
  }

  Weights weights{};
  const double sample_count = static_cast<double>(observations.size());
  for (std::size_t epoch = 0; epoch < fit_config.training_epochs; ++epoch) {
    Weights gradient{};
    for (const auto& observation : observations) {
      const auto probabilities = class_probabilities(
          observation.features, model.feature_means_, model.feature_scales_,
          weights);
      for (std::size_t c = 0; c < probabilities.size(); ++c) {
        const double error =
            probabilities[c] - (c == observation.label ? 1.0 : 0.0);
        for (std::size_t f = 0; f < observation.features.size(); ++f) {
          gradient[c][f] +=
              error * (observation.features[f] - model.feature_means_[f]) /
              model.feature_scales_[f];
        }
        gradient[c][7] += error;
      }
    }
    for (std::size_t c = 0; c < weights.size(); ++c) {
      for (std::size_t f = 0; f < weights[c].size(); ++f) {
        const double penalty = f == 7 ? 0.0
                                       : fit_config.l2_regularization * weights[c][f];
        weights[c][f] -= fit_config.learning_rate *
                         (gradient[c][f] / sample_count + penalty);
      }
    }
  }
  model.weights_ = weights;

  double loss = 0.0;
  for (const auto& observation : observations) {
    const auto probabilities = class_probabilities(
        observation.features, model.feature_means_, model.feature_scales_,
        model.weights_);
    loss -= std::log(std::max(probabilities[observation.label],
                              std::numeric_limits<double>::min()));
  }
  model.training_log_loss_ = loss / sample_count;
  return model;
}

HawkesDirectionPrediction HawkesDirectionModel::predict(
    const HawkesFlowState& state) const {
  if (training_observations_ == 0) {
    throw std::logic_error{"Hawkes direction model has not been trained"};
  }
  const auto probabilities = class_probabilities(
      make_features(state), feature_means_, feature_scales_, weights_);
  return {
      .probability_up = probabilities[2],
      .probability_down = probabilities[0],
      .probability_unchanged = probabilities[1],
      .mean_up_move_ticks = mean_up_move_ticks_,
      .mean_down_move_ticks = mean_down_move_ticks_,
  };
}

std::vector<PolicyComparisonResult> compare_market_making_policies(
    const std::vector<lob::stochastic::SyntheticMarketConfig>& training_runs,
    const std::vector<PolicyComparisonScenario>& evaluation_scenarios,
    const HawkesDirectionFitConfig& fit_config,
    const AvellanedaStoikovConfig& avellaneda_config,
    const HawkesAwareConfig& hawkes_config,
    const SymmetricMarketMakerConfig& symmetric_config,
    lob::strategy::StrategyEvaluatorConfig evaluation_config) {
  if (training_runs.empty() || evaluation_scenarios.empty() ||
      symmetric_config.quote_offset_ticks < 0 ||
      symmetric_config.quote_size_lots == 0 ||
      symmetric_config.inventory_limit_lots == 0 ||
      symmetric_config.inventory_limit_lots >
          static_cast<lob::core::QuantityLots>(
              std::numeric_limits<std::int64_t>::max()) ||
      !std::isfinite(hawkes_config.mean_market_order_size_lots) ||
      hawkes_config.mean_market_order_size_lots <= 0.0) {
    throw std::invalid_argument{"invalid policy comparison configuration"};
  }
  std::vector<std::uint64_t> training_seeds;
  for (const auto& run : training_runs) training_seeds.push_back(run.seed);
  std::sort(training_seeds.begin(), training_seeds.end());
  if (std::adjacent_find(training_seeds.begin(), training_seeds.end()) !=
      training_seeds.end()) {
    throw std::invalid_argument{"training simulations must use independent seeds"};
  }
  for (const auto& scenario : evaluation_scenarios) {
    if (scenario.label.empty() || scenario.evaluation_seeds.empty()) {
      throw std::invalid_argument{"evaluation scenarios need labels and seeds"};
    }
    auto scenario_seeds = scenario.evaluation_seeds;
    std::sort(scenario_seeds.begin(), scenario_seeds.end());
    if (std::adjacent_find(scenario_seeds.begin(), scenario_seeds.end()) !=
        scenario_seeds.end()) {
      throw std::invalid_argument{
          "an evaluation scenario contains duplicate random seeds"};
    }
    for (const auto seed : scenario_seeds) {
      if (std::find(training_seeds.begin(), training_seeds.end(), seed) !=
          training_seeds.end()) {
        throw std::invalid_argument{
            "training and evaluation random seeds must be independent"};
      }
    }
    const bool overlaps_training = std::any_of(
        training_runs.begin(), training_runs.end(),
        [&](const auto& training) {
          return same_hawkes_parameters(training.hawkes,
                                        scenario.market.hawkes);
        });
    if (overlaps_training) {
      throw std::invalid_argument{
          "evaluation Hawkes parameters must be unseen during training"};
    }
  }

  const auto direction_model =
      HawkesDirectionModel::train(training_runs, fit_config);
  std::vector<PolicyComparisonResult> results;
  results.reserve(evaluation_scenarios.size());
  for (const auto& scenario : evaluation_scenarios) {
    PolicyComparisonResult result;
    result.scenario_label = scenario.label;
    for (const auto seed : scenario.evaluation_seeds) {
      auto market = scenario.market;
      market.seed = seed;
      {
        AvellanedaStoikovAgent agent{avellaneda_config};
        result.classical_as.push_back(
            ExchangeAdapter{}.run(agent, market, evaluation_config));
      }
      {
        AvellanedaStoikovAgent agent{
            avellaneda_config, &direction_model,
            hawkes_config.mean_market_order_size_lots};
        result.hawkes_aware.push_back(
            ExchangeAdapter{}.run(agent, market, evaluation_config));
      }
      {
        SymmetricMarketMakerAgent agent{symmetric_config};
        result.symmetric.push_back(
            ExchangeAdapter{}.run(agent, market, evaluation_config));
      }
      const auto score = score_direction_model(direction_model, market);
      if (score.observations > 0) {
        result.direction_observations += score.observations;
        result.direction_log_loss +=
            score.log_loss * static_cast<double>(score.observations);
        result.direction_accuracy +=
            score.accuracy * static_cast<double>(score.observations);
        result.direction_base_rate_log_loss +=
            score.base_rate_log_loss *
            static_cast<double>(score.observations);
        result.direction_base_rate_accuracy +=
            score.base_rate_accuracy *
            static_cast<double>(score.observations);
      }
    }
    result.classical_summary = summarize(result.classical_as);
    result.hawkes_summary = summarize(result.hawkes_aware);
    result.symmetric_summary = summarize(result.symmetric);
    if (result.direction_observations > 0) {
      const double count = static_cast<double>(result.direction_observations);
      result.direction_log_loss /= count;
      result.direction_accuracy /= count;
      result.direction_base_rate_log_loss /= count;
      result.direction_base_rate_accuracy /= count;
    }
    results.push_back(std::move(result));
  }
  return results;
}

}  // namespace project3
