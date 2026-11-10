#include "project3/market_quality.hpp"
#include "project3/detail/parallel.hpp"

#include "lob/stochastic/hawkes_parameters.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>

namespace project3 {
namespace {

constexpr double kNanosecondsPerSecond = 1.0e9;

double side_sign(const lob::core::Side side) noexcept {
  return side == lob::core::Side::Buy ? 1.0 : -1.0;
}

lob::core::QuantityLots scaled_lots(const lob::core::QuantityLots lots,
                                   const double scale) {
  const double scaled =
      std::max(1.0, std::floor(static_cast<double>(lots) * scale));
  if (scaled > static_cast<double>(
                   std::numeric_limits<lob::core::QuantityLots>::max())) {
    throw std::overflow_error{"market stress size exceeds quantity range"};
  }
  return static_cast<lob::core::QuantityLots>(scaled);
}

void scale_baseline(lob::stochastic::SyntheticMarketConfig& market,
                    const std::size_t channel,
                    const double scale) {
  if (channel >= market.hawkes.baseline_events_per_second.size()) {
    throw std::invalid_argument{"stress requires six Hawkes event channels"};
  }
  market.hawkes.baseline_events_per_second[channel] *= scale;
}

lob::stochastic::SyntheticMarketConfig apply_stress(
    const lob::stochastic::SyntheticMarketConfig& baseline,
    const MarketStressKind kind,
    const MarketStressConfig& stress) {
  auto market = baseline;
  switch (kind) {
    case MarketStressKind::VolatilityIncrease:
      scale_baseline(market, 4, stress.volatility_flow_multiplier);
      scale_baseline(market, 5, stress.volatility_flow_multiplier);
      market.marks.market_size.min_lots =
          scaled_lots(market.marks.market_size.min_lots,
                      stress.volatility_flow_multiplier);
      market.marks.market_size.max_lots =
          scaled_lots(market.marks.market_size.max_lots,
                      stress.volatility_flow_multiplier);
      if (market.marks.price_placement.max_offset_ticks >
          std::numeric_limits<lob::core::PriceTicks>::max() - 2) {
        throw std::overflow_error{
            "volatility stress placement offset exceeds price range"};
      }
      market.marks.price_placement.max_offset_ticks += 2;
      break;
    case MarketStressKind::OneSidedOrderFlowBurst:
      scale_baseline(market, 4, stress.one_sided_flow_multiplier);
      break;
    case MarketStressKind::HawkesExcitationIncrease:
      for (auto& row : market.hawkes.alpha_branching) {
        for (double& value : row) {
          value *= stress.hawkes_excitation_multiplier;
        }
      }
      break;
    case MarketStressKind::LiquidityWithdrawal:
      market.initial_book.order_size =
          scaled_lots(market.initial_book.order_size,
                      stress.liquidity_remaining_fraction);
      market.initial_book.orders_per_level = std::max<std::size_t>(
          1, static_cast<std::size_t>(std::floor(
                 static_cast<double>(market.initial_book.orders_per_level) *
                 stress.liquidity_remaining_fraction)));
      for (auto& order : market.initial_book.explicit_orders) {
        order.quantity =
            scaled_lots(order.quantity, stress.liquidity_remaining_fraction);
      }
      break;
    case MarketStressKind::CancellationSurge:
      scale_baseline(market, 2, stress.cancellation_multiplier);
      scale_baseline(market, 3, stress.cancellation_multiplier);
      break;
    case MarketStressKind::SpreadWidening: {
      if (stress.spread_widening_ticks <= 0) {
        throw std::invalid_argument{
            "spread widening stress must be a positive number of ticks"};
      }
      const auto bid_shift = stress.spread_widening_ticks / 2;
      const auto ask_shift =
          stress.spread_widening_ticks - bid_shift;
      if (market.initial_book.explicit_orders.empty()) {
        if (market.initial_book.best_bid_ticks <
                std::numeric_limits<lob::core::PriceTicks>::min() +
                    bid_shift ||
            market.initial_book.best_ask_ticks >
                std::numeric_limits<lob::core::PriceTicks>::max() -
                    ask_shift) {
          throw std::overflow_error{
              "spread widening stress exceeds price range"};
        }
        market.initial_book.best_bid_ticks -= bid_shift;
        market.initial_book.best_ask_ticks += ask_shift;
      } else {
        for (auto& order : market.initial_book.explicit_orders) {
          if (order.side == lob::core::Side::Buy) {
            if (order.price_ticks <
                std::numeric_limits<lob::core::PriceTicks>::min() +
                    bid_shift) {
              throw std::overflow_error{
                  "spread widening stress exceeds price range"};
            }
            order.price_ticks -= bid_shift;
          } else {
            if (order.price_ticks >
                std::numeric_limits<lob::core::PriceTicks>::max() -
                    ask_shift) {
              throw std::overflow_error{
                  "spread widening stress exceeds price range"};
            }
            order.price_ticks += ask_shift;
          }
        }
      }
      break;
    }
    case MarketStressKind::PriceJumps:
      market.marks.market_size.min_lots =
          scaled_lots(market.marks.market_size.min_lots,
                      stress.price_jump_size_multiplier);
      market.marks.market_size.max_lots =
          scaled_lots(market.marks.market_size.max_lots,
                      stress.price_jump_size_multiplier);
      break;
  }
  lob::stochastic::validate_stationary(market.hawkes);
  return market;
}

struct Observation final {
  lob::core::TimestampNs timestamp_ns{};
  std::optional<double> midprice_ticks;
  std::int64_t inventory_lots{};
  double marked_equity_currency{};
};

struct EquitySample final {
  lob::core::TimestampNs timestamp_ns{};
  double equity_currency{};
};

struct CapturedOrder final {
  lob::core::Side side{lob::core::Side::Buy};
  lob::core::TimestampNs submitted_ns{};
  std::optional<double> arrival_midprice_ticks;
  bool cancelled{};
  bool has_fill{};
  lob::core::TimestampNs first_fill_ns{};
};

struct CapturedFill final {
  ExecutionNotification notification;
  lob::core::TimestampNs execution_ns{};
};

class RecordingGateway final : public OrderGateway {
 public:
  RecordingGateway(OrderGateway* downstream,
                   std::map<lob::core::OrderId, CapturedOrder>& orders,
                   const MarketView*& current_view)
      : downstream_(downstream), orders_(orders), current_view_(current_view) {}

  void update(OrderGateway& downstream) noexcept { downstream_ = &downstream; }

  [[nodiscard]] lob::core::OrderId submit_limit(
      const lob::core::Side side,
      const lob::core::PriceTicks price,
      const lob::core::QuantityLots quantity) override {
    const auto id = require_downstream().submit_limit(side, price, quantity);
    orders_.emplace(
        id, CapturedOrder{.side = side,
                          .submitted_ns = current_view_ == nullptr
                                              ? 0
                                              : current_view_->timestamp_ns,
                          .arrival_midprice_ticks =
                              current_view_ == nullptr
                                  ? std::nullopt
                                  : current_view_->midprice_ticks});
    return id;
  }

  [[nodiscard]] lob::core::OrderId submit_market(
      const lob::core::Side side,
      const lob::core::QuantityLots quantity) override {
    const auto id = require_downstream().submit_market(side, quantity);
    orders_.emplace(
        id, CapturedOrder{.side = side,
                          .submitted_ns = current_view_ == nullptr
                                              ? 0
                                              : current_view_->timestamp_ns,
                          .arrival_midprice_ticks =
                              current_view_ == nullptr
                                  ? std::nullopt
                                  : current_view_->midprice_ticks});
    return id;
  }

  void cancel(const lob::core::OrderId id) override {
    require_downstream().cancel(id);
  }

 private:
  [[nodiscard]] OrderGateway& require_downstream() const {
    if (downstream_ == nullptr) {
      throw std::logic_error{"recording gateway has no active exchange"};
    }
    return *downstream_;
  }

  OrderGateway* downstream_{};
  std::map<lob::core::OrderId, CapturedOrder>& orders_;
  const MarketView*& current_view_;
};

class RecordingAgent final : public TradingAgent {
 public:
  explicit RecordingAgent(TradingAgent& downstream) : downstream_(downstream) {}

  [[nodiscard]] std::string name() const override {
    return downstream_.name();
  }

  void on_start(const MarketView& view, OrderGateway& gateway) override {
    observe_equity(view);
    with_view(view, gateway, [&] {
      downstream_.on_start(view, recording_gateway_);
    });
  }

  void on_market_event(const MarketView& view,
                      OrderGateway& gateway) override {
    with_view(view, gateway, [&] {
      downstream_.on_market_event(view, recording_gateway_);
    });
  }

  void on_trade(const lob::strategy::StrategyMarketTrade& trade,
                const MarketView& view) override {
    observe(view);
    downstream_.on_trade(trade, view);
  }

  void on_execution(const ExecutionNotification& notification,
                    const MarketView& view,
                    OrderGateway& gateway) override {
    observe(view);
    with_view(view, gateway, [&] {
      downstream_.on_execution(notification, view, recording_gateway_);
    });
  }

  void on_timer(const MarketView& view, OrderGateway& gateway) override {
    observe_equity(view);
    with_view(view, gateway, [&] {
      downstream_.on_timer(view, recording_gateway_);
    });
  }

  void on_stop(const MarketView& view, OrderGateway& gateway) override {
    observe_equity(view);
    with_view(view, gateway, [&] {
      downstream_.on_stop(view, recording_gateway_);
    });
  }

  [[nodiscard]] const std::map<lob::core::OrderId, CapturedOrder>&
  orders() const noexcept {
    return orders_;
  }
  [[nodiscard]] const std::vector<Observation>& observations() const noexcept {
    return observations_;
  }
  [[nodiscard]] const std::vector<EquitySample>& equity_samples() const noexcept {
    return equity_samples_;
  }

 private:
  template <typename Callback>
  void with_view(const MarketView& view,
                 OrderGateway& gateway,
                 Callback&& callback) {
    observe(view);
    current_view_ = &view;
    recording_gateway_.update(gateway);
    std::forward<Callback>(callback)();
    current_view_ = nullptr;
  }

  void observe(const MarketView& view) {
    observations_.push_back({.timestamp_ns = view.timestamp_ns,
                             .midprice_ticks = view.midprice_ticks,
                             .inventory_lots = view.inventory_lots,
                             .marked_equity_currency =
                                 view.market_maker.account.marked_equity_currency});
  }

  void observe_equity(const MarketView& view) {
    const EquitySample sample{
        .timestamp_ns = view.timestamp_ns,
        .equity_currency =
            view.market_maker.account.marked_equity_currency};
    if (!equity_samples_.empty() &&
        equity_samples_.back().timestamp_ns == sample.timestamp_ns) {
      equity_samples_.back() = sample;
    } else {
      equity_samples_.push_back(sample);
    }
  }

  TradingAgent& downstream_;
  std::map<lob::core::OrderId, CapturedOrder> orders_;
  std::vector<Observation> observations_;
  std::vector<EquitySample> equity_samples_;
  const MarketView* current_view_{};
  RecordingGateway recording_gateway_{nullptr, orders_, current_view_};
};

void validate_config(const MarketQualityConfig& config) {
  if (config.markout_horizons_seconds.empty() ||
      !std::isfinite(config.expected_shortfall_confidence) ||
      config.expected_shortfall_confidence <= 0.0 ||
      config.expected_shortfall_confidence >= 1.0 ||
      !std::isfinite(config.inventory_limit_lots) ||
      config.inventory_limit_lots <= 0.0 ||
      !std::isfinite(config.currency_per_tick) ||
      config.currency_per_tick <= 0.0 ||
      !std::isfinite(config.sharpe_capital_base_currency) ||
      config.sharpe_capital_base_currency <= 0.0 ||
      !std::isfinite(config.annualization_seconds) ||
      config.annualization_seconds <= 0.0) {
    throw std::invalid_argument{"invalid market-quality analysis configuration"};
  }
  double previous = -1.0;
  for (const double horizon : config.markout_horizons_seconds) {
    const double maximum_horizon =
        static_cast<double>(
            std::numeric_limits<lob::core::TimestampNs>::max()) /
        kNanosecondsPerSecond;
    if (!std::isfinite(horizon) || horizon <= 0.0 ||
        horizon > maximum_horizon || horizon <= previous) {
      throw std::invalid_argument{
          "markout horizons must be positive, finite, and increasing"};
    }
    previous = horizon;
  }
}

std::vector<Observation> normalized_observations(
    const RecordingAgent& recorder,
    const lob::core::TimestampNs horizon_ns) {
  auto observations = recorder.observations();
  std::stable_sort(observations.begin(), observations.end(),
                   [](const Observation& left, const Observation& right) {
                     return left.timestamp_ns < right.timestamp_ns;
                   });
  std::vector<Observation> unique;
  unique.reserve(observations.size() + 1);
  for (const auto& observation : observations) {
    if (!unique.empty() &&
        unique.back().timestamp_ns == observation.timestamp_ns) {
      unique.back() = observation;
    } else {
      unique.push_back(observation);
    }
  }
  if (unique.empty() || unique.front().timestamp_ns > 0) {
    unique.insert(unique.begin(), Observation{});
  }
  if (unique.back().timestamp_ns < horizon_ns) {
    unique.push_back({.timestamp_ns = horizon_ns,
                      .midprice_ticks = unique.back().midprice_ticks,
                      .inventory_lots = unique.back().inventory_lots});
  }
  return unique;
}

double mean_or_zero(const double sum, const std::size_t count) {
  return count == 0 ? 0.0 : sum / static_cast<double>(count);
}

void compute_inventory_metrics(MarketQualityEpisode& episode,
                               const std::vector<Observation>& observations,
                               const double horizon_seconds,
                               const double inventory_limit) {
  double squared_inventory_seconds = 0.0;
  double breach_time = 0.0;
  double max_inventory = 0.0;
  std::size_t breach_intervals = 0;
  const double limit_squared = inventory_limit * inventory_limit;
  for (std::size_t i = 0; i + 1 < observations.size(); ++i) {
    const double duration =
        static_cast<double>(observations[i + 1].timestamp_ns -
                            observations[i].timestamp_ns) /
        kNanosecondsPerSecond;
    if (duration <= 0.0) continue;
    const double inventory =
        static_cast<double>(observations[i].inventory_lots);
    const double absolute_inventory = std::abs(inventory);
    max_inventory = std::max(max_inventory, absolute_inventory);
    episode.inventory_duration_lot_seconds +=
        absolute_inventory * duration;
    squared_inventory_seconds += inventory * inventory * duration;
    if (absolute_inventory > inventory_limit) {
      breach_time += duration;
      ++breach_intervals;
    }
    episode.inventory_risk_lot_squared_seconds = squared_inventory_seconds;
  }
  for (const auto& observation : observations) {
    max_inventory =
        std::max(max_inventory,
                 std::abs(static_cast<double>(observation.inventory_lots)));
  }
  episode.mean_absolute_inventory_lots =
      horizon_seconds > 0.0
          ? episode.inventory_duration_lot_seconds / horizon_seconds
          : 0.0;
  episode.inventory_concentration_ratio =
      horizon_seconds > 0.0 && limit_squared > 0.0
          ? squared_inventory_seconds / (horizon_seconds * limit_squared)
          : 0.0;
  episode.inventory_limit_breach_duration_seconds = breach_time;
  episode.inventory_limit_breach_intervals = breach_intervals;
  episode.maximum_inventory_lots = max_inventory;
}

double compute_sharpe_ratio(const std::vector<EquitySample>& samples,
                            const double capital_base,
                            const double annualization_seconds) {
  if (samples.size() < 3) return 0.0;
  std::vector<double> returns;
  double elapsed_seconds = 0.0;
  returns.reserve(samples.size() - 1);
  for (std::size_t i = 0; i + 1 < samples.size(); ++i) {
    const auto dt_ns =
        samples[i + 1].timestamp_ns - samples[i].timestamp_ns;
    if (dt_ns <= 0) continue;
    returns.push_back((samples[i + 1].equity_currency -
                       samples[i].equity_currency) /
                      capital_base);
    elapsed_seconds += static_cast<double>(dt_ns) / kNanosecondsPerSecond;
  }
  if (returns.size() < 2 || elapsed_seconds <= 0.0) return 0.0;
  const double mean =
      std::accumulate(returns.begin(), returns.end(), 0.0) /
      static_cast<double>(returns.size());
  double squared_error = 0.0;
  for (const double value : returns) {
    const double difference = value - mean;
    squared_error += difference * difference;
  }
  const double standard_deviation =
      std::sqrt(squared_error / static_cast<double>(returns.size() - 1));
  if (standard_deviation == 0.0) return 0.0;
  const double mean_interval_seconds =
      elapsed_seconds / static_cast<double>(returns.size());
  return mean / standard_deviation *
         std::sqrt(annualization_seconds / mean_interval_seconds);
}

double observation_at_or_after(const std::vector<Observation>& observations,
                               const lob::core::TimestampNs timestamp_ns) {
  const auto found = std::lower_bound(
      observations.begin(), observations.end(), timestamp_ns,
      [](const Observation& observation, const lob::core::TimestampNs time) {
        return observation.timestamp_ns < time;
      });
  for (auto current = found; current != observations.end(); ++current) {
    if (current->midprice_ticks.has_value()) return *current->midprice_ticks;
  }
  return std::numeric_limits<double>::quiet_NaN();
}

void compute_markout(MarketQualityEpisode& episode,
                     const std::map<lob::core::OrderId, CapturedOrder>& orders,
                     const std::vector<CapturedFill>& fills,
                     const std::vector<Observation>& observations,
                     const MarketQualityConfig& config) {
  episode.markouts.reserve(config.markout_horizons_seconds.size());
  for (const double horizon : config.markout_horizons_seconds) {
    episode.markouts.push_back({.horizon_seconds = horizon});
  }

  for (const auto& fill : fills) {
    const auto& notification = fill.notification;
    if (notification.filled_quantity <= 0) continue;
    const auto side = notification.side;
    const double sign = side_sign(side);
    const double quantity =
        static_cast<double>(notification.filled_quantity);
    const double execution_price =
        static_cast<double>(notification.price_ticks);
    const double signed_execution_cost_ticks =
        notification.execution_cost_currency /
        (quantity * config.currency_per_tick);
    const double arrival_mid =
        execution_price - signed_execution_cost_ticks / sign;

    auto& directional = side == lob::core::Side::Buy
                            ? episode.market_maker_buys
                            : episode.market_maker_sells;
    directional.execution_count += 1;
    directional.executed_quantity_lots += quantity;
    directional.effective_spread_ticks +=
        -2.0 * signed_execution_cost_ticks * quantity;
    const auto order = orders.find(notification.order_id);
    if (order != orders.end()) {
      if (order->second.arrival_midprice_ticks.has_value()) {
        directional.implementation_shortfall_ticks +=
            sign * (execution_price -
                    *order->second.arrival_midprice_ticks) *
            quantity;
        directional.shortfall_quantity_lots += quantity;
      }
    }
    if (order != orders.end() &&
        order->second.first_fill_ns == fill.execution_ns) {
      const double wait_seconds =
          static_cast<double>(fill.execution_ns -
                              order->second.submitted_ns) /
          kNanosecondsPerSecond;
      directional.queue_waiting_time_seconds +=
          std::max(0.0, wait_seconds);
    }

    for (std::size_t h = 0; h < config.markout_horizons_seconds.size(); ++h) {
      const auto horizon_ns = static_cast<lob::core::TimestampNs>(
          std::ceil(config.markout_horizons_seconds[h] *
                    kNanosecondsPerSecond));
      const auto target_time =
          fill.execution_ns >
                  std::numeric_limits<lob::core::TimestampNs>::max() -
                      horizon_ns
              ? std::numeric_limits<lob::core::TimestampNs>::max()
              : fill.execution_ns + horizon_ns;
      const double future_mid = observation_at_or_after(
          observations, target_time);
      auto& aggregate = side == lob::core::Side::Buy
                            ? episode.markouts[h].market_maker_buys
                            : episode.markouts[h].market_maker_sells;
      if (!std::isfinite(future_mid)) {
        ++aggregate.right_censored_execution_count;
        continue;
      }
      const double adverse =
          adverse_selection_ticks(side, future_mid, execution_price);
      const double impact = -sign * (future_mid - arrival_mid);
      aggregate.execution_count += 1;
      aggregate.executed_quantity_lots += quantity;
      aggregate.adverse_selection_ticks += adverse * quantity;
      aggregate.adverse_selection_currency +=
          adverse * quantity * config.currency_per_tick;
      aggregate.realized_spread_ticks += 2.0 * adverse * quantity;
      aggregate.price_impact_ticks += impact * quantity;
    }
  }

  const auto finalize_execution = [](DirectionalExecutionQuality& quality) {
    if (quality.executed_quantity_lots > 0.0) {
      quality.effective_spread_ticks /= quality.executed_quantity_lots;
    }
    if (quality.shortfall_quantity_lots > 0.0) {
      quality.implementation_shortfall_ticks /=
          quality.shortfall_quantity_lots;
    }
  };
  finalize_execution(episode.market_maker_buys);
  finalize_execution(episode.market_maker_sells);
  for (std::size_t h = 0; h < episode.markouts.size(); ++h) {
    auto& buy = episode.markouts[h].market_maker_buys;
    if (buy.executed_quantity_lots > 0.0) {
      buy.adverse_selection_ticks /= buy.executed_quantity_lots;
      buy.realized_spread_ticks /= buy.executed_quantity_lots;
      buy.price_impact_ticks /= buy.executed_quantity_lots;
    }
    auto& sell = episode.markouts[h].market_maker_sells;
    if (sell.executed_quantity_lots > 0.0) {
      sell.adverse_selection_ticks /= sell.executed_quantity_lots;
      sell.adverse_selection_currency /= sell.executed_quantity_lots;
      sell.realized_spread_ticks /= sell.executed_quantity_lots;
      sell.price_impact_ticks /= sell.executed_quantity_lots;
    }
    if (buy.executed_quantity_lots > 0.0) {
      buy.adverse_selection_currency /= buy.executed_quantity_lots;
    }
  }
}

MarketQualityEpisode evaluate_episode(
    const MarketQualityPolicy& policy,
    const MarketStressScenario& scenario,
    const std::uint64_t seed,
    const MarketQualityConfig& config) {
  auto market = scenario.market;
  market.seed = seed;
  auto agent = policy.make_agent();
  if (!agent) {
    throw std::runtime_error{"market-quality policy factory returned null"};
  }
  RecordingAgent recorder{*agent};
  auto evaluation = config.evaluation;
  evaluation.broker_config.currency_per_tick = config.currency_per_tick;
  if (evaluation.timer_interval_ns <= 0) {
    evaluation.timer_interval_ns = 100'000'000;
  }
  const auto report = ExchangeAdapter{}.run(recorder, market, evaluation);
  auto orders = recorder.orders();
  std::vector<CapturedFill> fills;
  for (const auto& execution : report.execution_history) {
    const auto inbound = evaluation.broker_config.latency.inbound_delay_ns;
    const auto execution_ns =
        execution.timestamp_ns >= inbound
            ? execution.timestamp_ns - inbound
            : execution.timestamp_ns;
    const auto order = orders.find(execution.client_order_id);
    if (order != orders.end()) {
      if (execution.filled_quantity > 0) {
        if (!order->second.has_fill) {
          order->second.has_fill = true;
          order->second.first_fill_ns = execution_ns;
        }
      }
      if (execution.status ==
          lob::strategy::StrategyOrderStatus::Cancelled) {
        order->second.cancelled = true;
      }
    }
    if (execution.filled_quantity > 0) {
      fills.push_back({
          .notification =
              {.order_id = execution.client_order_id,
               .timestamp_ns = execution.timestamp_ns,
               .side = execution.side,
               .price_ticks = execution.execution_price_ticks,
               .filled_quantity = execution.filled_quantity,
               .remaining_quantity = execution.remaining_quantity,
               .fee_paid_currency = execution.fee_paid_currency,
               .execution_cost_currency = execution.execution_cost_currency,
               .status = execution.status,
               .passive = execution.is_passive},
          .execution_ns = execution_ns});
    }
  }

  MarketQualityEpisode episode;
  episode.policy_name = policy.name;
  episode.scenario_label = scenario.label;
  episode.seed = seed;
  episode.net_pnl_currency =
      report.final_portfolio.net_total_pnl_currency();
  episode.gross_realized_pnl_currency =
      report.final_portfolio.gross_realized_pnl_currency;
  episode.unrealized_pnl_currency =
      report.final_portfolio.unrealized_pnl_currency;
  episode.fees_currency = report.final_portfolio.total_fees_currency;
  episode.charged_fees_currency =
      report.final_portfolio.total_charged_fees_currency;
  episode.rebates_currency =
      report.final_portfolio.total_rebates_currency;
  episode.gross_turnover_currency =
      report.final_portfolio.gross_turnover_currency;
  for (const auto& execution : report.execution_history) {
    if (execution.filled_quantity > 0) {
      episode.spread_capture_currency -= execution.execution_cost_currency;
    }
  }
  episode.sharpe_ratio = compute_sharpe_ratio(
      recorder.equity_samples(), config.sharpe_capital_base_currency,
      config.annualization_seconds);
  episode.maximum_drawdown_currency =
      report.final_portfolio.max_drawdown_currency;

  std::size_t filled_orders = 0;
  std::size_t cancelled_orders = 0;
  double buy_wait_sum = 0.0;
  double sell_wait_sum = 0.0;
  std::size_t buy_filled_orders = 0;
  std::size_t sell_filled_orders = 0;
  for (const auto& [id, order] : orders) {
    static_cast<void>(id);
    if (order.has_fill) {
      ++filled_orders;
      const double wait =
          static_cast<double>(order.first_fill_ns - order.submitted_ns) /
          kNanosecondsPerSecond;
      if (order.side == lob::core::Side::Buy) {
        buy_wait_sum += std::max(0.0, wait);
        ++buy_filled_orders;
      } else {
        sell_wait_sum += std::max(0.0, wait);
        ++sell_filled_orders;
      }
    }
    if (order.cancelled) ++cancelled_orders;
  }
  episode.fill_probability =
      orders.empty()
          ? 0.0
          : static_cast<double>(filled_orders) /
                static_cast<double>(orders.size());
  episode.cancellation_rate =
      orders.empty()
          ? 0.0
          : static_cast<double>(cancelled_orders) /
                static_cast<double>(orders.size());
  episode.market_maker_buys.queue_waiting_time_seconds =
      mean_or_zero(buy_wait_sum, buy_filled_orders);
  episode.market_maker_sells.queue_waiting_time_seconds =
      mean_or_zero(sell_wait_sum, sell_filled_orders);
  episode.market_maker_buys.filled_order_count = buy_filled_orders;
  episode.market_maker_sells.filled_order_count = sell_filled_orders;
  const std::size_t total_filled_orders =
      buy_filled_orders + sell_filled_orders;
  episode.queue_waiting_time_seconds =
      total_filled_orders == 0
          ? 0.0
          : (episode.market_maker_buys.queue_waiting_time_seconds *
                 static_cast<double>(buy_filled_orders) +
             episode.market_maker_sells.queue_waiting_time_seconds *
                 static_cast<double>(sell_filled_orders)) /
                static_cast<double>(total_filled_orders);

  const auto horizon_ns = static_cast<lob::core::TimestampNs>(
      std::ceil(market.horizon_seconds * kNanosecondsPerSecond));
  const auto observations = normalized_observations(recorder, horizon_ns);
  compute_inventory_metrics(episode, observations, market.horizon_seconds,
                            config.inventory_limit_lots);
  compute_markout(episode, orders, fills, observations, config);
  return episode;
}

double average(const std::vector<double>& values) {
  return values.empty()
             ? 0.0
             : std::accumulate(values.begin(), values.end(), 0.0) /
                   static_cast<double>(values.size());
}

MarketQualityScenarioSummary summarize_runs(
    const std::string& policy_name,
    const std::string& scenario_label,
    std::vector<MarketQualityEpisode> runs,
    const double confidence) {
  MarketQualityScenarioSummary summary;
  summary.policy_name = policy_name;
  summary.scenario_label = scenario_label;
  summary.episodes = runs.size();
  summary.runs = std::move(runs);
  std::vector<double> pnl;
  std::vector<double> losses;
  std::size_t total_filled_orders = 0;
  pnl.reserve(summary.runs.size());
  losses.reserve(summary.runs.size());
  if (!summary.runs.empty()) {
    for (const auto& markout : summary.runs.front().markouts) {
      summary.mean_markouts.push_back(
          {.horizon_seconds = markout.horizon_seconds});
    }
  }
  for (const auto& run : summary.runs) {
    pnl.push_back(run.net_pnl_currency);
    losses.push_back(-run.net_pnl_currency);
    summary.mean_maximum_drawdown_currency +=
        run.maximum_drawdown_currency;
    summary.mean_sharpe_ratio += run.sharpe_ratio;
    summary.worst_maximum_drawdown_currency =
        std::max(summary.worst_maximum_drawdown_currency,
                 run.maximum_drawdown_currency);
    summary.mean_fill_probability += run.fill_probability;
    summary.mean_cancellation_rate += run.cancellation_rate;
    summary.mean_inventory_duration_lot_seconds +=
        run.inventory_duration_lot_seconds;
    summary.mean_inventory_risk_lot_squared_seconds +=
        run.inventory_risk_lot_squared_seconds;
    summary.mean_inventory_concentration_ratio +=
        run.inventory_concentration_ratio;
    summary.mean_turnover_currency += run.gross_turnover_currency;
    summary.mean_transaction_costs_currency += run.fees_currency;
    summary.mean_spread_capture_currency += run.spread_capture_currency;
    if (!run.markouts.empty()) {
      const auto& markout = run.markouts.front();
      const double quantity =
          markout.market_maker_buys.executed_quantity_lots +
          markout.market_maker_sells.executed_quantity_lots;
      const double weighted_adverse =
          markout.market_maker_buys.adverse_selection_ticks *
              markout.market_maker_buys.executed_quantity_lots +
          markout.market_maker_sells.adverse_selection_ticks *
              markout.market_maker_sells.executed_quantity_lots;
      if (quantity > 0.0) {
        summary.mean_adverse_selection_ticks += weighted_adverse / quantity;
      }
    }
    summary.total_inventory_limit_breach_duration_seconds +=
        run.inventory_limit_breach_duration_seconds;
    summary.total_inventory_limit_breach_intervals +=
        run.inventory_limit_breach_intervals;
    summary.inventory_limit_breached =
        summary.inventory_limit_breached ||
        run.inventory_limit_breach_intervals > 0;
    const std::size_t run_filled_orders =
        run.market_maker_buys.filled_order_count +
        run.market_maker_sells.filled_order_count;
    summary.mean_queue_waiting_time_seconds +=
        run.queue_waiting_time_seconds *
        static_cast<double>(run_filled_orders);
    total_filled_orders += run_filled_orders;
    const auto accumulate_execution = [](DirectionalExecutionQuality& target,
                                         const DirectionalExecutionQuality& source) {
      const double combined_quantity =
          target.executed_quantity_lots + source.executed_quantity_lots;
      if (combined_quantity > 0.0) {
        target.effective_spread_ticks =
            (target.effective_spread_ticks * target.executed_quantity_lots +
             source.effective_spread_ticks * source.executed_quantity_lots) /
            combined_quantity;
      }
      const double combined_shortfall_quantity =
          target.shortfall_quantity_lots + source.shortfall_quantity_lots;
      if (combined_shortfall_quantity > 0.0) {
        target.implementation_shortfall_ticks =
            (target.implementation_shortfall_ticks *
                 target.shortfall_quantity_lots +
             source.implementation_shortfall_ticks *
                 source.shortfall_quantity_lots) /
            combined_shortfall_quantity;
      }
      target.execution_count += source.execution_count;
      const auto combined_filled_orders =
          target.filled_order_count + source.filled_order_count;
      if (combined_filled_orders > 0) {
        target.queue_waiting_time_seconds =
            (target.queue_waiting_time_seconds *
                 static_cast<double>(target.filled_order_count) +
             source.queue_waiting_time_seconds *
                 static_cast<double>(source.filled_order_count)) /
            static_cast<double>(combined_filled_orders);
      }
      target.filled_order_count = combined_filled_orders;
      target.executed_quantity_lots = combined_quantity;
      target.shortfall_quantity_lots = combined_shortfall_quantity;
    };
    accumulate_execution(summary.market_maker_buys,
                         run.market_maker_buys);
    accumulate_execution(summary.market_maker_sells,
                         run.market_maker_sells);
    for (std::size_t i = 0;
         i < std::min(summary.mean_markouts.size(), run.markouts.size()); ++i) {
      const auto accumulate_markout = [](
          DirectionalMarkout& target, const DirectionalMarkout& source) {
        const double combined_quantity =
            target.executed_quantity_lots + source.executed_quantity_lots;
        if (combined_quantity > 0.0) {
          target.adverse_selection_ticks =
              (target.adverse_selection_ticks *
                   target.executed_quantity_lots +
               source.adverse_selection_ticks *
                   source.executed_quantity_lots) /
              combined_quantity;
          target.adverse_selection_currency =
              (target.adverse_selection_currency *
                   target.executed_quantity_lots +
               source.adverse_selection_currency *
                   source.executed_quantity_lots) /
              combined_quantity;
          target.realized_spread_ticks =
              (target.realized_spread_ticks *
                   target.executed_quantity_lots +
               source.realized_spread_ticks *
                   source.executed_quantity_lots) /
              combined_quantity;
          target.price_impact_ticks =
              (target.price_impact_ticks * target.executed_quantity_lots +
               source.price_impact_ticks * source.executed_quantity_lots) /
              combined_quantity;
        }
        target.execution_count += source.execution_count;
        target.executed_quantity_lots = combined_quantity;
        target.right_censored_execution_count +=
            source.right_censored_execution_count;
      };
      accumulate_markout(summary.mean_markouts[i].market_maker_buys,
                         run.markouts[i].market_maker_buys);
      accumulate_markout(summary.mean_markouts[i].market_maker_sells,
                         run.markouts[i].market_maker_sells);
    }
  }
  summary.mean_net_pnl_currency = average(pnl);
  summary.worst_net_pnl_currency =
      pnl.empty() ? 0.0 : *std::min_element(pnl.begin(), pnl.end());
  if (pnl.size() > 1) {
    double squared_error = 0.0;
    for (const double value : pnl) {
      const double difference = value - summary.mean_net_pnl_currency;
      squared_error += difference * difference;
    }
    summary.standard_deviation_net_pnl_currency =
        std::sqrt(squared_error / static_cast<double>(pnl.size() - 1));
  }
  if (!losses.empty()) {
    std::sort(losses.begin(), losses.end());
    const auto rank = static_cast<std::size_t>(
        std::ceil(confidence * static_cast<double>(losses.size())));
    const std::size_t index = std::min(rank, losses.size()) - 1;
    summary.value_at_risk_loss_currency = losses[index];
    double tail_sum = 0.0;
    std::size_t tail_count = 0;
    for (const double loss : losses) {
      if (loss >= summary.value_at_risk_loss_currency) {
        tail_sum += loss;
        ++tail_count;
      }
    }
    summary.expected_shortfall_loss_currency =
        mean_or_zero(tail_sum, tail_count);
  }
  if (summary.episodes > 0) {
    const double count = static_cast<double>(summary.episodes);
    summary.mean_maximum_drawdown_currency /= count;
    summary.mean_sharpe_ratio /= count;
    summary.mean_fill_probability /= count;
    summary.mean_cancellation_rate /= count;
    summary.mean_inventory_duration_lot_seconds /= count;
    summary.mean_inventory_risk_lot_squared_seconds /= count;
    summary.mean_inventory_concentration_ratio /= count;
    summary.mean_turnover_currency /= count;
    summary.mean_transaction_costs_currency /= count;
    summary.mean_spread_capture_currency /= count;
    summary.mean_adverse_selection_ticks /= count;
  }
  if (total_filled_orders > 0) {
    summary.mean_queue_waiting_time_seconds /=
        static_cast<double>(total_filled_orders);
  }
  return summary;
}

}  // namespace

std::vector<MarketStressScenario> make_market_stress_scenarios(
    const lob::stochastic::SyntheticMarketConfig& baseline,
    const std::vector<std::uint64_t>& seeds,
    const MarketStressConfig& stress) {
  if (seeds.empty()) {
    throw std::invalid_argument{"stress scenario seed list must not be empty"};
  }
  const std::array<double, 5> multipliers{
      stress.volatility_flow_multiplier,
      stress.one_sided_flow_multiplier,
      stress.hawkes_excitation_multiplier,
      stress.liquidity_remaining_fraction,
      stress.cancellation_multiplier};
  for (const double multiplier : multipliers) {
    if (!std::isfinite(multiplier) || multiplier <= 0.0) {
      throw std::invalid_argument{
          "market stress multipliers must be positive and finite"};
    }
  }
  if (!std::isfinite(stress.price_jump_size_multiplier) ||
      stress.price_jump_size_multiplier <= 1.0) {
    throw std::invalid_argument{
        "price-jump stress multiplier must be finite and greater than one"};
  }
  if (stress.volatility_flow_multiplier <= 1.0 ||
      stress.one_sided_flow_multiplier <= 1.0 ||
      stress.hawkes_excitation_multiplier <= 1.0 ||
      stress.liquidity_remaining_fraction > 1.0 ||
      stress.cancellation_multiplier <= 1.0) {
    throw std::invalid_argument{
        "stress severity must increase risk factors and keep remaining "
        "liquidity in (0, 1]"};
  }
  lob::stochastic::validate_stationary(baseline.hawkes);
  const std::array<std::pair<MarketStressKind, const char*>, 7> definitions{{
      {MarketStressKind::VolatilityIncrease,
       "sudden-volatility-increase (post-shock regime)"},
      {MarketStressKind::OneSidedOrderFlowBurst,
       "one-sided-order-flow-burst"},
      {MarketStressKind::HawkesExcitationIncrease,
       "hawkes-excitation-increase"},
      {MarketStressKind::LiquidityWithdrawal, "liquidity-withdrawal"},
      {MarketStressKind::CancellationSurge, "cancellation-surge"},
      {MarketStressKind::SpreadWidening,
       "sudden-spread-widening (post-shock regime)"},
      {MarketStressKind::PriceJumps, "price-jumps"},
  }};
  std::vector<MarketStressScenario> scenarios;
  scenarios.reserve(definitions.size());
  for (const auto& [kind, label] : definitions) {
    scenarios.push_back({.label = label,
                         .kind = kind,
                         .market = apply_stress(baseline, kind, stress),
                         .seeds = seeds});
  }
  return scenarios;
}

double adverse_selection_ticks(const lob::core::Side market_maker_side,
                               const double future_midprice_ticks,
                               const double execution_price_ticks) {
  if (!std::isfinite(future_midprice_ticks) ||
      !std::isfinite(execution_price_ticks)) {
    throw std::invalid_argument{"markout prices must be finite"};
  }
  return side_sign(market_maker_side) *
         (future_midprice_ticks - execution_price_ticks);
}

std::vector<MarketQualityScenarioSummary> run_market_quality_study(
    const std::vector<MarketQualityPolicy>& policies,
    const std::vector<MarketStressScenario>& scenarios,
    const MarketQualityConfig& config) {
  validate_config(config);
  if (policies.empty() || scenarios.empty()) {
    throw std::invalid_argument{
        "market-quality study needs policies and stress scenarios"};
  }
  for (const auto& policy : policies) {
    if (policy.name.empty() || !policy.make_agent) {
      throw std::invalid_argument{"market-quality policy is incomplete"};
    }
  }
  std::vector<MarketQualityScenarioSummary> summaries;
  summaries.reserve(policies.size() * scenarios.size());
  for (const auto& scenario : scenarios) {
    if (scenario.label.empty() || scenario.seeds.empty()) {
      throw std::invalid_argument{"market stress scenario is incomplete"};
    }
    for (const auto& policy : policies) {
      std::vector<MarketQualityEpisode> runs(scenario.seeds.size());
      detail::parallel_for(
          scenario.seeds.size(), config.worker_count,
          [&](const std::size_t index) {
            runs[index] = evaluate_episode(policy, scenario,
                                           scenario.seeds[index], config);
          });
      summaries.push_back(summarize_runs(
          policy.name, scenario.label, std::move(runs),
          config.expected_shortfall_confidence));
    }
  }
  return summaries;
}

}  // namespace project3
