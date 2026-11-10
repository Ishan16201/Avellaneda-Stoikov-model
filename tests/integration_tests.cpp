#include "project3/exchange_adapter.hpp"
#include "project3/execution_layer.hpp"
#include "project3/avellaneda_stoikov.hpp"
#include "project3/benchmark.hpp"
#include "project3/differentiable_surrogate.hpp"
#include "project3/hawkes_policy.hpp"
#include "project3/market_quality.hpp"
#include "project3/policy_optimization.hpp"
#include "project3/reduced_state_solver.hpp"

#include "lob/stochastic/hawkes_market_adapter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error{message};
}

lob::stochastic::SyntheticMarketConfig market_config() {
  lob::stochastic::SyntheticMarketConfig market;
  market.seed = 42;
  market.horizon_seconds = 2.0;
  market.hawkes.baseline_events_per_second = {2.0, 1.0, 0.5, 0.5, 1.0, 1.0};
  market.hawkes.alpha_branching = {
      {0.05, 0.01, 0.01, 0.01, 0.01, 0.01},
      {0.01, 0.05, 0.01, 0.01, 0.01, 0.01},
      {0.02, 0.01, 0.03, 0.01, 0.01, 0.01},
      {0.01, 0.02, 0.01, 0.03, 0.01, 0.01},
      {0.01, 0.01, 0.01, 0.01, 0.05, 0.02},
      {0.01, 0.01, 0.01, 0.01, 0.02, 0.05},
  };
  market.hawkes.beta_per_second.assign(6, std::vector<double>(6, 3.0));
  market.marks.price_placement = {
      .reference_price_ticks = 1000, .min_offset_ticks = 0, .max_offset_ticks = 2};
  market.marks.limit_size = {.min_lots = 1, .max_lots = 3};
  market.marks.market_size = {.min_lots = 1, .max_lots = 2};
  market.initial_book.best_bid_ticks = 999;
  market.initial_book.best_ask_ticks = 1001;
  market.initial_book.levels_per_side = 3;
  market.initial_book.orders_per_level = 2;
  market.initial_book.order_size = 5;
  return market;
}

template <typename T>
concept HasFutureEvents = requires(const T& value) { value.future_events(); };
template <typename T>
concept HasRandomGenerator = requires(const T& value) { value.random_generator(); };
static_assert(!HasFutureEvents<project3::MarketView>);
static_assert(!HasRandomGenerator<project3::MarketView>);

class RecorderAgent final : public project3::TradingAgent {
 public:
  [[nodiscard]] std::string name() const override { return "Recorder"; }
  void on_start(const project3::MarketView& view, project3::OrderGateway&) override {
    snapshots.push_back(view);
  }
  void on_market_event(const project3::MarketView& view,
                       project3::OrderGateway&) override {
    snapshots.push_back(view);
  }
  void on_trade(const lob::strategy::StrategyMarketTrade& trade,
                const project3::MarketView&) override {
    trades.push_back(trade);
  }

  std::vector<project3::MarketView> snapshots;
  std::vector<lob::strategy::StrategyMarketTrade> trades;
};

class PartialFillAgent final : public project3::TradingAgent {
 public:
  [[nodiscard]] std::string name() const override { return "PartialFillAgent"; }
  void on_start(const project3::MarketView&, project3::OrderGateway& gateway) override {
    order_id = gateway.submit_limit(lob::core::Side::Sell, 100, 3);
  }
  void on_execution(const project3::ExecutionNotification& report,
                    const project3::MarketView& view,
                    project3::OrderGateway&) override {
    if (report.order_id == order_id && report.filled_quantity > 0) {
      reports.push_back(report);
      state = view.market_maker;
    }
  }

  lob::core::OrderId order_id{};
  std::vector<project3::ExecutionNotification> reports;
  project3::MarketMakerState state;
};

class CancelAgent final : public project3::TradingAgent {
 public:
  [[nodiscard]] std::string name() const override { return "CancelAgent"; }
  void on_start(const project3::MarketView&, project3::OrderGateway& gateway) override {
    order_id = gateway.submit_limit(lob::core::Side::Buy, 50, 1);
  }
  void on_timer(const project3::MarketView&, project3::OrderGateway& gateway) override {
    if (!cancelled) {
      gateway.cancel(order_id);
      cancelled = true;
    }
  }

  lob::core::OrderId order_id{};
  bool cancelled{};
};

class TimerRecorder final : public project3::TradingAgent {
 public:
  [[nodiscard]] std::string name() const override { return "TimerRecorder"; }
  void on_timer(const project3::MarketView& view,
                project3::OrderGateway&) override {
    times.push_back(view.timestamp_ns);
    intensities.push_back(view.hawkes_intensities);
  }

  std::vector<lob::core::TimestampNs> times;
  std::vector<std::vector<double>> intensities;
};

class RoundTripAgent final : public project3::TradingAgent {
 public:
  [[nodiscard]] std::string name() const override { return "RoundTripAgent"; }
  void on_start(const project3::MarketView&, project3::OrderGateway& gateway) override {
    static_cast<void>(gateway.submit_market(lob::core::Side::Buy, 1));
  }
  void on_timer(const project3::MarketView&, project3::OrderGateway& gateway) override {
    if (!sold_) {
      sold_ = true;
      static_cast<void>(gateway.submit_market(lob::core::Side::Sell, 1));
    }
  }
  void on_stop(const project3::MarketView& view, project3::OrderGateway&) override {
    final_state = view.market_maker;
  }

  bool sold_{};
  project3::MarketMakerState final_state;
};

class MarketQualityQuoteAgent final : public project3::TradingAgent {
 public:
  [[nodiscard]] std::string name() const override {
    return "MarketQualityQuote";
  }
  void on_start(const project3::MarketView& view,
                project3::OrderGateway& gateway) override {
    if (view.best_bid) {
      static_cast<void>(
          gateway.submit_limit(lob::core::Side::Buy, *view.best_bid, 1));
    }
    if (view.best_ask) {
      static_cast<void>(
          gateway.submit_limit(lob::core::Side::Sell, *view.best_ask, 1));
    }
  }
};

class MarketQualityCancelAgent final : public project3::TradingAgent {
 public:
  [[nodiscard]] std::string name() const override {
    return "MarketQualityCancel";
  }
  void on_start(const project3::MarketView&, project3::OrderGateway& gateway) override {
    order_id_ = gateway.submit_limit(lob::core::Side::Buy, 100, 1);
  }
  void on_timer(const project3::MarketView&,
                project3::OrderGateway& gateway) override {
    if (!cancelled_) {
      gateway.cancel(order_id_);
      cancelled_ = true;
    }
  }

 private:
  lob::core::OrderId order_id_{};
  bool cancelled_{};
};

class MarketQualityAggressorAgent final : public project3::TradingAgent {
 public:
  explicit MarketQualityAggressorAgent(
      const lob::core::Side side,
      const lob::core::QuantityLots quantity = 1)
      : side_(side), quantity_(quantity) {}

  [[nodiscard]] std::string name() const override {
    return side_ == lob::core::Side::Buy ? "MarketQualityBuyer"
                                         : "MarketQualitySeller";
  }
  void on_start(const project3::MarketView&,
                project3::OrderGateway& gateway) override {
    static_cast<void>(gateway.submit_market(side_, quantity_));
  }

 private:
  lob::core::Side side_;
  lob::core::QuantityLots quantity_;
};

void verify_replay_preserves_reference_behavior() {
  auto config = market_config();
  RecorderAgent agent;
  const auto report = project3::ExchangeAdapter{}.run(agent, config);
  const auto reference = lob::stochastic::HawkesMarketAdapter{}.run(config);
  require(report.seed == config.seed, "seed missing from strategy report");
  require(agent.snapshots.size() == reference.book_snapshots.size(),
          "event-step count changed from reference: observed " +
              std::to_string(agent.snapshots.size()) + ", expected " +
              std::to_string(reference.book_snapshots.size()));

  for (std::size_t i = 0; i < reference.book_snapshots.size(); ++i) {
    const auto& expected = reference.book_snapshots[i];
    const auto& observed = agent.snapshots[i];
    require(observed.timestamp_ns == expected.timestamp_ns,
            "snapshot timestamp differs from reference");
    require(observed.bids.size() == expected.bids.size() &&
                observed.asks.size() == expected.asks.size(),
            "book depth differs from reference");
    const auto expected_bid = expected.bids.empty()
                                  ? std::optional<lob::core::PriceTicks>{}
                                  : std::optional<lob::core::PriceTicks>{
                                        expected.bids.front().price_ticks};
    const auto expected_ask = expected.asks.empty()
                                  ? std::optional<lob::core::PriceTicks>{}
                                  : std::optional<lob::core::PriceTicks>{
                                        expected.asks.front().price_ticks};
    require(observed.best_bid == expected_bid && observed.best_ask == expected_ask,
            "best bid or ask differs from book snapshot");
    if (expected_bid && expected_ask) {
      require(observed.spread_ticks == *expected_ask - *expected_bid,
              "spread differs from book snapshot");
      require(observed.midprice_ticks ==
                  (static_cast<double>(*expected_bid) +
                   static_cast<double>(*expected_ask)) * 0.5,
              "midprice differs from book snapshot");
    } else {
      require(!observed.spread_ticks && !observed.midprice_ticks,
              "midprice or spread exists for an incomplete book");
    }
    for (std::size_t j = 0; j < expected.bids.size(); ++j) {
      require(observed.bids[j] ==
                  project3::PriceLevel{expected.bids[j].price_ticks, expected.bids[j].quantity},
              "bid level differs from reference");
    }
    for (std::size_t j = 0; j < expected.asks.size(); ++j) {
      require(observed.asks[j] ==
                  project3::PriceLevel{expected.asks[j].price_ticks, expected.asks[j].quantity},
              "ask level differs from reference");
    }
    if (i > 0) {
      auto expected_intensity =
          reference.events[i - 1].pre_event_intensities;
      const auto event_index = lob::stochastic::event_type_index(
          reference.events[i - 1].type);
      for (std::size_t channel = 0; channel < expected_intensity.size(); ++channel) {
        expected_intensity[channel] +=
            config.hawkes.alpha_branching[channel][event_index] *
            config.hawkes.beta_per_second[channel][event_index];
      }
      require(observed.hawkes_intensities.size() == expected_intensity.size(),
              "causal Hawkes intensity dimension differs from reference");
      for (std::size_t j = 0; j < expected_intensity.size(); ++j) {
        require(std::abs(observed.hawkes_intensities[j] - expected_intensity[j]) <=
                    1e-12 * std::max(1.0, std::abs(expected_intensity[j])),
                "causal Hawkes intensity differs from reference");
      }
      require(observed.hawkes_flow.available &&
                  observed.hawkes_flow.market_buy ==
                      observed.hawkes_intensities[
                          lob::stochastic::event_type_index(
                              lob::stochastic::EventType::MarketBuy)] &&
                  observed.hawkes_flow.market_sell ==
                      observed.hawkes_intensities[
                          lob::stochastic::event_type_index(
                              lob::stochastic::EventType::MarketSell)],
              "named Hawkes market-order intensities differ from raw state");
      require(observed.timestamp_ns == static_cast<lob::core::TimestampNs>(
                  std::floor(reference.events[i - 1].time_seconds * 1.0e9)),
              "Hawkes event timestamp differs from reference");
    }
  }

  require(agent.trades.size() == reference.trades.size(),
          "matching trade count differs from reference");
  for (std::size_t i = 0; i < reference.trades.size(); ++i) {
    const auto& actual = agent.trades[i];
    const auto& expected = reference.trades[i];
    require(actual.timestamp_ns == expected.timestamp_ns &&
                actual.price_ticks == expected.price_ticks &&
                actual.quantity == expected.quantity &&
                actual.aggressor_side == expected.aggressor_side,
            "matching trade differs from reference");
  }
}

void verify_partial_fill_notifications() {
  lob::stochastic::SyntheticMarketConfig config;
  config.seed = 77;
  config.horizon_seconds = 0.05;
  config.hawkes.baseline_events_per_second = {0.001, 0.001, 0.001, 0.001, 2000, 0.001};
  config.hawkes.alpha_branching.assign(6, std::vector<double>(6, 0.0));
  config.hawkes.beta_per_second.assign(6, std::vector<double>(6, 1.0));
  config.marks.market_size = {.min_lots = 1, .max_lots = 1};
  config.initial_book.explicit_orders = {
      {1, lob::core::Side::Buy, 99, 20},
      {2, lob::core::Side::Sell, 101, 20},
  };

  PartialFillAgent agent;
  lob::strategy::StrategyEvaluatorConfig evaluation;
  evaluation.timer_interval_ns = 0;
  evaluation.broker_config.fees.maker_fee_per_lot = -0.25;
  evaluation.broker_config.fees.commission_per_order = 2.0;
  evaluation.broker_config.risk.max_order_lots = 100;
  evaluation.broker_config.risk.max_position_lots = 100;
  const auto report = project3::ExchangeAdapter{}.run(agent, config, evaluation);
  bool saw_partial = false;
  bool saw_complete = false;
  for (const auto& fill : agent.reports) {
    saw_partial = saw_partial ||
                  fill.status == lob::strategy::StrategyOrderStatus::PartiallyFilled;
    saw_complete = saw_complete ||
                   fill.status == lob::strategy::StrategyOrderStatus::Filled;
  }
  require(saw_partial, "partial fill notification was not delivered");
  require(saw_complete, "completion notification was not delivered");
  require(report.execution_history.size() >= 3,
          "fill reports were not retained in the evaluation report");
  lob::core::QuantityLots filled = 0;
  double expected_cash = 0.0;
  for (const auto& fill : agent.reports) {
    filled += fill.filled_quantity;
    const double sign = fill.side == lob::core::Side::Buy ? 1.0 : -1.0;
    expected_cash -= sign * static_cast<double>(fill.filled_quantity) *
                     static_cast<double>(fill.price_ticks);
    expected_cash -= fill.fee_paid_currency;
  }
  require(filled == 3, "partial fills do not conserve submitted order quantity");
  require(report.final_portfolio.position_lots == -3,
          "sell fills did not conserve inventory");
  require(std::abs(report.final_portfolio.cash_currency - expected_cash) < 1e-10,
          "cash does not reconcile to executions and fees");
  require(std::abs(report.final_portfolio.total_charged_fees_currency - 2.0) < 1e-10,
          "gross charged fees were not separated from rebates");
  require(std::abs(report.final_portfolio.total_rebates_currency - 0.75) < 1e-10,
          "maker rebates were not accumulated correctly");
  require(std::abs(report.final_portfolio.total_fees_currency - 1.25) < 1e-10,
          "net fees do not equal charges less rebates");
  require(report.to_json().find("\"gross_charged_fees\": 2.000000") !=
              std::string::npos &&
              report.to_json().find("\"rebates_received\": 0.750000") !=
                  std::string::npos,
          "serialized report omitted fee and rebate components");
  require(report.market_making.max_inventory_exposure == 3.0,
          "maximum inventory exposure was not tracked");
  require(agent.state.order_execution_history.size() == 3,
          "market-maker state did not retain order-level fill history");
  require(agent.state.current_inventory_exposure_lots == 3 &&
              agent.state.max_inventory_exposure_lots == 3,
          "market-maker state inventory exposure is inconsistent");
  require(std::abs(agent.state.account.cash_currency -
                   report.final_portfolio.cash_currency) < 1e-10,
          "market-maker state cash diverges from broker accounting");

  lob::core::QuantityLots report_fills = 0;
  std::size_t fill_reports = 0;
  for (const auto& execution : report.execution_history) {
    if (execution.client_order_id == agent.order_id &&
        execution.filled_quantity > 0) {
      report_fills += execution.filled_quantity;
      ++fill_reports;
    }
  }
  require(report_fills == 3 && fill_reports == 3,
          "execution history double-counted or omitted partial fills");
  require(std::abs(report.final_portfolio.marked_equity_currency -
                   (report.final_portfolio.cash_currency +
                    static_cast<double>(report.final_portfolio.position_lots) *
                        100.0)) < 1e-10,
          "mark-to-market equity does not equal cash plus inventory value");
  require(std::abs(report.final_portfolio.unrealized_pnl_currency -
                   static_cast<double>(report.final_portfolio.position_lots) *
                       (100.0 - report.final_portfolio.average_entry_price_ticks)) <
              1e-10,
          "unrealized P&L is inconsistent with the final mark");
}

void verify_round_trip_realized_pnl_and_equity_reconciliation() {
  lob::stochastic::SyntheticMarketConfig config;
  config.seed = 91;
  config.horizon_seconds = 20e-9;
  config.hawkes.baseline_events_per_second.assign(6, 0.001);
  config.hawkes.alpha_branching.assign(6, std::vector<double>(6, 0.0));
  config.hawkes.beta_per_second.assign(6, std::vector<double>(6, 1.0));
  config.initial_book.explicit_orders = {
      {1, lob::core::Side::Buy, 99, 10},
      {2, lob::core::Side::Sell, 101, 10},
  };
  lob::strategy::StrategyEvaluatorConfig evaluation;
  evaluation.timer_interval_ns = 10;

  RoundTripAgent agent;
  const auto report = project3::ExchangeAdapter{}.run(agent, config, evaluation);
  require(report.final_portfolio.position_lots == 0,
          "round trip did not restore inventory to zero");
  require(std::abs(report.final_portfolio.cash_currency + 2.0) < 1e-10,
          "round-trip cash does not equal signed execution cashflows");
  require(std::abs(report.final_portfolio.gross_realized_pnl_currency + 2.0) < 1e-10,
          "realized P&L did not use the average entry price");
  require(std::abs(report.final_portfolio.unrealized_pnl_currency) < 1e-10,
          "flat inventory should have zero unrealized P&L");
  require(std::abs(report.final_portfolio.marked_equity_currency + 2.0) < 1e-10,
          "flat-account equity does not equal cash");
  require(std::abs(report.final_portfolio.net_total_pnl_currency() + 2.0) < 1e-10,
          "net P&L does not reconcile with marked equity");
  require(std::abs(agent.final_state.account.gross_realized_pnl_currency + 2.0) <
              1e-10,
          "market-maker state omitted realized P&L");
}

project3::MarketView queue_view(const lob::core::QuantityLots bid_depth) {
  project3::MarketView view;
  view.timestamp_ns = 0;
  if (bid_depth > 0) view.bids.push_back({99, bid_depth});
  view.best_bid = bid_depth > 0 ? std::optional<lob::core::PriceTicks>{99}
                                : std::nullopt;
  return view;
}

void verify_queue_depletion_cancellation_and_replacement() {
  project3::QueueTracker tracker;
  tracker.observe_book(queue_view(10));
  tracker.add_order(40, lob::core::Side::Buy, 99, 4, queue_view(10));
  tracker.add_order(39, lob::core::Side::Buy, 99, 2, queue_view(10));
  auto later_position = tracker.position(39);
  require(later_position && later_position->quantity_ahead == 14,
          "later same-price order did not join behind the earlier own order");

  tracker.observe_book(queue_view(18));
  auto position = tracker.position(40);
  require(position && position->quantity_ahead == 10 &&
              position->quantity_behind == 4,
          "same-price own order and external additions were not placed behind");

  tracker.observe_book(queue_view(14));
  position = tracker.position(40);
  require(position && position->quantity_ahead == 10 &&
              position->quantity_behind == 0,
          "conservative cancellation model did not remove behind quantity first");

  tracker.observe_trade(lob::strategy::StrategyMarketTrade{
      .timestamp_ns = 1,
      .price_ticks = 99,
      .quantity = 7,
      .aggressor_side = lob::core::Side::Sell,
  });
  position = tracker.position(40);
  require(position && position->quantity_ahead == 3 &&
              position->own_remaining == 4,
          "public trade did not deplete queue-ahead quantity");
  later_position = tracker.position(39);
  require(later_position && later_position->quantity_ahead == 5,
          "trade depletion did not preserve relative own-order priority");

  tracker.synchronize_book(queue_view(7));
  tracker.apply_fill(project3::ExecutionNotification{
      .order_id = 40,
      .timestamp_ns = 2,
      .side = lob::core::Side::Buy,
      .price_ticks = 99,
      .filled_quantity = 2,
      .remaining_quantity = 2,
      .status = lob::strategy::StrategyOrderStatus::PartiallyFilled,
      .passive = true,
  });
  tracker.observe_trade(lob::strategy::StrategyMarketTrade{
      .timestamp_ns = 2,
      .price_ticks = 99,
      .quantity = 5,
      .aggressor_side = lob::core::Side::Sell,
  });
  position = tracker.position(40);
  require(position && position->quantity_ahead == 0 &&
              position->own_remaining == 2,
          "partial fill was double-counted as queue depletion");
  later_position = tracker.position(39);
  require(later_position && later_position->quantity_ahead == 0,
          "partial fill did not deplete quantity ahead of the later own order");

  tracker.cancel_order(39);
  tracker.replace_order(40, 41, lob::core::Side::Buy, 99, 3, queue_view(5));
  require(!tracker.position(40), "replacement preserved the old order ID");
  position = tracker.position(41);
  require(position && position->quantity_ahead == 5 &&
              position->own_remaining == 3,
          "replacement did not receive a fresh queue position");

  project3::QueueTracker ahead_first({
      .cancellation_allocation = project3::CancellationAllocation::AheadFirst});
  ahead_first.observe_book(queue_view(10));
  ahead_first.add_order(50, lob::core::Side::Buy, 99, 1, queue_view(10));
  ahead_first.observe_book(queue_view(6));
  const auto ahead_position = ahead_first.position(50);
  require(ahead_position && ahead_position->quantity_ahead == 6,
          "ahead-first cancellation assumption was not applied");
}

void verify_cancel_gateway() {
  auto config = market_config();
  config.horizon_seconds = 0.25;
  lob::strategy::StrategyEvaluatorConfig evaluation;
  evaluation.timer_interval_ns = 100'000'000;
  CancelAgent agent;
  const auto report = project3::ExchangeAdapter{}.run(agent, config, evaluation);
  require(agent.cancelled, "agent timer did not invoke cancellation");
  for (const auto& order : report.final_open_orders) {
    require(order.client_order_id != agent.order_id,
            "cancelled order remains open");
  }
}

void verify_seeded_runs_are_reproducible() {
  const auto config = market_config();
  RecorderAgent first_agent;
  RecorderAgent second_agent;
  const auto first = project3::ExchangeAdapter{}.run(first_agent, config);
  const auto second = project3::ExchangeAdapter{}.run(second_agent, config);
  require(first_agent.snapshots.size() == second_agent.snapshots.size(),
          "same seed produced different event counts");
  require(first_agent.trades.size() == second_agent.trades.size(),
          "same seed produced different trade counts");
  for (std::size_t i = 0; i < first_agent.snapshots.size(); ++i) {
    require(first_agent.snapshots[i].timestamp_ns ==
                second_agent.snapshots[i].timestamp_ns &&
                first_agent.snapshots[i].bids == second_agent.snapshots[i].bids &&
                first_agent.snapshots[i].asks == second_agent.snapshots[i].asks &&
                first_agent.snapshots[i].hawkes_intensities ==
                    second_agent.snapshots[i].hawkes_intensities,
            "same seed produced a different market observation");
  }
  require(first.market_making.total_trades == second.market_making.total_trades,
          "same seed produced a different evaluation report");
}

void verify_timer_intensities_use_only_observed_events() {
  auto config = market_config();
  config.horizon_seconds = 0.4;
  const auto reference = lob::stochastic::HawkesMarketAdapter{}.run(config);
  TimerRecorder agent;
  lob::strategy::StrategyEvaluatorConfig evaluation;
  evaluation.timer_interval_ns = 100'000'000;
  static_cast<void>(project3::ExchangeAdapter{}.run(agent, config, evaluation));

  require(!agent.times.empty(), "timer callbacks were not delivered");
  for (std::size_t tick = 0; tick < agent.times.size(); ++tick) {
    const auto timer_seconds = static_cast<double>(agent.times[tick]) * 1.0e-9;
    auto expected = config.hawkes.baseline_events_per_second;
    for (const auto& event : reference.events) {
      const auto event_time_ns = static_cast<lob::core::TimestampNs>(
          std::floor(event.time_seconds * 1.0e9));
      if (event_time_ns >= agent.times[tick]) continue;
      const auto parent = lob::stochastic::event_type_index(event.type);
      const double elapsed = timer_seconds - event.time_seconds;
      for (std::size_t child = 0; child < expected.size(); ++child) {
        const double beta = config.hawkes.beta_per_second[child][parent];
        expected[child] += config.hawkes.alpha_branching[child][parent] * beta *
                           std::exp(-beta * elapsed);
      }
    }
    require(agent.intensities[tick].size() == expected.size(),
            "timer intensity vector has the wrong dimension");
    for (std::size_t i = 0; i < expected.size(); ++i) {
      require(std::abs(agent.intensities[tick][i] - expected[i]) < 1e-10,
              "timer intensity mismatch at tick " + std::to_string(tick) +
                  ", channel " + std::to_string(i) + ": got " +
                  std::to_string(agent.intensities[tick][i]) + ", expected " +
                  std::to_string(expected[i]));
    }
  }
}

void verify_avellaneda_stoikov_quote_limits() {
  project3::AvellanedaStoikovConfig config;
  config.volatility_ticks_per_sqrt_second = 2.0;
  config.risk_aversion_per_tick_lot = 0.5;
  config.arrival_rate_per_second = 3.0;
  config.intensity_decay_per_tick = 2.0;
  config.horizon_seconds = 10.0;

  const auto flat = project3::avellaneda_stoikov_quotes(100.0, 0, 4.0, config);
  const double c = std::log1p(0.5 / 2.0) / 0.5;
  const double risk = 0.5 * 2.0 * 2.0 * 6.0;
  require(std::abs(flat.reservation_price_ticks - 100.0) < 1e-12,
          "zero inventory shifted the reservation price");
  require(std::abs(flat.bid_offset_ticks - (c + risk / 2.0)) < 1e-12 &&
              std::abs(flat.ask_offset_ticks - (c + risk / 2.0)) < 1e-12,
          "zero-inventory quotes are not symmetric");

  const auto long_inventory =
      project3::avellaneda_stoikov_quotes(100.0, 2, 4.0, config);
  require(std::abs(long_inventory.reservation_price_ticks - (100.0 - 2.0 * risk)) <
              1e-12,
          "reservation price does not reflect inventory risk");
  require(long_inventory.bid_offset_ticks > long_inventory.ask_offset_ticks,
          "long inventory did not skew quotes downward");
  require(long_inventory.bid_intensity_per_second > 0.0 &&
              long_inventory.ask_intensity_per_second > 0.0,
          "quote intensities must be positive");
  require(std::abs(long_inventory.bid_intensity_per_second -
                   3.0 * std::exp(-2.0 *
                                  (100.0 - long_inventory.bid_price_ticks))) <
              1e-12,
          "bid intensity does not use A exp(-k delta) at the rounded quote");

  config.risk_aversion_per_tick_lot = 0.0;
  const auto risk_neutral =
      project3::avellaneda_stoikov_quotes(100.0, 7, 4.0, config);
  require(risk_neutral.reservation_price_ticks == 100.0 &&
              std::abs(risk_neutral.bid_offset_ticks - 0.5) < 1e-12 &&
              std::abs(risk_neutral.ask_offset_ticks - 0.5) < 1e-12,
          "zero risk aversion did not converge to the 1/k symmetric quotes");
  config.risk_aversion_per_tick_lot = 1.0e-15;
  const auto near_risk_neutral =
      project3::avellaneda_stoikov_quotes(100.0, 0, 4.0, config);
  require(std::abs(near_risk_neutral.bid_offset_ticks - 0.5) < 1e-12 &&
              std::abs(near_risk_neutral.ask_offset_ticks - 0.5) < 1e-12,
          "small positive risk aversion was numerically unstable");

  config.risk_aversion_per_tick_lot = 0.5;
  const auto terminal =
      project3::avellaneda_stoikov_quotes(100.0, 2, 10.0, config);
  require(terminal.reservation_price_ticks == 100.0 &&
              std::abs(terminal.bid_offset_ticks - c) < 1e-12 &&
              std::abs(terminal.ask_offset_ticks - c) < 1e-12,
          "terminal-time risk adjustment did not vanish");
}

void verify_avellaneda_stoikov_baseline_evaluation() {
  auto market = market_config();
  market.horizon_seconds = 0.4;
  project3::AvellanedaStoikovConfig policy;
  policy.volatility_ticks_per_sqrt_second = 0.5;
  policy.risk_aversion_per_tick_lot = 0.05;
  policy.arrival_rate_per_second = 1.5;
  policy.intensity_decay_per_tick = 0.5;
  policy.horizon_seconds = market.horizon_seconds;
  policy.inventory_limit_lots = 3;
  policy.base_quote_size_lots = 1;
  policy.max_quote_size_lots = 2;
  lob::strategy::StrategyEvaluatorConfig evaluation;
  evaluation.timer_interval_ns = 100'000'000;

  const auto result =
      project3::evaluate_avellaneda_stoikov(policy, market, evaluation);
  require(result.submitted_orders > 0,
          "Avellaneda-Stoikov agent did not submit any quotes");
  require(result.fill_rate >= 0.0 && result.fill_rate <= 1.0,
          "quote fill rate is outside [0, 1]");
  require(result.inventory_distribution.size() > 0,
          "inventory distribution was not collected");
  require(result.bid_intensity.active_order_seconds > 0.0 ||
              result.ask_intensity.active_order_seconds > 0.0,
          "quote exposure was not measured");
  require(std::isfinite(result.bid_intensity.theoretical_per_order_second) &&
              std::isfinite(result.ask_intensity.empirical_per_order_second) &&
              std::isfinite(result.sharpe_ratio),
          "baseline report contains non-finite statistics");
  require(std::abs(result.total_pnl_currency -
                   (result.spread_capture_currency +
                    result.inventory_pnl_currency - result.fees_currency)) <
              1e-8,
          "P&L attribution does not reconcile");
  require(std::abs(result.final_account.position_lots) <=
              static_cast<std::int64_t>(policy.inventory_limit_lots),
          "inventory exceeded the configured limit");
  require(result.to_formatted_report().find("theory/emp") != std::string::npos,
          "baseline report omitted the theoretical/empirical intensity comparison");
  require(result.to_formatted_report().find("Inventory distribution") !=
              std::string::npos,
          "baseline report omitted the inventory distribution");
}

void verify_hawkes_state_and_out_of_sample_policy_comparison() {
  project3::HawkesFlowState flow{
      .limit_buy = 2.0,
      .limit_sell = 1.0,
      .cancel_buy = 0.5,
      .cancel_sell = 0.5,
      .market_buy = 4.0,
      .market_sell = 1.0,
      .available = true,
  };
  require(flow.intensity(lob::stochastic::EventType::MarketBuy) == 4.0 &&
              flow.intensity(lob::stochastic::EventType::MarketSell) == 1.0 &&
              flow.intensity(lob::stochastic::EventType::CancelBuy) == 0.5 &&
              flow.intensity(lob::stochastic::EventType::LimitSell) == 1.0,
          "named Hawkes intensities do not map to the simulator event taxonomy");
  require(flow.order_flow_imbalance() > 0.0 &&
              flow.order_flow_imbalance() <= 1.0,
          "directional order-flow imbalance has invalid sign or scale");

  std::vector<lob::stochastic::SyntheticMarketConfig> training;
  for (std::uint64_t seed : {101U, 102U, 103U}) {
    auto run = market_config();
    run.seed = seed;
    run.horizon_seconds = 1.5;
    training.push_back(run);
  }
  project3::HawkesDirectionFitConfig fit;
  fit.forecast_horizon_seconds = 0.15;
  fit.training_epochs = 120;
  const auto model = project3::HawkesDirectionModel::train(training, fit);
  require(model.training_observations() > 3 &&
              std::isfinite(model.training_log_loss()),
          "Hawkes directional model did not fit usable training data");
  const auto prediction = model.predict(flow);
  const double probability_sum = prediction.probability_up +
                                prediction.probability_down +
                                prediction.probability_unchanged;
  require(std::abs(probability_sum - 1.0) < 1e-12 &&
              prediction.probability_up >= 0.0 &&
              prediction.probability_down >= 0.0 &&
              prediction.mean_up_move_ticks >= 0.0 &&
              prediction.mean_down_move_ticks >= 0.0,
          "Hawkes price-direction prediction is not a valid probability model");

  auto unseen_market = market_config();
  unseen_market.horizon_seconds = 1.5;
  unseen_market.hawkes.baseline_events_per_second[4] *= 1.7;
  unseen_market.hawkes.baseline_events_per_second[5] *= 0.7;
  unseen_market.hawkes.alpha_branching[4][4] *= 0.5;
  project3::PolicyComparisonScenario scenario{
      .label = "held-out-Hawkes-parameters",
      .market = unseen_market,
      .evaluation_seeds = {201, 202},
  };
  project3::AvellanedaStoikovConfig as;
  as.horizon_seconds = unseen_market.horizon_seconds;
  as.volatility_ticks_per_sqrt_second = 0.5;
  as.inventory_limit_lots = 4;
  as.max_quote_size_lots = 2;
  project3::HawkesAwareConfig hawkes{
      .mean_market_order_size_lots = 1.5,
  };
  project3::SymmetricMarketMakerConfig symmetric{
      .quote_offset_ticks = 1,
      .quote_size_lots = 1,
      .inventory_limit_lots = 4,
  };
  lob::strategy::StrategyEvaluatorConfig evaluation;
  evaluation.timer_interval_ns = 100'000'000;
  const auto comparison = project3::compare_market_making_policies(
      training, {scenario}, fit, as, hawkes, symmetric, evaluation);
  require(comparison.size() == 1 &&
              comparison[0].classical_as.size() == 2 &&
              comparison[0].hawkes_aware.size() == 2 &&
              comparison[0].symmetric.size() == 2,
          "three-policy evaluation did not use all independent seeds");
  require(comparison[0].classical_summary.runs == 2 &&
              comparison[0].hawkes_summary.runs == 2 &&
              comparison[0].symmetric_summary.runs == 2,
          "policy comparison aggregates have an incorrect run count");
  require(comparison[0].direction_observations > 0 &&
              std::isfinite(comparison[0].direction_log_loss) &&
              std::isfinite(comparison[0].direction_base_rate_log_loss) &&
              comparison[0].direction_accuracy >= 0.0 &&
              comparison[0].direction_accuracy <= 1.0 &&
              comparison[0].direction_base_rate_accuracy >= 0.0 &&
              comparison[0].direction_base_rate_accuracy <= 1.0,
          "unseen-parameter directional forecast was not evaluated");
  require(comparison[0].classical_as[0].seed == 201 &&
              comparison[0].hawkes_aware[0].seed == 201 &&
              comparison[0].symmetric[0].seed == 201,
          "policy runs were not paired on the same evaluation seed");

  auto overlapping_scenario = scenario;
  overlapping_scenario.label = "invalid-overlapping-parameters";
  overlapping_scenario.market.hawkes = training.front().hawkes;
  bool rejected_overlap = false;
  try {
    static_cast<void>(project3::compare_market_making_policies(
        training, {overlapping_scenario}, fit, as, hawkes, symmetric,
        evaluation));
  } catch (const std::invalid_argument&) {
    rejected_overlap = true;
  }
  require(rejected_overlap,
          "comparison accepted Hawkes parameters seen during training");
}

void verify_reduced_state_dynamic_programming() {
  project3::ReducedStateConfig config;
  config.horizon_seconds = 1.0;
  config.time_steps = 500;
  config.inventory_limit_lots = 4;
  config.arrival_rate_per_second = 1.5;
  config.intensity_decay_per_tick = 1.0;
  config.min_quote_offset_ticks = 0.0;
  config.max_quote_offset_ticks = 3.0;
  config.quote_offset_step_ticks = 0.1;
  const auto solution = project3::ReducedStateSolver::solve(config);
  const auto flat_policy = solution.policy(0.0, 0.0);
  require(std::abs(flat_policy.bid_offset_ticks - 1.0) < 1e-12 &&
              std::abs(flat_policy.ask_offset_ticks - 1.0) < 1e-12,
          "risk-neutral DP did not match the analytical zero-risk A-S offset");
  const double analytical_value =
      2.0 * config.horizon_seconds * config.arrival_rate_per_second /
      std::exp(1.0);
  require(std::abs(solution.value(0.0, 0.0) - analytical_value) < 0.01,
          "risk-neutral DP value did not converge to analytical quote revenue");
  require(std::abs(solution.value(config.horizon_seconds, 2.0) + 0.0) <
              1e-12,
          "zero terminal penalty boundary condition is incorrect");
  require(std::abs(solution.value(0.0, 0.5) -
                   0.5 * (solution.value(0.0, 0.0) +
                          solution.value(0.0, 1.0))) <
              1e-12,
          "inventory interpolation is not linear between grid points");

  auto unstable = config;
  unstable.time_steps = 1;
  bool rejected_unstable_step = false;
  try {
    static_cast<void>(project3::ReducedStateSolver::solve(unstable));
  } catch (const std::invalid_argument&) {
    rejected_unstable_step = true;
  }
  require(rejected_unstable_step,
          "solver did not reject a time step violating the monotone CFL condition");

  config.terminal_inventory_penalty_currency_per_lot_squared = 1.0;
  const auto penalized = project3::ReducedStateSolver::solve(config);
  const auto long_policy = penalized.policy(0.0, 1.0);
  require(long_policy.bid_offset_ticks > long_policy.ask_offset_ticks,
          "terminal inventory penalty did not skew quotes to reduce long inventory");
  const auto upper_boundary =
      penalized.policy(0.0, static_cast<double>(config.inventory_limit_lots));
  require(!upper_boundary.bid_enabled && upper_boundary.ask_enabled,
          "inventory boundary did not disable further inventory-increasing fills");
  require(std::abs(penalized.value(config.horizon_seconds, 2.0) + 4.0) <
              1e-12,
          "quadratic terminal penalty boundary condition is incorrect");
  auto stand_aside_config = config;
  stand_aside_config.terminal_inventory_penalty_currency_per_lot_squared =
      1000.0;
  const auto stand_aside =
      project3::ReducedStateSolver::solve(stand_aside_config).policy(0.0, 1.0);
  require(!stand_aside.bid_enabled,
          "solver did not stand aside when every bid action had negative value");

  config.include_hawkes_activity_state = true;
  config.hawkes_baseline_activity_per_second = 1.0;
  config.hawkes_activity_max_per_second = 2.0;
  config.hawkes_activity_points = 11;
  config.hawkes_mean_reversion_per_second = 0.5;
  config.hawkes_jump_size_per_event = 0.2;
  const auto flow_solution = project3::ReducedStateSolver::solve(config);
  const auto low_activity_value = flow_solution.value(0.0, 0.0, 0.2);
  const auto high_activity_value = flow_solution.value(0.0, 0.0, 1.8);
  require(flow_solution.activity_grid_points() == 11 &&
              std::isfinite(low_activity_value) &&
              std::isfinite(high_activity_value) &&
              high_activity_value > low_activity_value,
          "Hawkes activity state did not affect the reduced value function");
  require(flow_solution.estimated_memory_bytes() > solution.estimated_memory_bytes(),
          "Hawkes state memory cost was not reflected in diagnostics");

  auto coarse = config;
  coarse.include_hawkes_activity_state = false;
  coarse.terminal_inventory_penalty_currency_per_lot_squared = 0.0;
  coarse.time_steps = 250;
  coarse.quote_offset_step_ticks = 0.4;
  auto fine = coarse;
  fine.time_steps = 500;
  fine.quote_offset_step_ticks = 0.2;
  fine.inventory_limit_lots = 6;
  auto finest = fine;
  finest.time_steps = 1000;
  finest.quote_offset_step_ticks = 0.1;
  finest.inventory_limit_lots = 8;
  const auto convergence =
      project3::run_grid_convergence_study({coarse, fine, finest}, true);
  require(convergence.size() == 3 &&
              convergence[0].analytical_zero_risk_offset_error_ticks >
                  convergence[1].analytical_zero_risk_offset_error_ticks &&
              convergence[1].analytical_zero_risk_offset_error_ticks <= 0.01 &&
              convergence[2].analytical_zero_risk_offset_error_ticks <=
                  convergence[1].analytical_zero_risk_offset_error_ticks &&
              convergence[1].value_change_from_previous_currency <
                  convergence[0].initial_value_currency &&
              convergence[1].policy_change_from_previous_ticks > 0.0 &&
              convergence[2].policy_change_from_previous_ticks <=
                  convergence[1].policy_change_from_previous_ticks &&
              convergence[2].estimated_memory_bytes >
                  convergence[1].estimated_memory_bytes &&
              convergence[2].runtime_milliseconds >= 0.0 &&
              convergence[2].boundary_value_effect_currency >= 0.0,
          "grid convergence diagnostics are incomplete or inconsistent");
}

void verify_differentiable_market_making_surrogate() {
  auto paths =
      project3::make_surrogate_training_batch(9341, 24, 12);
  auto repeated_paths =
      project3::make_surrogate_training_batch(9341, 24, 12);
  const auto enrich = [](std::vector<project3::SurrogatePath>& samples) {
    for (std::size_t p = 0; p < samples.size(); ++p) {
      for (std::size_t s = 0; s < samples[p].steps.size(); ++s) {
        auto& step = samples[p].steps[s];
        step.midprice_feature = 0.02 * static_cast<double>(s % 5);
        step.spread_ticks = 1.0 + static_cast<double>(s % 3);
        step.bid_depth_lots = 2.0 + static_cast<double>(p % 7);
        step.ask_depth_lots = 3.0 + static_cast<double>(s % 9);
        step.volatility_ticks_per_sqrt_second =
            0.1 + 0.05 * static_cast<double>(p % 4);
        step.time_remaining_fraction =
            1.0 - static_cast<double>(s) /
                      static_cast<double>(samples[p].steps.size());
        for (std::size_t i = 0; i < step.hawkes_intensities.size(); ++i) {
          step.hawkes_intensities[i] =
              0.1 * static_cast<double>(1 + i + p + s);
        }
      }
    }
  };
  enrich(paths);
  enrich(repeated_paths);
  require(paths.size() == repeated_paths.size() &&
              paths.front().steps.front().market_buy_intensity_per_second ==
                  repeated_paths.front().steps.front()
                      .market_buy_intensity_per_second &&
              paths.back().steps.back().midprice_return_ticks ==
                  repeated_paths.back().steps.back().midprice_return_ticks,
          "surrogate training batch is not reproducible for a fixed seed");

  for (const double scale : {0.1, 1.0, 5.0}) {
    project3::SurrogateParameters parameters;
    for (std::size_t i = 0; i < parameters.values.size(); ++i) {
      const auto centered = static_cast<int>(i % 9) - 4;
      parameters.values[i] =
          0.1 * scale * static_cast<double>(centered);
    }
    const auto check = project3::check_surrogate_gradient(
        parameters, paths, {}, 1.0e-5, 1.0e-12);
    require(check.maximum_relative_error < 2.0e-5,
            "forward-mode surrogate gradients disagree with finite differences");
  }

  project3::SurrogateParameters initial{
      .values = {-0.3, -0.5, 0.15, -0.05, 0.1, 0.2}};
  const auto finite_gradient =
      project3::finite_difference_surrogate_gradient(initial, paths, {}, 1.0e-5);
  project3::SurrogateOptimizerConfig one_step;
  one_step.optimizer = project3::SurrogateOptimizer::GradientDescent;
  one_step.learning_rate_schedule =
      project3::LearningRateSchedule::Constant;
  one_step.iterations = 1;
  one_step.learning_rate = 0.005;
  one_step.gradient_clip_norm = 100.0;
  const auto autodiff_step = project3::optimize_differentiable_surrogate(
      initial, paths, {}, one_step);
  for (std::size_t i = 0; i < project3::kSurrogateParameterCount; ++i) {
    const double finite_difference_parameter =
        initial.values[i] + one_step.learning_rate * finite_gradient[i];
    require(std::abs(autodiff_step.parameters.values[i] -
                     finite_difference_parameter) < 1.0e-7,
            "gradient-descent policy update differs from finite-difference update");
  }
  const auto raw_gradient =
      project3::differentiable_surrogate_gradient(initial, paths).gradient;
  double raw_norm = 0.0;
  for (const double component : raw_gradient) {
    raw_norm = std::hypot(raw_norm, component);
  }
  auto clipped_config = one_step;
  clipped_config.gradient_clip_norm = raw_norm * 0.5;
  const auto clipped = project3::optimize_differentiable_surrogate(
      initial, paths, {}, clipped_config);
  double clipped_update_norm = 0.0;
  for (std::size_t i = 0; i < project3::kSurrogateParameterCount; ++i) {
    clipped_update_norm =
        std::hypot(clipped_update_norm,
                   clipped.parameters.values[i] - initial.values[i]);
  }
  require(raw_norm > clipped_config.gradient_clip_norm &&
              std::abs(clipped_update_norm -
                       clipped_config.learning_rate *
                           clipped_config.gradient_clip_norm) <
                  1.0e-9,
          "global gradient clipping did not enforce its configured norm");

  auto iterative_config = one_step;
  iterative_config.iterations = 20;
  iterative_config.learning_rate = 0.01;
  iterative_config.gradient_clip_norm = 1.0;
  const auto optimized = project3::optimize_differentiable_surrogate(
      initial, paths, {}, iterative_config);
  const auto repeated_optimization =
      project3::optimize_differentiable_surrogate(
          initial, repeated_paths, {}, iterative_config);
  require(optimized.final_reward_currency >
              optimized.initial_reward_currency &&
              optimized.final_reward_currency ==
                  repeated_optimization.final_reward_currency &&
              optimized.parameters.values ==
                  repeated_optimization.parameters.values &&
              optimized.seed == iterative_config.seed,
          "surrogate optimization failed to improve or reproduce its result");
  auto finite_difference_policy = initial;
  for (std::size_t iteration = 0;
       iteration < iterative_config.iterations; ++iteration) {
    auto gradient = project3::finite_difference_surrogate_gradient(
        finite_difference_policy, paths, {}, 1.0e-5);
    double gradient_norm_squared = 0.0;
    for (const double component : gradient) {
      gradient_norm_squared += component * component;
    }
    const double gradient_norm = std::sqrt(gradient_norm_squared);
    if (gradient_norm > iterative_config.gradient_clip_norm) {
      const double scale = iterative_config.gradient_clip_norm / gradient_norm;
      for (double& component : gradient) component *= scale;
    }
    for (std::size_t i = 0; i < project3::kSurrogateParameterCount; ++i) {
      finite_difference_policy.values[i] +=
          iterative_config.learning_rate * gradient[i];
    }
  }
  for (std::size_t i = 0; i < project3::kSurrogateParameterCount; ++i) {
    require(std::abs(optimized.parameters.values[i] -
                     finite_difference_policy.values[i]) < 2.0e-6,
            "optimized policy diverged from finite-difference gradient ascent");
  }
  require(optimized.trace.size() == iterative_config.iterations &&
              std::isfinite(optimized.trace.front().gradient_norm) &&
              optimized.trace.back().learning_rate ==
                  iterative_config.learning_rate,
          "optimizer trace or constant learning-rate behavior is invalid");

  auto adam_config = iterative_config;
  adam_config.optimizer = project3::SurrogateOptimizer::Adam;
  adam_config.learning_rate_schedule =
      project3::LearningRateSchedule::CosineDecay;
  adam_config.minimum_learning_rate = 0.001;
  const auto adam = project3::optimize_differentiable_surrogate(
      initial, paths, {}, adam_config);
  require(adam.final_reward_currency > adam.initial_reward_currency &&
              adam.trace.back().learning_rate ==
                  adam_config.minimum_learning_rate,
          "Adam or cosine learning-rate schedule did not execute correctly");
  auto linear_config = iterative_config;
  linear_config.iterations = 3;
  linear_config.learning_rate = 0.03;
  linear_config.minimum_learning_rate = 0.01;
  linear_config.learning_rate_schedule =
      project3::LearningRateSchedule::LinearDecay;
  const auto linear = project3::optimize_differentiable_surrogate(
      initial, paths, {}, linear_config);
  require(std::abs(linear.trace.front().learning_rate - 0.03) < 1.0e-15 &&
              std::abs(linear.trace[1].learning_rate - 0.02) < 1.0e-15 &&
              std::abs(linear.trace.back().learning_rate - 0.01) < 1.0e-15,
          "linear learning-rate schedule is incorrect");

  project3::SurrogateParameters constrained{
      .values = {100.0, -100.0, 0.0, 0.0, 0.0, 0.0}};
  const auto projected = project3::optimize_differentiable_surrogate(
      constrained, paths, {}, one_step);
  require(std::abs(projected.parameters.values[0]) <=
                  project3::DifferentiableSurrogateConfig{}
                      .parameter_absolute_limit &&
              std::abs(projected.parameters.values[1]) <=
                  project3::DifferentiableSurrogateConfig{}
                      .parameter_absolute_limit,
          "optimizer did not project initial policy parameters into bounds");
  const auto evaluation =
      project3::evaluate_differentiable_surrogate(initial, paths);
  const auto decision = project3::smooth_surrogate_decision(
      initial, 2.0, paths.front().steps.front());
  require(decision.bid_offset_ticks >= 0.0 &&
              decision.bid_offset_ticks <= 5.0 &&
              decision.ask_offset_ticks >= 0.0 &&
              decision.ask_offset_ticks <= 5.0 &&
              decision.bid_fill_probability >= 0.0 &&
              decision.bid_fill_probability <= 1.0 &&
              decision.suggested_bid_size_lots >= 0.0 &&
              decision.suggested_bid_size_lots <=
                  project3::DifferentiableSurrogateConfig{}
                      .maximum_executed_quantity_lots &&
              decision.bid_cancel_probability >= 0.0 &&
              decision.bid_cancel_probability <= 1.0 &&
              std::abs(decision.next_inventory_lots -
                       (2.0 + decision.expected_bid_quantity_lots -
                        decision.expected_ask_quantity_lots)) <
                  1.0e-12,
          "single-observation surrogate decision is inconsistent");
  require(evaluation.decision_count == paths.size() * 12 * 2 &&
              evaluation.expected_executed_quantity_lots > 0.0 &&
              evaluation.mean_fill_probability > 0.0 &&
              std::isfinite(evaluation.adverse_selection_cost_currency) &&
              std::isfinite(evaluation.mean_terminal_inventory_lots),
          "surrogate evaluation omitted fill, inventory, or adverse-selection outputs");
}

void verify_policy_optimization_study() {
  auto training_market = market_config();
  training_market.horizon_seconds = 0.6;
  auto validation_market = training_market;
  auto evaluation_market = training_market;
  evaluation_market.hawkes.baseline_events_per_second[4] *= 2.0;
  evaluation_market.hawkes.baseline_events_per_second[5] *= 2.0;
  evaluation_market.marks.price_placement.max_offset_ticks += 2;
  evaluation_market.marks.market_size.max_lots += 1;

  project3::PolicyOptimizationStudyConfig config;
  config.training_scenarios = {
      {.label = "train-low-flow", .market = training_market, .seeds = {3001, 3002}}};
  config.validation_scenarios = {
      {.label = "validation-low-flow",
       .market = validation_market,
       .seeds = {3101}}};
  config.out_of_sample_scenarios = {
      {.label = "heldout-higher-flow-and-volatility",
       .market = evaluation_market,
       .seeds = {3201, 3202}}};
  config.policy_gradient.epochs = 2;
  config.policy_gradient.episodes_per_epoch = 2;
  config.policy_gradient.validation_patience = 2;
  config.policy_gradient.learning_rate = 0.002;
  config.policy_gradient.minimum_learning_rate = 0.001;
  config.policy_gradient.reward_normalization_scale_currency = 50.0;
  config.policy_gradient.gradient_clip_norm = 2.0;
  config.policy_gradient.inventory_limit_lots = 3.0;
  config.policy_gradient.decision_interval_seconds = 0.1;
  config.policy_gradient.quote_sizes_lots = {1, 1};
  config.policy_gradient.seed = 333;
  config.policy_gradient.evaluation.timer_interval_ns = 100'000'000;
  config.policy_gradient.evaluation.broker_config.risk.max_position_lots = 3;
  config.policy_gradient.evaluation.broker_config.risk.max_order_lots = 1;
  config.surrogate.epochs = 3;
  config.surrogate.validation_patience = 2;
  config.surrogate.learning_rate = 0.01;
  config.surrogate.minimum_learning_rate = 0.002;
  config.surrogate.surrogate.inventory_limit_lots = 3.0;
  config.surrogate.seed = 444;
  config.dynamic_programming.time_steps = 100;
  config.dynamic_programming.inventory_limit_lots = 3;
  config.dynamic_programming.max_quote_offset_ticks = 2.0;
  config.dynamic_programming.quote_offset_step_ticks = 0.5;

  const std::string pg_checkpoint = "policy-gradient-test-checkpoint.txt";
  const std::string pg_curve = "policy-gradient-test-curve.csv";
  const std::string surrogate_checkpoint = "surrogate-test-checkpoint.txt";
  const std::string surrogate_curve = "surrogate-test-curve.csv";
  config.policy_gradient.checkpoint_path = pg_checkpoint;
  config.policy_gradient.learning_curve_csv_path = pg_curve;
  config.surrogate.checkpoint_path = surrogate_checkpoint;
  config.surrogate.learning_curve_csv_path = surrogate_curve;

  const auto result = project3::run_policy_optimization_study(config);
  require(result.policy_gradient.completed_epochs > 0 &&
              result.policy_gradient.completed_epochs <=
                  config.policy_gradient.epochs &&
              !result.policy_gradient.learning_curve.empty() &&
              std::isfinite(result.policy_gradient.training_reward_currency) &&
              std::isfinite(
                  result.policy_gradient.best_validation_reward_currency),
          "simulator REINFORCE did not report separate train and validation rewards");
  require(!result.surrogate.learning_curve.empty() &&
              std::isfinite(result.surrogate.training_reward_currency) &&
              std::isfinite(
                  result.surrogate.training_exact_market_reward_currency) &&
              std::isfinite(
                  result.surrogate.validation_exact_market_reward_currency),
          "surrogate training did not report surrogate and exact-market rewards");
  require(result.out_of_sample.size() == 4 &&
              std::all_of(result.out_of_sample.begin(),
                          result.out_of_sample.end(), [](const auto& summary) {
                            return summary.scenario_label ==
                                       "heldout-higher-flow-and-volatility" &&
                                   summary.episodes == 2 &&
                                   std::isfinite(
                                       summary.mean_net_pnl_currency) &&
                                   std::isfinite(
                                       summary.mean_objective_reward_currency);
                          }),
          "learned and baseline policies were not evaluated on held-out market runs");
  const auto has_policy = [&](const std::string& name) {
    return std::any_of(result.out_of_sample.begin(),
                       result.out_of_sample.end(),
                       [&](const auto& summary) {
                         return summary.policy_name == name;
                       });
  };
  require(has_policy("DifferentiableSurrogatePolicy") &&
              has_policy("SimulatorPolicyGradient") &&
              has_policy("AvellanedaStoikov") &&
              has_policy("ReducedStateDynamicProgramming"),
          "out-of-sample results omitted one of the baseline/learned policies");
  require(std::filesystem::exists(pg_checkpoint) &&
              std::filesystem::exists(pg_curve) &&
              std::filesystem::exists(surrogate_checkpoint) &&
              std::filesystem::exists(surrogate_curve),
          "policy checkpoints or learning curves were not persisted");
  std::filesystem::remove(pg_checkpoint);
  std::filesystem::remove(pg_curve);
  std::filesystem::remove(surrogate_checkpoint);
  std::filesystem::remove(surrogate_curve);

  auto overlapping = config.training_scenarios;
  overlapping.front().seeds = config.validation_scenarios.front().seeds;
  bool rejected_overlap = false;
  try {
    static_cast<void>(project3::train_simulator_policy_gradient(
        overlapping, config.validation_scenarios, config.policy_gradient));
  } catch (const std::invalid_argument&) {
    rejected_overlap = true;
  }
  require(rejected_overlap,
          "policy training accepted overlapping train/validation seeds");
}

void verify_market_quality_analysis_and_stress_scenarios() {
  require(project3::adverse_selection_ticks(lob::core::Side::Buy, 101.0,
                                             100.0) == 1.0 &&
              project3::adverse_selection_ticks(lob::core::Side::Buy, 99.0,
                                                 100.0) == -1.0 &&
              project3::adverse_selection_ticks(lob::core::Side::Sell, 99.0,
                                                 100.0) == 1.0 &&
              project3::adverse_selection_ticks(lob::core::Side::Sell, 101.0,
                                                 100.0) == -1.0,
          "adverse-selection sign convention is inconsistent by execution side");

  auto baseline = market_config();
  const auto stresses =
      project3::make_market_stress_scenarios(baseline, {8101, 8102});
  require(stresses.size() == 7 &&
              stresses[0].market.hawkes.baseline_events_per_second[4] >
                  baseline.hawkes.baseline_events_per_second[4] &&
              stresses[1].market.hawkes.baseline_events_per_second[4] >
                  baseline.hawkes.baseline_events_per_second[4] &&
              stresses[1].market.hawkes.baseline_events_per_second[5] ==
                  baseline.hawkes.baseline_events_per_second[5] &&
              stresses[2].market.hawkes.alpha_branching[0][0] >
                  baseline.hawkes.alpha_branching[0][0] &&
              stresses[3].market.initial_book.order_size <
                  baseline.initial_book.order_size &&
              stresses[4].market.hawkes.baseline_events_per_second[2] >
                  baseline.hawkes.baseline_events_per_second[2] &&
              stresses[5].market.initial_book.best_ask_ticks -
                      stresses[5].market.initial_book.best_bid_ticks ==
                  baseline.initial_book.best_ask_ticks -
                      baseline.initial_book.best_bid_ticks + 4 &&
              stresses[6].market.marks.market_size.max_lots >
                  baseline.marks.market_size.max_lots,
          "one or more stress configurations did not change the intended market feature");

  baseline.horizon_seconds = 2.0;
  baseline.hawkes.baseline_events_per_second[4] = 20.0;
  baseline.hawkes.baseline_events_per_second[5] = 20.0;
  baseline.initial_book.levels_per_side = 2;
  baseline.initial_book.orders_per_level = 1;
  baseline.initial_book.order_size = 1;
  project3::MarketStressScenario evaluation_scenario{
      .label = "quality-analysis-test",
      .kind = project3::MarketStressKind::OneSidedOrderFlowBurst,
      .market = baseline,
      .seeds = {8201, 8202}};

  project3::MarketQualityConfig config;
  config.markout_horizons_seconds = {0.05, 0.2};
  config.inventory_limit_lots = 2.0;
  config.evaluation.timer_interval_ns = 25'000'000;
  config.evaluation.broker_config.risk.max_position_lots = 20;
  config.evaluation.broker_config.risk.max_order_lots = 2;
  const std::vector<project3::MarketQualityPolicy> policies{
      {.name = "quality-maker",
       .make_agent = [] {
         return std::make_unique<MarketQualityQuoteAgent>();
       }},
      {.name = "quality-canceller",
       .make_agent = [] {
         return std::make_unique<MarketQualityCancelAgent>();
       }},
  };
  const auto summaries = project3::run_market_quality_study(
      policies, {evaluation_scenario}, config);
  require(summaries.size() == 2 && summaries[0].episodes == 2 &&
              summaries[0].runs.size() == 2,
          "market-quality study omitted paired per-seed runs");
  const auto& maker = summaries[0];
  require(std::isfinite(maker.mean_net_pnl_currency) &&
              std::isfinite(maker.value_at_risk_loss_currency) &&
              std::isfinite(maker.expected_shortfall_loss_currency) &&
              maker.expected_shortfall_loss_currency >=
                  maker.value_at_risk_loss_currency &&
              maker.mean_maximum_drawdown_currency >= 0.0 &&
              maker.mean_markouts.size() == 2,
          "market-quality tail-risk or drawdown metrics are invalid");
  bool observed_fill = false;
  for (const auto& run : maker.runs) {
    observed_fill = observed_fill ||
                    run.market_maker_buys.execution_count +
                            run.market_maker_sells.execution_count >
                        0;
    require(std::abs(run.net_pnl_currency -
                     (run.gross_realized_pnl_currency +
                      run.unrealized_pnl_currency - run.fees_currency)) <
                1.0e-8 &&
                std::abs(run.fees_currency -
                         (run.charged_fees_currency -
                          run.rebates_currency)) <
                    1.0e-8 &&
                run.fill_probability >= 0.0 && run.fill_probability <= 1.0 &&
                run.cancellation_rate >= 0.0 &&
                run.cancellation_rate <= 1.0 &&
                run.inventory_concentration_ratio >= 0.0 &&
                run.inventory_limit_breach_duration_seconds >= 0.0,
            "market-quality episode accounting or rates are inconsistent");
    for (const auto& markout : run.markouts) {
      require(markout.market_maker_buys.execution_count +
                      markout.market_maker_buys
                          .right_censored_execution_count ==
                  run.market_maker_buys.execution_count &&
                  markout.market_maker_sells.execution_count +
                          markout.market_maker_sells
                              .right_censored_execution_count ==
                      run.market_maker_sells.execution_count,
              "markout horizon did not account for observed or censored fills");
    }
  }
  require(observed_fill,
          "market-quality instrumentation failed to record any execution");
  require(summaries[1].mean_cancellation_rate > 0.0,
          "market-quality instrumentation failed to record cancellations");

  auto shortfall_market = market_config();
  shortfall_market.horizon_seconds = 0.5;
  const project3::MarketStressScenario shortfall_scenario{
      .label = "shortfall-sign-test",
      .kind = project3::MarketStressKind::OneSidedOrderFlowBurst,
      .market = shortfall_market,
      .seeds = {8301}};
  const std::vector<project3::MarketQualityPolicy> aggressors{
      {.name = "buyer",
       .make_agent = [] {
         return std::make_unique<MarketQualityAggressorAgent>(
             lob::core::Side::Buy);
       }},
      {.name = "seller",
       .make_agent = [] {
         return std::make_unique<MarketQualityAggressorAgent>(
             lob::core::Side::Sell);
       }},
  };
  const auto shortfall_results = project3::run_market_quality_study(
      aggressors, {shortfall_scenario}, config);
  require(shortfall_results.size() == 2 &&
              std::abs(shortfall_results[0]
                           .market_maker_buys
                           .implementation_shortfall_ticks -
                       1.0) <
                  1.0e-12 &&
              std::abs(shortfall_results[1]
                           .market_maker_sells
                           .implementation_shortfall_ticks -
                       1.0) <
                  1.0e-12 &&
              shortfall_results[0]
                      .market_maker_buys.effective_spread_ticks <
                  0.0 &&
              shortfall_results[1]
                      .market_maker_sells.effective_spread_ticks <
                  0.0,
          "effective-spread or implementation-shortfall side sign is wrong");
  for (const auto& result : shortfall_results) {
    const auto& markout = result.mean_markouts.front();
    const auto& execution =
        result.policy_name == "buyer" ? result.market_maker_buys
                                      : result.market_maker_sells;
    const auto& side_markout =
        result.policy_name == "buyer" ? markout.market_maker_buys
                                      : markout.market_maker_sells;
    if (side_markout.execution_count > 0) {
      require(std::abs(execution.effective_spread_ticks -
                       (side_markout.realized_spread_ticks +
                        2.0 * side_markout.price_impact_ticks)) <
                  1.0e-12,
              "effective, realized, and impact spread decomposition failed");
    }
  }

  auto breach_config = config;
  breach_config.evaluation.broker_config.risk.max_order_lots = 5;
  const std::vector<project3::MarketQualityPolicy> over_limit_policy{
      {.name = "over-limit",
       .make_agent = [] {
         return std::make_unique<MarketQualityAggressorAgent>(
             lob::core::Side::Buy, 3);
       }},
  };
  const auto breach_result = project3::run_market_quality_study(
      over_limit_policy, {shortfall_scenario}, breach_config);
  require(breach_result.front().inventory_limit_breached &&
              breach_result.front()
                      .runs.front()
                      .inventory_limit_breach_duration_seconds > 0.0 &&
              breach_result.front()
                      .runs.front()
                      .inventory_concentration_ratio > 1.0,
          "market-quality analysis failed to identify an inventory-limit breach");

  const auto stress_results =
      project3::run_market_quality_study({policies.front()}, stresses, config);
  require(stress_results.size() == stresses.size() &&
              std::all_of(stress_results.begin(), stress_results.end(),
                          [](const auto& summary) {
                            return summary.episodes == 2 &&
                                   std::isfinite(
                                       summary.mean_net_pnl_currency) &&
                                   std::isfinite(
                                       summary.expected_shortfall_loss_currency) &&
                                   summary.inventory_limit_breached ==
                                       (summary.total_inventory_limit_breach_intervals >
                                        0);
                          }),
          "market-quality stress study did not evaluate policies stably across all scenarios");
}

void verify_reproducible_benchmark_runner() {
  const auto make_scenario =
      [](const std::string& label, const std::string& volatility,
         const std::string& liquidity, const std::uint64_t first_seed,
         const double flow_scale, const double excitation_scale,
         const lob::core::PriceTicks reference_price,
         const std::uint32_t book_order_size, const bool unseen,
         const bool perturbation) {
        auto market = market_config();
        market.horizon_seconds = 2.0;
        market.marks.price_placement.reference_price_ticks = reference_price;
        market.initial_book.best_bid_ticks = reference_price - 1;
        market.initial_book.best_ask_ticks = reference_price + 1;
        market.initial_book.levels_per_side = 1;
        market.initial_book.orders_per_level = 1;
        market.initial_book.order_size = book_order_size;
        for (auto& intensity : market.hawkes.baseline_events_per_second) {
          intensity *= 8.0 * flow_scale;
        }
        market.marks.market_size.min_lots = 2;
        market.marks.market_size.max_lots = 8;
        for (auto& row : market.hawkes.alpha_branching) {
          for (auto& value : row) value *= excitation_scale;
        }
        return project3::BenchmarkScenario{
            .label = label,
            .volatility_regime = volatility,
            .liquidity_regime = liquidity,
            .market = std::move(market),
            .seeds = {first_seed, first_seed + 1, first_seed + 2,
                      first_seed + 3},
            .parameter_perturbation = perturbation,
            .unseen_configuration = unseen};
      };

  project3::BenchmarkExperimentConfig config;
  config.experiment_id = "paired-benchmark-integration";
  config.market_quality.markout_horizons_seconds = {0.02};
  config.market_quality.inventory_limit_lots = 3.0;
  config.primary_adverse_selection_horizon_seconds = 0.02;
  config.training_scenarios = {
      make_scenario("train-calm", "calm", "deep", 9101, 1.0, 1.0, 1000, 5,
                    false, false),
      make_scenario("train-active", "active", "thin", 9111, 1.8, 1.2, 1010, 2,
                    false, false)};
  config.validation_scenarios = {
      make_scenario("validation-calm", "calm", "deep", 9201, 1.3, 1.1, 1020,
                    4, false, false),
      make_scenario("validation-active", "active", "thin", 9211, 2.1, 1.3,
                    1030, 2, false, false)};
  config.final_test_scenarios = {
      make_scenario("test-unseen-a", "volatile", "thin", 9301, 2.8, 1.5,
                    1100, 2, true, true),
      make_scenario("test-unseen-b", "very-volatile", "shallow", 9311, 3.5,
                    1.7, 1120, 1, true, true)};

  config.policies = {
      {.name = "baseline",
       .model_version = "baseline-v1",
       .configuration = "fixed one-lot best-price quotes",
       .make_agent = [] {
         return std::make_unique<MarketQualityQuoteAgent>();
       }},
      {.name = "cancel-reference",
       .model_version = "cancel-v1",
       .configuration = "single resting bid with timer cancellation",
       .make_agent = [] {
         return std::make_unique<MarketQualityCancelAgent>();
       }},
  };
  constexpr std::array<project3::AblationKind, 5> ablation_kinds{
      project3::AblationKind::WithoutHawkesInformation,
      project3::AblationKind::WithoutQueueInformation,
      project3::AblationKind::WithoutInventoryPenalties,
      project3::AblationKind::WithoutRegimeInformation,
      project3::AblationKind::WithoutDynamicQuoteAdjustment};
  constexpr std::array<const char*, 5> ablation_names{
      "no-hawkes", "no-queue", "no-inventory-penalty", "no-regime",
      "static-quotes"};
  for (std::size_t i = 0; i < ablation_kinds.size(); ++i) {
    const std::string name = ablation_names[i];
    config.ablations.push_back(
        {.kind = ablation_kinds[i],
         .baseline_policy_name = "baseline",
         .ablated_policy =
             {.name = name,
              .model_version = "ablation-v1",
              .configuration =
                  std::string{"integration fixture for "} + name,
              .make_agent = [] {
                return std::make_unique<MarketQualityQuoteAgent>();
              }}});
  }

  const auto unique_suffix = std::chrono::steady_clock::now()
                                 .time_since_epoch()
                                 .count();
  const auto output_prefix =
      (std::filesystem::temp_directory_path() /
       ("project3-benchmark-" + std::to_string(unique_suffix)))
          .string();
  config.output_prefix = output_prefix;
  const auto result = project3::run_market_making_benchmark(config);
  require(result.summaries.size() == 3 * 3 * 7 &&
              result.paired_comparisons.size() == 3 * 2 * 21 &&
              result.manifest_path == output_prefix + ".manifest.json" &&
              result.runs_csv_path == output_prefix + ".runs.csv" &&
              result.paired_csv_path == output_prefix + ".paired.csv" &&
              result.summaries_csv_path == output_prefix + ".summaries.csv",
          "benchmark did not evaluate every split, policy, and paired comparison");

  const auto final_summary = std::find_if(
      result.summaries.begin(), result.summaries.end(),
      [](const project3::BenchmarkMetricSummary& summary) {
        return summary.split == project3::BenchmarkSplit::FinalTest &&
               summary.scenario_label == "ALL" &&
               summary.policy_name == "baseline";
      });
  require(final_summary != result.summaries.end() &&
              final_summary->sample_count == 8 &&
              final_summary->unseen_configuration &&
              final_summary->parameter_perturbation &&
              final_summary->pnl_confidence_interval_low_currency <=
                  final_summary->mean_pnl_currency &&
              final_summary->mean_pnl_currency <=
                  final_summary->pnl_confidence_interval_high_currency &&
              std::isfinite(final_summary->expected_shortfall_loss_currency),
          "final-test aggregate lost its unseen/perturbation or risk metadata");

  const auto identical_pair = std::find_if(
      result.paired_comparisons.begin(), result.paired_comparisons.end(),
      [](const project3::PairedBenchmarkComparison& comparison) {
        return comparison.split == project3::BenchmarkSplit::Training &&
               comparison.scenario_label == "train-calm" &&
               comparison.reference_policy == "baseline" &&
               comparison.comparison_policy == "no-hawkes";
      });
  require(identical_pair != result.paired_comparisons.end() &&
              identical_pair->paired_samples == 4 &&
              identical_pair->mean_pnl_difference_currency == 0.0 &&
              identical_pair->difference_confidence_interval_low_currency ==
                  0.0 &&
              identical_pair->difference_confidence_interval_high_currency ==
                  0.0,
          "paired comparison failed to preserve identical-seed differences");

  const auto variable_pair = std::find_if(
      result.paired_comparisons.begin(), result.paired_comparisons.end(),
      [](const project3::PairedBenchmarkComparison& comparison) {
        return comparison.split == project3::BenchmarkSplit::Training &&
               comparison.scenario_label == "train-calm" &&
               comparison.reference_policy == "baseline" &&
               comparison.comparison_policy == "cancel-reference";
      });
  require(variable_pair != result.paired_comparisons.end() &&
              variable_pair->pnl_difference_standard_deviation_currency > 0.0,
          "paired confidence-interval fixture did not produce varying returns");
  const double paired_standard_error =
      variable_pair->pnl_difference_standard_deviation_currency /
      std::sqrt(static_cast<double>(variable_pair->paired_samples));
  const double paired_critical =
      (variable_pair->difference_confidence_interval_high_currency -
       variable_pair->mean_pnl_difference_currency) /
      paired_standard_error;
  require(variable_pair->paired_samples == 4 &&
              std::abs(paired_critical - 3.182446305284263) < 1.0e-7,
          "four-observation paired interval does not use the exact t critical value");

  const auto read_file = [](const std::string& path) {
    std::ifstream input{path};
    if (!input) throw std::runtime_error{"benchmark artifact was not created"};
    return std::string{std::istreambuf_iterator<char>{input},
                       std::istreambuf_iterator<char>{}};
  };
  const auto manifest = read_file(result.manifest_path);
  const auto runs_csv = read_file(result.runs_csv_path);
  const auto paired_csv = read_file(result.paired_csv_path);
  const auto summaries_csv = read_file(result.summaries_csv_path);
  require(manifest.find("\"final_test\"") != std::string::npos &&
              manifest.find("\"model_version\": \"baseline-v1\"") !=
                  std::string::npos &&
              manifest.find("\"kind\": \"without_queue_information\"") !=
                  std::string::npos &&
              runs_csv.find("volatility_regime,liquidity_regime") !=
                  std::string::npos &&
              runs_csv.find("net_transaction_costs_currency") !=
                  std::string::npos &&
              runs_csv.find("fees_currency,charged_fees_currency") ==
                  std::string::npos &&
              paired_csv.find("difference_std_currency") != std::string::npos &&
              summaries_csv.find("ci_low_currency") != std::string::npos &&
              summaries_csv.find("final_test,\"ALL\",true,true") !=
                  std::string::npos,
          "benchmark artifacts omitted split, policy, paired, or summary data");

  auto overlapping_seeds = config;
  overlapping_seeds.output_prefix.clear();
  overlapping_seeds.final_test_scenarios.front().seeds.front() =
      overlapping_seeds.training_scenarios.front().seeds.front();
  bool rejected_seed_overlap = false;
  try {
    static_cast<void>(
        project3::run_market_making_benchmark(overlapping_seeds));
  } catch (const std::invalid_argument&) {
    rejected_seed_overlap = true;
  }
  require(rejected_seed_overlap,
          "benchmark accepted random-seed overlap across data splits");

  auto duplicate_configuration = config;
  duplicate_configuration.output_prefix.clear();
  duplicate_configuration.final_test_scenarios.front().market =
      duplicate_configuration.training_scenarios.front().market;
  bool rejected_duplicate_test_market = false;
  try {
    static_cast<void>(
        project3::run_market_making_benchmark(duplicate_configuration));
  } catch (const std::invalid_argument&) {
    rejected_duplicate_test_market = true;
  }
  require(rejected_duplicate_test_market,
          "benchmark accepted a development configuration as final test");

  for (const auto& path : {result.manifest_path, result.runs_csv_path,
                           result.paired_csv_path, result.summaries_csv_path}) {
    std::filesystem::remove(path);
  }
}

}  // namespace

int main() {
  try {
    verify_replay_preserves_reference_behavior();
    verify_partial_fill_notifications();
    verify_round_trip_realized_pnl_and_equity_reconciliation();
    verify_cancel_gateway();
    verify_seeded_runs_are_reproducible();
    verify_timer_intensities_use_only_observed_events();
    verify_queue_depletion_cancellation_and_replacement();
    verify_avellaneda_stoikov_quote_limits();
    verify_avellaneda_stoikov_baseline_evaluation();
    verify_hawkes_state_and_out_of_sample_policy_comparison();
    verify_reduced_state_dynamic_programming();
    verify_differentiable_market_making_surrogate();
    verify_policy_optimization_study();
    verify_market_quality_analysis_and_stress_scenarios();
    verify_reproducible_benchmark_runner();
    std::cout << "All Project 3 integration tests passed.\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "Integration test failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
