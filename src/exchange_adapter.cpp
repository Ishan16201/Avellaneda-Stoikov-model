#include "project3/exchange_adapter.hpp"
#include "project3/execution_layer.hpp"

#include "lob/strategy/strategy_evaluator.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace project3 {
namespace {

MarketView make_view(const lob::strategy::StrategyContext& context,
                     const QueueTracker& queue_tracker,
                     const std::vector<ExecutionNotification>& history) {
  const auto& source = context.current_book();
  MarketView view;
  view.timestamp_ns = context.current_time_ns();
  view.bids.reserve(source.bids.size());
  view.asks.reserve(source.asks.size());
  for (const auto& level : source.bids) {
    view.bids.push_back({level.price_ticks, level.quantity});
  }
  for (const auto& level : source.asks) {
    view.asks.push_back({level.price_ticks, level.quantity});
  }
  view.hawkes_intensities = context.hawkes_intensities();
  if (view.hawkes_intensities.size() == lob::stochastic::kEventTypeCount) {
    view.hawkes_flow = {
        .limit_buy = view.hawkes_intensities[lob::stochastic::event_type_index(
            lob::stochastic::EventType::LimitBuy)],
        .limit_sell = view.hawkes_intensities[lob::stochastic::event_type_index(
            lob::stochastic::EventType::LimitSell)],
        .cancel_buy = view.hawkes_intensities[lob::stochastic::event_type_index(
            lob::stochastic::EventType::CancelBuy)],
        .cancel_sell = view.hawkes_intensities[lob::stochastic::event_type_index(
            lob::stochastic::EventType::CancelSell)],
        .market_buy = view.hawkes_intensities[lob::stochastic::event_type_index(
            lob::stochastic::EventType::MarketBuy)],
        .market_sell = view.hawkes_intensities[lob::stochastic::event_type_index(
            lob::stochastic::EventType::MarketSell)],
        .available = true,
    };
  }
  view.best_bid = context.best_bid();
  view.best_ask = context.best_ask();
  view.midprice_ticks = context.midprice();
  if (view.best_bid && view.best_ask) {
    view.spread_ticks = *view.best_ask - *view.best_bid;
  }
  view.inventory_lots = context.portfolio().position_lots;
  view.cash_currency = context.portfolio().cash_currency;
  view.market_maker.account = context.portfolio();
  view.market_maker.current_inventory_exposure_lots =
      static_cast<lob::core::QuantityLots>(
          context.portfolio().position_lots < 0
              ? -context.portfolio().position_lots
              : context.portfolio().position_lots);
  view.market_maker.max_inventory_exposure_lots =
      static_cast<lob::core::QuantityLots>(context.max_inventory_exposure_lots());
  view.market_maker.order_execution_history = history;
  view.open_orders = context.open_orders();
  view.queue_positions = queue_tracker.positions();
  return view;
}

class AgentGateway final : public OrderGateway {
 public:
  AgentGateway(TradingAgent& agent, QueueTracker& queue_tracker)
      : agent_(agent), queue_tracker_(queue_tracker) {}

  void update_view(const lob::strategy::StrategyContext& context) {
    current_view_ = make_view(context, queue_tracker_, history_);
  }

  void cancel(const lob::core::OrderId order_id) override {
    forwarded_.cancel(order_id);
  }

  [[nodiscard]] lob::core::OrderId submit_limit(
      const lob::core::Side side,
      const lob::core::PriceTicks price,
      const lob::core::QuantityLots quantity) override {
    const auto id = next_id();
    if (current_view_.has_value()) {
      queue_tracker_.add_order(id, side, price, quantity, *current_view_);
    }
    forwarded_.submit_limit(id, side, price, quantity);
    return id;
  }

  [[nodiscard]] lob::core::OrderId submit_market(
      const lob::core::Side side,
      const lob::core::QuantityLots quantity) override {
    const auto id = next_id();
    forwarded_.submit_market(id, side, quantity);
    return id;
  }

  void on_report(const lob::strategy::StrategyContext& context,
                 const lob::strategy::StrategyExecutionReport& report) {
    ExecutionNotification notification{
        .order_id = report.client_order_id,
        .timestamp_ns = report.timestamp_ns,
        .side = report.side,
        .price_ticks = report.execution_price_ticks,
        .filled_quantity = report.filled_quantity,
        .remaining_quantity = report.remaining_quantity,
        .fee_paid_currency = report.fee_paid_currency,
        .execution_cost_currency = report.execution_cost_currency,
        .status = report.status,
        .passive = report.is_passive,
    };
    const bool filled = notification.filled_quantity > 0;
    const bool cancelled =
        notification.status == lob::strategy::StrategyOrderStatus::Cancelled;
    if (filled) {
      queue_tracker_.apply_fill(notification);
    }
    if (cancelled) {
      queue_tracker_.cancel_order(notification.order_id);
    } else if (notification.status == lob::strategy::StrategyOrderStatus::Rejected ||
               notification.status == lob::strategy::StrategyOrderStatus::Expired) {
      queue_tracker_.reject_order(notification.order_id);
    }
    history_.push_back(notification);
    auto view = make_view(context, queue_tracker_, history_);
    // synchronize_book only reads displayed depth, so one resync after all
    // tracker mutations is equivalent to the previous per-step resyncs.
    if ((filled && !notification.passive) || cancelled) {
      queue_tracker_.synchronize_book(view);
    }
    current_view_ = view;
    agent_.on_execution(
        notification, view,
        *this);
  }

  void observe_trade(const lob::strategy::StrategyMarketTrade& trade,
                     const lob::strategy::StrategyContext& context) {
    queue_tracker_.observe_trade(trade);
    auto current = make_view(context, queue_tracker_, history_);
    queue_tracker_.synchronize_book(current);
    current.queue_positions = queue_tracker_.positions();
    current_view_ = std::move(current);
  }

  void observe_book(const lob::strategy::StrategyContext& context) {
    auto current = make_view(context, queue_tracker_, history_);
    queue_tracker_.observe_book(current);
    current.queue_positions = queue_tracker_.positions();
    current_view_ = std::move(current);
  }

  [[nodiscard]] MarketView view(const lob::strategy::StrategyContext& context) {
    if (!current_view_.has_value()) {
      current_view_ = make_view(context, queue_tracker_, history_);
    }
    return *current_view_;
  }

  void update_emitter(lob::strategy::StrategyActionEmitter& emitter) noexcept {
    forwarded_.update(emitter);
  }

 private:
  [[nodiscard]] lob::core::OrderId next_id() {
    if (next_order_id_ == 0) {
      throw std::overflow_error{"Project 3 agent order IDs exhausted"};
    }
    return next_order_id_--;
  }

  class Forwarder final : public lob::strategy::StrategyActionEmitter {
   public:
    void update(lob::strategy::StrategyActionEmitter& emitter) noexcept {
      emitter_ = &emitter;
    }
    void submit_limit(lob::core::OrderId id, lob::core::Side side,
                      lob::core::PriceTicks price,
                      lob::core::QuantityLots quantity) override {
      require_emitter().submit_limit(id, side, price, quantity);
    }
    void submit_market(lob::core::OrderId id, lob::core::Side side,
                       lob::core::QuantityLots quantity) override {
      require_emitter().submit_market(id, side, quantity);
    }
    void cancel(lob::core::OrderId id) override {
      require_emitter().cancel(id);
    }
   private:
    [[nodiscard]] lob::strategy::StrategyActionEmitter& require_emitter() const {
      if (emitter_ == nullptr) {
        throw std::logic_error{"agent gateway has no active broker"};
      }
      return *emitter_;
    }
    lob::strategy::StrategyActionEmitter* emitter_{};
  };

 private:
  TradingAgent& agent_;
  QueueTracker& queue_tracker_;
  std::vector<ExecutionNotification> history_;
  std::optional<MarketView> current_view_;
  lob::core::OrderId next_order_id_{std::numeric_limits<lob::core::OrderId>::max()};
  Forwarder forwarded_{};
};

class AgentBridge final : public lob::strategy::Strategy {
 public:
  explicit AgentBridge(TradingAgent& agent)
      : agent_(agent), gateway_(agent, queue_tracker_) {}

  void on_start(const lob::strategy::StrategyContext& context,
                lob::strategy::StrategyActionEmitter& emitter) override {
    update(emitter);
    gateway_.observe_book(context);
    agent_.on_start(gateway_.view(context), gateway_);
  }

  void on_quote(const lob::strategy::StrategyContext& context,
                lob::strategy::StrategyActionEmitter& emitter) override {
    update(emitter);
    gateway_.observe_book(context);
    agent_.on_market_event(gateway_.view(context), gateway_);
  }

  void on_trade(const lob::strategy::StrategyContext& context,
                const lob::strategy::StrategyMarketTrade& trade,
                lob::strategy::StrategyActionEmitter& emitter) override {
    update(emitter);
    gateway_.observe_trade(trade, context);
    agent_.on_trade(trade, gateway_.view(context));
  }

  void on_execution_report(const lob::strategy::StrategyContext& context,
                           const lob::strategy::StrategyExecutionReport& report,
                           lob::strategy::StrategyActionEmitter& emitter) override {
    update(emitter);
    gateway_.on_report(context, report);
  }

  void on_timer(const lob::strategy::StrategyContext& context,
                lob::strategy::StrategyActionEmitter& emitter) override {
    update(emitter);
    gateway_.observe_book(context);
    agent_.on_timer(gateway_.view(context), gateway_);
  }

  void on_stop(const lob::strategy::StrategyContext& context,
               lob::strategy::StrategyActionEmitter& emitter) override {
    update(emitter);
    gateway_.observe_book(context);
    agent_.on_stop(gateway_.view(context), gateway_);
  }

  [[nodiscard]] std::string name() const override { return agent_.name(); }

 private:
  void update(lob::strategy::StrategyActionEmitter& emitter) {
    gateway_.update_emitter(emitter);
  }

  TradingAgent& agent_;
  QueueTracker queue_tracker_{};
  AgentGateway gateway_;
};

}  // namespace

lob::strategy::StrategyEvaluationReport ExchangeAdapter::run(
    TradingAgent& agent,
    const lob::stochastic::SyntheticMarketConfig& market,
    const lob::strategy::StrategyEvaluatorConfig& evaluation) const {
  AgentBridge bridge(agent);
  return lob::strategy::StrategyEvaluator::run_single(bridge, market, evaluation);
}

}  // namespace project3
