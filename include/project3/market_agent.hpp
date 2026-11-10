#pragma once

#include "lob/core/types.hpp"
#include "lob/stochastic/hawkes_market_adapter.hpp"
#include "lob/strategy/strategy.hpp"
#include "lob/strategy/types.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace project3 {

struct HawkesFlowState final {
  double limit_buy{};
  double limit_sell{};
  double cancel_buy{};
  double cancel_sell{};
  double market_buy{};
  double market_sell{};
  bool available{};

  [[nodiscard]] double intensity(lob::stochastic::EventType type) const noexcept {
    switch (type) {
      case lob::stochastic::EventType::LimitBuy: return limit_buy;
      case lob::stochastic::EventType::LimitSell: return limit_sell;
      case lob::stochastic::EventType::CancelBuy: return cancel_buy;
      case lob::stochastic::EventType::CancelSell: return cancel_sell;
      case lob::stochastic::EventType::MarketBuy: return market_buy;
      case lob::stochastic::EventType::MarketSell: return market_sell;
    }
    return 0.0;
  }

  /// Signed liquidity/market-order pressure, normalized by total flow rate.
  /// Bid additions, ask cancellations and buy market orders are positive;
  /// ask additions, bid cancellations and sell market orders are negative.
  [[nodiscard]] double order_flow_imbalance() const noexcept {
    if (!available) return 0.0;
    const double scale = std::max({limit_buy, limit_sell, cancel_buy,
                                   cancel_sell, market_buy, market_sell});
    if (scale <= 0.0) return 0.0;
    const double bid_pressure =
        limit_buy / scale + cancel_sell / scale + market_buy / scale;
    const double ask_pressure =
        limit_sell / scale + cancel_buy / scale + market_sell / scale;
    const double total = bid_pressure + ask_pressure;
    return total > 0.0 ? (bid_pressure - ask_pressure) / total : 0.0;
  }
};

struct PriceLevel final {
  lob::core::PriceTicks price_ticks{};
  lob::core::QuantityLots quantity{};
  [[nodiscard]] bool operator==(const PriceLevel&) const = default;
};

struct ExecutionNotification final {
  lob::core::OrderId order_id{};
  lob::core::TimestampNs timestamp_ns{};
  lob::core::Side side{lob::core::Side::Buy};
  lob::core::PriceTicks price_ticks{};
  lob::core::QuantityLots filled_quantity{};
  lob::core::QuantityLots remaining_quantity{};
  double fee_paid_currency{};
  double execution_cost_currency{};
  lob::strategy::StrategyOrderStatus status{lob::strategy::StrategyOrderStatus::New};
  bool passive{};
};

struct QueuePosition final {
  lob::core::OrderId order_id{};
  lob::core::Side side{lob::core::Side::Buy};
  lob::core::PriceTicks price_ticks{};
  lob::core::QuantityLots quantity_ahead{};
  lob::core::QuantityLots own_remaining{};
  lob::core::QuantityLots quantity_behind{};
  bool estimated{true};
};

struct MarketMakerState final {
  lob::strategy::PortfolioState account{};
  lob::core::QuantityLots current_inventory_exposure_lots{};
  lob::core::QuantityLots max_inventory_exposure_lots{};
  std::vector<ExecutionNotification> order_execution_history;
};

/// Point-in-time observations only; never includes future events or book states.
struct MarketView final {
  lob::core::TimestampNs timestamp_ns{};
  std::vector<PriceLevel> bids;
  std::vector<PriceLevel> asks;
  std::vector<double> hawkes_intensities;
  HawkesFlowState hawkes_flow;
  std::optional<lob::core::PriceTicks> best_bid;
  std::optional<lob::core::PriceTicks> best_ask;
  std::optional<double> midprice_ticks;
  std::optional<lob::core::PriceTicks> spread_ticks;
  MarketMakerState market_maker;
  std::vector<lob::strategy::OpenStrategyOrder> open_orders;
  std::int64_t inventory_lots{};
  double cash_currency{};
  std::vector<QueuePosition> queue_positions;
};

class OrderGateway {
 public:
  virtual ~OrderGateway() = default;
  [[nodiscard]] virtual lob::core::OrderId submit_limit(
      lob::core::Side side,
      lob::core::PriceTicks price_ticks,
      lob::core::QuantityLots quantity) = 0;
  [[nodiscard]] virtual lob::core::OrderId submit_market(
      lob::core::Side side,
      lob::core::QuantityLots quantity) = 0;
  virtual void cancel(lob::core::OrderId order_id) = 0;
};

/// Agent boundary used by Project 3; exchange internals remain behind the adapter.
class TradingAgent {
 public:
  virtual ~TradingAgent() = default;
  [[nodiscard]] virtual std::string name() const = 0;
  virtual void on_start(const MarketView&, OrderGateway&) {}
  virtual void on_market_event(const MarketView&, OrderGateway&) {}
  virtual void on_trade(const lob::strategy::StrategyMarketTrade&, const MarketView&) {}
  virtual void on_execution(const ExecutionNotification&,
                            const MarketView&,
                            OrderGateway&) {}
  virtual void on_timer(const MarketView&, OrderGateway&) {}
  virtual void on_stop(const MarketView&, OrderGateway&) {}
};

}  // namespace project3
