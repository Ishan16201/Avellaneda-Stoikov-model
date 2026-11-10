#pragma once

#include "project3/market_agent.hpp"

#include <map>
#include <optional>
#include <tuple>

namespace project3 {

/// Policy for interpreting unexplained reductions in aggregate displayed depth.
/// These are queue-estimation assumptions, not observable exchange facts.
enum class CancellationAllocation {
  /// Attribute cancellations to behind the agent first; conservative for queue progress.
  BehindFirst,
  /// Attribute cancellations to ahead first; optimistic queue progress.
  AheadFirst,
  /// Split each reduction proportionally over estimated ahead/behind quantity.
  ProRata,
};

class QueueTracker final {
 public:
  struct Config final {
    CancellationAllocation cancellation_allocation{CancellationAllocation::BehindFirst};
  };

  QueueTracker();
  explicit QueueTracker(Config config);

  /// Seed/update aggregate displayed depth. Negative net changes not explained
  /// by public trades or known own-order actions use cancellation_allocation.
  void observe_book(const MarketView& view);
  /// Update the depth baseline after a change already explained by a trade,
  /// own fill, or confirmed own cancellation.
  void synchronize_book(const MarketView& view);

  /// Add a limit order behind all currently known same-price queue quantity.
  /// Assumes immediate order arrival; cancellations remain estimators based on
  /// aggregate depth deltas and the configured allocation rule.
  void add_order(lob::core::OrderId order_id,
                 lob::core::Side side,
                 lob::core::PriceTicks price_ticks,
                 lob::core::QuantityLots quantity,
                 const MarketView& view);

  /// Apply a confirmed own fill. A later public trade with the same timestamp,
  /// price and passive side subtracts this quantity to avoid double depletion.
  void apply_fill(const ExecutionNotification& execution);
  void cancel_order(lob::core::OrderId order_id);
  void reject_order(lob::core::OrderId order_id);

  /// Apply public trade volume at a resting price. Own fills already reported
  /// for this print are excluded from external queue depletion.
  void observe_trade(const lob::strategy::StrategyMarketTrade& trade);

  /// Cancel and re-add with a new ID/priority; the old queue position is never kept.
  void replace_order(lob::core::OrderId old_order_id,
                     lob::core::OrderId new_order_id,
                     lob::core::Side side,
                     lob::core::PriceTicks price_ticks,
                     lob::core::QuantityLots quantity,
                     const MarketView& view);

  [[nodiscard]] std::optional<QueuePosition> position(
      lob::core::OrderId order_id) const;
  [[nodiscard]] std::vector<QueuePosition> positions() const;

 private:
  using LevelKey = std::pair<lob::core::Side, lob::core::PriceTicks>;
  using TradeKey = std::tuple<lob::core::TimestampNs,
                              lob::core::PriceTicks,
                              lob::core::Side>;

  [[nodiscard]] lob::core::QuantityLots depth_at(
      const MarketView& view, LevelKey key) const;
  void apply_depth_change(LevelKey key, lob::core::QuantityLots old_quantity,
                          lob::core::QuantityLots new_quantity);
  void apply_cancellation(LevelKey key, lob::core::QuantityLots quantity);
  void decrease_pending_arrival(LevelKey key, lob::core::QuantityLots quantity);

  Config config_;
  std::map<lob::core::OrderId, QueuePosition> positions_;
  std::map<lob::core::OrderId, std::uint64_t> insertion_sequence_;
  std::map<LevelKey, lob::core::QuantityLots> last_depth_;
  std::map<LevelKey, lob::core::QuantityLots> pending_own_arrivals_;
  std::map<TradeKey, lob::core::QuantityLots> reported_own_fills_;
  std::uint64_t next_sequence_{};
  bool has_depth_snapshot_{};
};

}  // namespace project3
