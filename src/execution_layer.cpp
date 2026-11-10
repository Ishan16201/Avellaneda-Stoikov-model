#include "project3/execution_layer.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace project3 {
namespace {

using Quantity = lob::core::QuantityLots;

Quantity checked_add(const Quantity lhs, const Quantity rhs) {
  if (std::numeric_limits<Quantity>::max() - lhs < rhs) {
    throw std::overflow_error{"queue quantity overflow"};
  }
  return lhs + rhs;
}

void subtract_clamped(Quantity& value, const Quantity amount) noexcept {
  value = amount >= value ? 0 : value - amount;
}

}  // namespace

QueueTracker::QueueTracker() : QueueTracker(Config{}) {}

QueueTracker::QueueTracker(Config config) : config_(config) {}

Quantity QueueTracker::depth_at(const MarketView& view, const LevelKey key) const {
  const auto& levels = key.first == lob::core::Side::Buy ? view.bids : view.asks;
  const auto found = std::find_if(levels.begin(), levels.end(), [key](const auto& level) {
    return level.price_ticks == key.second;
  });
  return found == levels.end() ? 0 : found->quantity;
}

void QueueTracker::apply_depth_change(const LevelKey key,
                                      const Quantity old_quantity,
                                      const Quantity new_quantity) {
  if (new_quantity > old_quantity) {
    Quantity increase = new_quantity - old_quantity;
    const auto pending = pending_own_arrivals_.find(key);
    if (pending != pending_own_arrivals_.end()) {
      const Quantity own_arrival = std::min(increase, pending->second);
      increase -= own_arrival;
      pending->second -= own_arrival;
      if (pending->second == 0) pending_own_arrivals_.erase(pending);
    }
    if (increase > 0) {
      for (auto& [id, position] : positions_) {
        static_cast<void>(id);
        if (position.side == key.first && position.price_ticks == key.second) {
          position.quantity_behind =
              checked_add(position.quantity_behind, increase);
        }
      }
    }
  } else if (old_quantity > new_quantity) {
    apply_cancellation(key, old_quantity - new_quantity);
  }
}

void QueueTracker::observe_book(const MarketView& view) {
  std::vector<std::pair<LevelKey, Quantity>> observed;
  observed.reserve(view.bids.size() + view.asks.size());
  for (const auto& level : view.bids) {
    observed.emplace_back(
        LevelKey{lob::core::Side::Buy, level.price_ticks}, level.quantity);
  }
  for (const auto& level : view.asks) {
    observed.emplace_back(
        LevelKey{lob::core::Side::Sell, level.price_ticks}, level.quantity);
  }
  std::stable_sort(observed.begin(), observed.end(),
                   [](const auto& left, const auto& right) {
                     return left.first < right.first;
                   });
  std::size_t unique = 0;
  for (std::size_t i = 0; i < observed.size(); ++i) {
    if (unique > 0 && observed[i].first == observed[unique - 1].first) {
      observed[unique - 1].second = observed[i].second;
    } else {
      observed[unique++] = observed[i];
    }
  }
  observed.resize(unique);

  if (!has_depth_snapshot_) {
    last_depth_.clear();
    for (const auto& [key, quantity] : observed) {
      last_depth_[key] = quantity;
    }
    has_depth_snapshot_ = true;
    return;
  }

  std::size_t next_observed = 0;
  for (const auto& [key, old_quantity] : last_depth_) {
    while (next_observed < observed.size() &&
           observed[next_observed].first < key) {
      apply_depth_change(observed[next_observed].first, 0,
                         observed[next_observed].second);
      ++next_observed;
    }
    Quantity new_quantity = 0;
    if (next_observed < observed.size() &&
        observed[next_observed].first == key) {
      new_quantity = observed[next_observed].second;
      ++next_observed;
    }
    apply_depth_change(key, old_quantity, new_quantity);
  }
  while (next_observed < observed.size()) {
    apply_depth_change(observed[next_observed].first, 0,
                       observed[next_observed].second);
    ++next_observed;
  }
  last_depth_.clear();
  for (const auto& [key, quantity] : observed) {
    last_depth_[key] = quantity;
  }
}

void QueueTracker::synchronize_book(const MarketView& view) {
  last_depth_.clear();
  for (const auto& level : view.bids) {
    last_depth_[{lob::core::Side::Buy, level.price_ticks}] = level.quantity;
  }
  for (const auto& level : view.asks) {
    last_depth_[{lob::core::Side::Sell, level.price_ticks}] = level.quantity;
  }
  has_depth_snapshot_ = true;
}

void QueueTracker::add_order(const lob::core::OrderId order_id,
                             const lob::core::Side side,
                             const lob::core::PriceTicks price_ticks,
                             const Quantity quantity,
                             const MarketView& view) {
  if (order_id == 0 || !lob::core::is_valid_side(side) || price_ticks <= 0 ||
      quantity == 0 || positions_.contains(order_id)) {
    throw std::invalid_argument{"invalid or duplicate queue order"};
  }
  if (next_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error{"queue order sequence exhausted"};
  }

  const LevelKey key{side, price_ticks};
  const Quantity displayed = depth_at(view, key);
  Quantity tracked_own = 0;
  for (const auto& [id, position] : positions_) {
    static_cast<void>(id);
    if (position.side == side && position.price_ticks == price_ticks) {
      tracked_own = checked_add(tracked_own, position.own_remaining);
    }
  }
  Quantity visible_own = 0;
  for (const auto& order : view.open_orders) {
    if (order.side == side && order.price_ticks == price_ticks) {
      visible_own = checked_add(visible_own, order.remaining_quantity);
    }
  }
  visible_own = std::min(displayed, visible_own);
  const Quantity external_ahead = displayed - visible_own;
  const Quantity ahead = checked_add(external_ahead, tracked_own);

  for (auto& [id, position] : positions_) {
    static_cast<void>(id);
    if (position.side == side && position.price_ticks == price_ticks) {
      position.quantity_behind = checked_add(position.quantity_behind, quantity);
    }
  }
  positions_.emplace(order_id, QueuePosition{
                                   .order_id = order_id,
                                   .side = side,
                                   .price_ticks = price_ticks,
                                   .quantity_ahead = ahead,
                                   .own_remaining = quantity,
                                   .quantity_behind = 0,
                                   .estimated = true});
  insertion_sequence_.emplace(order_id, next_sequence_++);
  auto& pending = pending_own_arrivals_[key];
  pending = checked_add(pending, quantity);
}

void QueueTracker::decrease_pending_arrival(const LevelKey key, const Quantity quantity) {
  const auto found = pending_own_arrivals_.find(key);
  if (found == pending_own_arrivals_.end()) return;
  subtract_clamped(found->second, quantity);
  if (found->second == 0) pending_own_arrivals_.erase(found);
}

void QueueTracker::apply_fill(const ExecutionNotification& execution) {
  if (execution.filled_quantity == 0) return;
  const auto found = positions_.find(execution.order_id);
  if (found == positions_.end()) return;
  auto& position = found->second;
  if (execution.filled_quantity > position.own_remaining) {
    throw std::logic_error{"execution exceeds tracked order remainder"};
  }
  position.own_remaining -= execution.filled_quantity;
  decrease_pending_arrival({position.side, position.price_ticks},
                           execution.filled_quantity);

  const auto filled_sequence = insertion_sequence_.at(execution.order_id);
  for (auto& [id, other] : positions_) {
    if (id == execution.order_id || other.side != position.side ||
        other.price_ticks != position.price_ticks) {
      continue;
    }
    if (insertion_sequence_.at(id) > filled_sequence) {
      subtract_clamped(other.quantity_ahead, execution.filled_quantity);
    } else {
      subtract_clamped(other.quantity_behind, execution.filled_quantity);
    }
  }

  if (execution.passive) {
    const TradeKey key{execution.timestamp_ns, execution.price_ticks, execution.side};
    auto& known_fill = reported_own_fills_[key];
    known_fill = checked_add(known_fill, execution.filled_quantity);
  }
  if (position.own_remaining == 0) {
    insertion_sequence_.erase(execution.order_id);
    positions_.erase(found);
  }
}

void QueueTracker::cancel_order(const lob::core::OrderId order_id) {
  const auto found = positions_.find(order_id);
  if (found == positions_.end()) return;
  const auto key = LevelKey{found->second.side, found->second.price_ticks};
  const Quantity cancelled = found->second.own_remaining;
  decrease_pending_arrival(key, cancelled);
  const auto cancelled_sequence = insertion_sequence_.at(order_id);
  for (auto& [id, other] : positions_) {
    if (id == order_id || other.side != key.first || other.price_ticks != key.second) {
      continue;
    }
    if (insertion_sequence_.at(id) > cancelled_sequence) {
      subtract_clamped(other.quantity_ahead, cancelled);
    } else {
      subtract_clamped(other.quantity_behind, cancelled);
    }
  }
  insertion_sequence_.erase(order_id);
  positions_.erase(found);
}

void QueueTracker::reject_order(const lob::core::OrderId order_id) {
  cancel_order(order_id);
}

void QueueTracker::observe_trade(const lob::strategy::StrategyMarketTrade& trade) {
  const lob::core::Side passive_side = lob::core::opposite(trade.aggressor_side);
  const TradeKey key{trade.timestamp_ns, trade.price_ticks, passive_side};
  Quantity external_quantity = trade.quantity;
  const auto own = reported_own_fills_.find(key);
  if (own != reported_own_fills_.end()) {
    const Quantity matched_fill = std::min(external_quantity, own->second);
    external_quantity -= matched_fill;
    own->second -= matched_fill;
    if (own->second == 0) reported_own_fills_.erase(own);
  }

  if (external_quantity > 0) {
    for (auto& [id, position] : positions_) {
      static_cast<void>(id);
      if (position.side != passive_side || position.price_ticks != trade.price_ticks) {
        continue;
      }
      const Quantity ahead_depletion =
          std::min(position.quantity_ahead, external_quantity);
      position.quantity_ahead -= ahead_depletion;
      const Quantity remaining =
          external_quantity - ahead_depletion;
      subtract_clamped(position.quantity_behind, remaining);
    }
  }
}

void QueueTracker::apply_cancellation(const LevelKey key, Quantity quantity) {
  for (auto& [id, position] : positions_) {
    static_cast<void>(id);
    if (position.side != key.first || position.price_ticks != key.second) continue;

    Quantity ahead_cancel = 0;
    Quantity behind_cancel = 0;
    switch (config_.cancellation_allocation) {
      case CancellationAllocation::BehindFirst:
        behind_cancel = std::min(position.quantity_behind, quantity);
        ahead_cancel = std::min(position.quantity_ahead, quantity - behind_cancel);
        break;
      case CancellationAllocation::AheadFirst:
        ahead_cancel = std::min(position.quantity_ahead, quantity);
        behind_cancel = std::min(position.quantity_behind, quantity - ahead_cancel);
        break;
      case CancellationAllocation::ProRata: {
        const Quantity total =
            checked_add(position.quantity_ahead, position.quantity_behind);
        if (total == 0) break;
        ahead_cancel = static_cast<Quantity>(
            (static_cast<long double>(quantity) *
             static_cast<long double>(position.quantity_ahead)) /
            static_cast<long double>(total));
        behind_cancel = std::min(position.quantity_behind, quantity - ahead_cancel);
        ahead_cancel = std::min(position.quantity_ahead, ahead_cancel);
        break;
      }
    }
    subtract_clamped(position.quantity_ahead, ahead_cancel);
    subtract_clamped(position.quantity_behind, behind_cancel);
  }
}

void QueueTracker::replace_order(const lob::core::OrderId old_order_id,
                                 const lob::core::OrderId new_order_id,
                                 const lob::core::Side side,
                                 const lob::core::PriceTicks price_ticks,
                                 const Quantity quantity,
                                 const MarketView& view) {
  cancel_order(old_order_id);
  add_order(new_order_id, side, price_ticks, quantity, view);
}

std::optional<QueuePosition> QueueTracker::position(
    const lob::core::OrderId order_id) const {
  const auto found = positions_.find(order_id);
  return found == positions_.end() ? std::nullopt
                                   : std::optional<QueuePosition>{found->second};
}

std::vector<QueuePosition> QueueTracker::positions() const {
  std::vector<QueuePosition> result;
  result.reserve(positions_.size());
  for (const auto& [id, position] : positions_) {
    static_cast<void>(id);
    result.push_back(position);
  }
  return result;
}

}  // namespace project3
