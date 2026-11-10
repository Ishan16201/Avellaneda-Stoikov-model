#include "project3/avellaneda_stoikov.hpp"
#include "project3/hawkes_policy.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace project3 {
namespace {

void validate_config(const AvellanedaStoikovConfig& config) {
  if (!std::isfinite(config.volatility_ticks_per_sqrt_second) ||
      config.volatility_ticks_per_sqrt_second < 0.0 ||
      !std::isfinite(config.risk_aversion_per_tick_lot) ||
      config.risk_aversion_per_tick_lot < 0.0 ||
      !std::isfinite(config.arrival_rate_per_second) ||
      config.arrival_rate_per_second <= 0.0 ||
      !std::isfinite(config.intensity_decay_per_tick) ||
      config.intensity_decay_per_tick <= 0.0 ||
      !std::isfinite(config.horizon_seconds) || config.horizon_seconds <= 0.0 ||
      config.inventory_limit_lots <= 0 ||
      config.base_quote_size_lots <= 0 ||
      config.max_quote_size_lots <= 0 ||
      config.base_quote_size_lots > config.max_quote_size_lots ||
      config.inventory_limit_lots >
          static_cast<lob::core::QuantityLots>(
              std::numeric_limits<std::int64_t>::max()) ||
      config.max_quote_size_lots >
          static_cast<lob::core::QuantityLots>(
              std::numeric_limits<std::int64_t>::max()) ||
      !std::isfinite(config.currency_per_tick) || config.currency_per_tick <= 0.0 ||
      !std::isfinite(config.capital_base_currency) ||
      config.capital_base_currency <= 0.0) {
    throw std::invalid_argument{"invalid Avellaneda-Stoikov configuration"};
  }
}

lob::core::PriceTicks rounded_price(const double price, const bool round_up) {
  if (!std::isfinite(price)) {
    throw std::overflow_error{"Avellaneda-Stoikov quote price is not finite"};
  }
  const long double rounded = round_up
                                  ? std::ceil(static_cast<long double>(price))
                                  : std::floor(static_cast<long double>(price));
  const long double min_price = static_cast<long double>(
      std::numeric_limits<lob::core::PriceTicks>::min());
  const long double max_price_exclusive = -min_price;
  if (rounded < min_price || rounded >= max_price_exclusive) {
    throw std::overflow_error{"Avellaneda-Stoikov quote price is out of range"};
  }
  return static_cast<lob::core::PriceTicks>(rounded);
}

IntensityComparison intensity_comparison(const double expected,
                                         const double exposure_seconds,
                                         const std::size_t fills) {
  return {
      .theoretical_per_order_second =
          exposure_seconds > 0.0 ? expected / exposure_seconds : 0.0,
      .empirical_per_order_second =
          exposure_seconds > 0.0 ? static_cast<double>(fills) / exposure_seconds : 0.0,
      .active_order_seconds = exposure_seconds,
      .expected_fills = expected,
      .observed_fills = fills,
  };
}

double modeled_intensity(const double arrival_rate,
                         const double decay,
                         const double offset) {
  const double exponent = -decay * offset;
  const double log_intensity = std::log(arrival_rate) + exponent;
  if (std::isnan(log_intensity) ||
      log_intensity > std::log(std::numeric_limits<double>::max())) {
    throw std::overflow_error{"modeled quote intensity is out of range"};
  }
  if (log_intensity <
      std::log(std::numeric_limits<double>::denorm_min())) {
    return 0.0;
  }
  return std::exp(log_intensity);
}

}  // namespace

AvellanedaStoikovQuotes avellaneda_stoikov_quotes(
    const double midprice_ticks,
    const std::int64_t inventory_lots,
    const double elapsed_seconds,
    const AvellanedaStoikovConfig& config) {
  validate_config(config);
  if (!std::isfinite(midprice_ticks) || !std::isfinite(elapsed_seconds) ||
      elapsed_seconds < 0.0) {
    throw std::invalid_argument{"invalid Avellaneda-Stoikov quote input"};
  }

  const double tau = std::max(0.0, config.horizon_seconds - elapsed_seconds);
  const double gamma = config.risk_aversion_per_tick_lot;
  const double variance_horizon =
      config.volatility_ticks_per_sqrt_second *
      config.volatility_ticks_per_sqrt_second * tau;
  const double risk_term = gamma * variance_horizon;
  const double center =
      midprice_ticks - static_cast<double>(inventory_lots) * risk_term;
  const double gamma_decay_ratio = gamma / config.intensity_decay_per_tick;
  const double arrival_half_spread =
      gamma == 0.0 || gamma_decay_ratio < 1.0e-6
          ? (1.0 / config.intensity_decay_per_tick) *
                (1.0 - 0.5 * gamma_decay_ratio +
                 gamma_decay_ratio * gamma_decay_ratio / 3.0)
          : std::log1p(gamma_decay_ratio) / gamma;
  const double bid_offset =
      arrival_half_spread + (0.5 + static_cast<double>(inventory_lots)) * risk_term;
  const double ask_offset =
      arrival_half_spread + (0.5 - static_cast<double>(inventory_lots)) * risk_term;
  const auto bid = rounded_price(midprice_ticks - bid_offset, false);
  const auto ask = rounded_price(midprice_ticks + ask_offset, true);
  const double actual_bid_offset = midprice_ticks - static_cast<double>(bid);
  const double actual_ask_offset = static_cast<double>(ask) - midprice_ticks;

  return {
      .reservation_price_ticks = center,
      .bid_offset_ticks = bid_offset,
      .ask_offset_ticks = ask_offset,
      .bid_price_ticks = bid,
      .ask_price_ticks = ask,
      .bid_intensity_per_second = modeled_intensity(
          config.arrival_rate_per_second, config.intensity_decay_per_tick,
          actual_bid_offset),
      .ask_intensity_per_second = modeled_intensity(
          config.arrival_rate_per_second, config.intensity_decay_per_tick,
          actual_ask_offset),
  };
}

std::string AvellanedaStoikovEvaluation::to_formatted_report() const {
  std::ostringstream out;
  out << std::fixed << std::setprecision(6)
      << "Avellaneda-Stoikov Baseline Evaluation\n"
      << "  Total P&L:                 " << total_pnl_currency << '\n'
      << "  Spread capture:            " << spread_capture_currency << '\n'
      << "  Inventory P&L (residual):  " << inventory_pnl_currency << '\n'
      << "  Net fees:                  " << fees_currency << '\n'
      << "  Quote fill rate:           " << fill_rate << '\n'
      << "  Filled / submitted orders: " << filled_orders << " / "
      << submitted_orders << '\n'
      << "  Maximum drawdown:          " << maximum_drawdown_currency << '\n'
      << "  Annualized Sharpe:         " << sharpe_ratio << '\n'
      << "  Gross turnover:            " << gross_turnover_currency << '\n'
      << "  Inventory distribution:    ";
  if (inventory_distribution.empty()) {
    out << "no samples\n";
  } else {
    bool first = true;
    for (const auto& [inventory, count] : inventory_distribution) {
      if (!first) out << ", ";
      out << inventory << ':' << count;
      first = false;
    }
    out << " (lots:samples)\n";
  }
  out << "  Bid intensity (theory/emp): "
      << bid_intensity.theoretical_per_order_second << " / "
      << bid_intensity.empirical_per_order_second << " fills/order-second\n"
      << "  Ask intensity (theory/emp): "
      << ask_intensity.theoretical_per_order_second << " / "
      << ask_intensity.empirical_per_order_second << " fills/order-second\n";
  return out.str();
}

AvellanedaStoikovAgent::AvellanedaStoikovAgent(
    AvellanedaStoikovConfig config,
    const HawkesDirectionModel* hawkes_model,
    const double mean_market_order_size_lots)
    : config_(std::move(config)),
      hawkes_model_(hawkes_model),
      mean_market_order_size_lots_(mean_market_order_size_lots) {
  validate_config(config_);
  if (hawkes_model_ != nullptr &&
      (!std::isfinite(mean_market_order_size_lots_) ||
       mean_market_order_size_lots_ <= 0.0)) {
    throw std::invalid_argument{"invalid mean market-order size"};
  }
  results_.strategy_name = name();
}

std::string AvellanedaStoikovAgent::name() const {
  return hawkes_model_ == nullptr ? "AvellanedaStoikov"
                                  : "HawkesAwareAvellanedaStoikov";
}

void AvellanedaStoikovAgent::advance_exposure(
    const lob::core::TimestampNs timestamp_ns) {
  if (!has_observation_) {
    last_observation_ns_ = timestamp_ns;
    has_observation_ = true;
    return;
  }
  if (timestamp_ns < last_observation_ns_) {
    throw std::logic_error{"Avellaneda-Stoikov observations moved backward in time"};
  }
  const double elapsed =
      static_cast<double>(timestamp_ns - last_observation_ns_) * 1.0e-9;
  for (const auto& [id, quote] : working_quotes_) {
    static_cast<void>(id);
    if (quote.remaining_quantity <= 0) continue;
    auto& intensity = quote.side == lob::core::Side::Buy
                          ? results_.bid_intensity
                          : results_.ask_intensity;
    intensity.active_order_seconds += elapsed;
    intensity.expected_fills +=
        quote.theoretical_intensity_per_second * elapsed;
  }
  last_observation_ns_ = timestamp_ns;
}

void AvellanedaStoikovAgent::sample_equity(const MarketView& view) {
  if (!equity_samples_.empty() &&
      equity_samples_.back().first == view.timestamp_ns) {
    equity_samples_.back().second = view.market_maker.account.marked_equity_currency;
    return;
  }
  equity_samples_.emplace_back(view.timestamp_ns,
                               view.market_maker.account.marked_equity_currency);
  ++results_.inventory_distribution[view.inventory_lots];
}

void AvellanedaStoikovAgent::observe(const MarketView& view,
                                     const bool sample_inventory) {
  advance_exposure(view.timestamp_ns);
  results_.final_account = view.market_maker.account;
  last_inventory_lots_ = view.inventory_lots;
  if (view.midprice_ticks) last_midprice_ticks_ = *view.midprice_ticks;
  last_hawkes_intensities_ = view.hawkes_intensities;
  if (sample_inventory) sample_equity(view);
}

void AvellanedaStoikovAgent::on_start(const MarketView& view,
                                      OrderGateway& gateway) {
  observe(view, true);
  refresh_quotes(view, gateway);
}

void AvellanedaStoikovAgent::on_market_event(const MarketView& view,
                                             OrderGateway& gateway) {
  const auto old_inventory = last_inventory_lots_;
  const double old_midprice = last_midprice_ticks_;
  const auto old_intensities = last_hawkes_intensities_;
  observe(view, false);
  if (view.inventory_lots != old_inventory ||
      (view.midprice_ticks && *view.midprice_ticks != old_midprice) ||
      (hawkes_model_ != nullptr &&
       view.hawkes_intensities != old_intensities)) {
    refresh_quotes(view, gateway);
  }
}

void AvellanedaStoikovAgent::on_execution(const ExecutionNotification& report,
                                           const MarketView& view,
                                           OrderGateway&) {
  observe(view, false);
  const auto it = working_quotes_.find(report.order_id);
  if (report.filled_quantity > 0) {
    if (order_has_fill_.emplace(report.order_id, true).second) {
      ++results_.filled_orders;
    }
    results_.spread_capture_currency -= report.execution_cost_currency;
    if (it != working_quotes_.end()) {
      auto& intensity = report.side == lob::core::Side::Buy
                            ? results_.bid_intensity
                            : results_.ask_intensity;
      ++intensity.observed_fills;
      it->second.remaining_quantity = report.remaining_quantity;
    }
  }
  if (report.status == lob::strategy::StrategyOrderStatus::Filled ||
      report.status == lob::strategy::StrategyOrderStatus::Cancelled ||
      report.status == lob::strategy::StrategyOrderStatus::Rejected ||
      report.status == lob::strategy::StrategyOrderStatus::Expired) {
    if (it != working_quotes_.end()) {
      side_order_ids_.erase(it->second.side);
      working_quotes_.erase(it);
    }
  }
}

void AvellanedaStoikovAgent::on_timer(const MarketView& view,
                                      OrderGateway& gateway) {
  observe(view, true);
  refresh_quotes(view, gateway);
}

void AvellanedaStoikovAgent::on_stop(const MarketView& view,
                                     OrderGateway& gateway) {
  observe(view, false);
  for (auto& [id, quote] : working_quotes_) {
    if (!quote.cancel_requested) {
      gateway.cancel(id);
      quote.cancel_requested = true;
    }
  }
}

void AvellanedaStoikovAgent::refresh_quotes(const MarketView& view,
                                             OrderGateway& gateway) {
  if (!view.midprice_ticks || view.timestamp_ns < 0) return;
  const double elapsed = static_cast<double>(view.timestamp_ns) * 1.0e-9;
  auto quotes = avellaneda_stoikov_quotes(
      *view.midprice_ticks, view.inventory_lots, elapsed, config_);
  if (hawkes_model_ != nullptr && view.hawkes_flow.available) {
    const auto forecast = hawkes_model_->predict(view.hawkes_flow);
    const auto fill_probability = [&](const lob::core::Side side,
                                      const lob::core::PriceTicks price) {
      const auto order_id = side_order_ids_.find(side);
      double quantity_ahead = 0.0;
      double own_quantity =
          static_cast<double>(config_.base_quote_size_lots);
      if (order_id != side_order_ids_.end()) {
        for (const auto& queue : view.queue_positions) {
          if (queue.order_id == order_id->second) {
            quantity_ahead = static_cast<double>(queue.quantity_ahead);
            own_quantity = static_cast<double>(queue.own_remaining);
            break;
          }
        }
      } else {
        const auto& levels =
            side == lob::core::Side::Buy ? view.bids : view.asks;
        for (const auto& level : levels) {
          if (level.price_ticks == price) {
            quantity_ahead = static_cast<double>(level.quantity);
            break;
          }
        }
      }
      const double aggressive_intensity =
          side == lob::core::Side::Buy ? view.hawkes_flow.market_sell
                                       : view.hawkes_flow.market_buy;
      const double queue_lots =
          quantity_ahead + 0.5 * std::max(1.0, own_quantity);
      const double expected_depletion =
          aggressive_intensity * mean_market_order_size_lots_ *
          hawkes_model_->forecast_horizon_seconds();
      return -std::expm1(-expected_depletion / queue_lots);
    };
    const double bid_markout =
        forecast.expected_down_move_ticks() *
        fill_probability(lob::core::Side::Buy, quotes.bid_price_ticks);
    const double ask_markout =
        forecast.expected_up_move_ticks() *
        fill_probability(lob::core::Side::Sell, quotes.ask_price_ticks);
    quotes.bid_price_ticks = rounded_price(
        static_cast<double>(quotes.bid_price_ticks) - bid_markout, false);
    quotes.ask_price_ticks = rounded_price(
        static_cast<double>(quotes.ask_price_ticks) + ask_markout, true);
  }
  const std::int64_t limit =
      static_cast<std::int64_t>(config_.inventory_limit_lots);
  const std::int64_t position = view.inventory_lots;
  auto bid_size = position >= limit ? 0 : config_.base_quote_size_lots;
  auto ask_size = position <= -limit ? 0 : config_.base_quote_size_lots;
  if (view.best_ask && quotes.bid_price_ticks >= *view.best_ask) {
    if (*view.best_ask == std::numeric_limits<lob::core::PriceTicks>::min()) {
      bid_size = 0;
    } else {
      quotes.bid_price_ticks = *view.best_ask - 1;
      quotes.bid_intensity_per_second = modeled_intensity(
          config_.arrival_rate_per_second, config_.intensity_decay_per_tick,
          *view.midprice_ticks - static_cast<double>(quotes.bid_price_ticks));
    }
  }
  if (view.best_bid && quotes.ask_price_ticks <= *view.best_bid) {
    if (*view.best_bid == std::numeric_limits<lob::core::PriceTicks>::max()) {
      ask_size = 0;
    } else {
      quotes.ask_price_ticks = *view.best_bid + 1;
      quotes.ask_intensity_per_second = modeled_intensity(
          config_.arrival_rate_per_second, config_.intensity_decay_per_tick,
          static_cast<double>(quotes.ask_price_ticks) - *view.midprice_ticks);
    }
  }

  refresh_side(view, gateway, lob::core::Side::Buy,
               quotes.bid_price_ticks, bid_size);
  refresh_side(view, gateway, lob::core::Side::Sell,
               quotes.ask_price_ticks, ask_size);
}

void AvellanedaStoikovAgent::refresh_side(
    const MarketView& view,
    OrderGateway& gateway,
    const lob::core::Side side,
    const lob::core::PriceTicks price,
    const lob::core::QuantityLots requested_size) {
  const auto current = side_order_ids_.find(side);
  if (current != side_order_ids_.end()) {
    const auto quote_it = working_quotes_.find(current->second);
    if (quote_it != working_quotes_.end()) {
      if (quote_it->second.cancel_requested) return;
      if (quote_it->second.price_ticks == price && requested_size > 0) return;
      gateway.cancel(current->second);
      quote_it->second.cancel_requested = true;
      return;
    }
    side_order_ids_.erase(current);
  }
  if (requested_size <= 0) return;

  const auto maximum_signed =
      static_cast<std::int64_t>(config_.max_quote_size_lots);
  std::int64_t reserved = 0;
  for (const auto& [id, quote] : working_quotes_) {
    static_cast<void>(id);
    if (quote.side == side && reserved < maximum_signed) {
      const auto remaining_capacity =
          static_cast<lob::core::QuantityLots>(maximum_signed - reserved);
      reserved += static_cast<std::int64_t>(
          std::min(quote.remaining_quantity, remaining_capacity));
    }
  }
  const std::int64_t limit =
      static_cast<std::int64_t>(config_.inventory_limit_lots);
  const std::int64_t position = view.inventory_lots;
  if (position > limit || position < -limit) return;
  std::int64_t capacity = maximum_signed;
  if (side == lob::core::Side::Buy) {
    if (position >= 0) {
      capacity = std::min(maximum_signed, limit - position);
    } else {
      const auto deficit = -position;
      if (limit < maximum_signed && deficit < maximum_signed - limit) {
        capacity = limit + deficit;
      }
    }
  } else if (position <= 0) {
    capacity = std::min(maximum_signed, limit + position);
  } else if (limit < maximum_signed && position < maximum_signed - limit) {
    capacity = limit + position;
  }
  const std::int64_t available =
      capacity > reserved ? capacity - reserved : 0;
  const auto requested_signed = static_cast<std::int64_t>(requested_size);
  const auto quantity = static_cast<lob::core::QuantityLots>(
      std::min<std::int64_t>(
          {requested_signed, maximum_signed, available}));
  if (quantity <= 0) return;

  const auto order_id = gateway.submit_limit(side, price, quantity);
  const double offset =
      side == lob::core::Side::Buy
          ? *view.midprice_ticks - static_cast<double>(price)
          : static_cast<double>(price) - *view.midprice_ticks;
  const double intensity = modeled_intensity(
      config_.arrival_rate_per_second, config_.intensity_decay_per_tick, offset);
  working_quotes_.emplace(order_id, WorkingQuote{
      .side = side,
      .theoretical_intensity_per_second = intensity,
      .remaining_quantity = quantity,
      .price_ticks = price,
      .cancel_requested = false,
  });
  side_order_ids_[side] = order_id;
  ++results_.submitted_orders;
}

AvellanedaStoikovEvaluation evaluate_avellaneda_stoikov(
    AvellanedaStoikovConfig policy_config,
    const lob::stochastic::SyntheticMarketConfig& market_config,
    lob::strategy::StrategyEvaluatorConfig evaluation_config) {
  validate_config(policy_config);
  if (evaluation_config.timer_interval_ns == 0) {
    evaluation_config.timer_interval_ns = 1'000'000'000;
  }
  evaluation_config.broker_config.currency_per_tick =
      policy_config.currency_per_tick;
  evaluation_config.broker_config.risk.max_position_lots =
      std::min(evaluation_config.broker_config.risk.max_position_lots,
               policy_config.inventory_limit_lots);
  evaluation_config.broker_config.risk.max_order_lots =
      std::min(evaluation_config.broker_config.risk.max_order_lots,
               policy_config.max_quote_size_lots);

  AvellanedaStoikovAgent agent{policy_config};
  const auto report =
      ExchangeAdapter{}.run(agent, market_config, evaluation_config);
  auto results = agent.results();
  results.final_account = report.final_portfolio;
  results.total_pnl_currency = report.final_portfolio.net_total_pnl_currency();
  results.fees_currency = report.final_portfolio.total_fees_currency;
  results.maximum_drawdown_currency =
      report.final_portfolio.max_drawdown_currency;
  results.gross_turnover_currency =
      report.final_portfolio.gross_turnover_currency;
  const double gross_pnl = report.final_portfolio.gross_realized_pnl_currency +
                           report.final_portfolio.unrealized_pnl_currency;
  results.inventory_pnl_currency =
      gross_pnl - results.spread_capture_currency;
  results.fill_rate =
      results.submitted_orders == 0
          ? 0.0
          : static_cast<double>(results.filled_orders) /
                static_cast<double>(results.submitted_orders);
  results.bid_intensity = intensity_comparison(
      results.bid_intensity.expected_fills,
      results.bid_intensity.active_order_seconds,
      results.bid_intensity.observed_fills);
  results.ask_intensity = intensity_comparison(
      results.ask_intensity.expected_fills,
      results.ask_intensity.active_order_seconds,
      results.ask_intensity.observed_fills);

  std::vector<double> returns;
  returns.reserve(agent.equity_samples_.size());
  const auto& samples = agent.equity_samples_;
  for (std::size_t i = 1; i < samples.size(); ++i) {
    const double previous_equity =
        policy_config.capital_base_currency + samples[i - 1].second;
    const double current_equity =
        policy_config.capital_base_currency + samples[i].second;
    if (previous_equity > 0.0) {
      returns.push_back(current_equity / previous_equity - 1.0);
    }
  }
  if (returns.size() > 1) {
    double mean = 0.0;
    for (const double value : returns) mean += value;
    mean /= static_cast<double>(returns.size());
    double squared_error = 0.0;
    for (const double value : returns) {
      const double difference = value - mean;
      squared_error += difference * difference;
    }
    const double sample_std =
        std::sqrt(squared_error / static_cast<double>(returns.size() - 1));
    const double elapsed_seconds =
        static_cast<double>(samples.back().first - samples.front().first) * 1.0e-9;
    if (sample_std > 0.0 && elapsed_seconds > 0.0) {
      constexpr double seconds_per_year = 365.25 * 24.0 * 60.0 * 60.0;
      const double observations_per_second =
          static_cast<double>(returns.size()) / elapsed_seconds;
      results.sharpe_ratio =
          mean / sample_std *
          std::sqrt(observations_per_second * seconds_per_year);
    }
  }
  return results;
}

}  // namespace project3
