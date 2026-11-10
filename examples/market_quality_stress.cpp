#include "project3/avellaneda_stoikov.hpp"
#include "project3/hawkes_policy.hpp"
#include "project3/market_quality.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

class TopOfBookAgent final : public project3::TradingAgent {
 public:
  explicit TopOfBookAgent(const lob::core::QuantityLots inventory_limit)
      : inventory_limit_(inventory_limit) {}

  [[nodiscard]] std::string name() const override {
    return "TopOfBookMarketMaker";
  }

  void on_start(const project3::MarketView& view,
                project3::OrderGateway& gateway) override {
    quote_missing_sides(view, gateway);
  }

  void on_market_event(const project3::MarketView& view,
                       project3::OrderGateway& gateway) override {
    quote_missing_sides(view, gateway);
  }

 private:
  void quote_missing_sides(const project3::MarketView& view,
                           project3::OrderGateway& gateway) const {
    const bool has_bid = std::any_of(
        view.open_orders.begin(), view.open_orders.end(), [](const auto& order) {
          return order.side == lob::core::Side::Buy;
        });
    const bool has_ask = std::any_of(
        view.open_orders.begin(), view.open_orders.end(), [](const auto& order) {
          return order.side == lob::core::Side::Sell;
        });
    if (!has_bid && view.best_bid &&
        view.inventory_lots < inventory_limit_) {
      static_cast<void>(
          gateway.submit_limit(lob::core::Side::Buy, *view.best_bid, 1));
    }
    if (!has_ask && view.best_ask &&
        view.inventory_lots > -inventory_limit_) {
      static_cast<void>(
          gateway.submit_limit(lob::core::Side::Sell, *view.best_ask, 1));
    }
  }

  lob::core::QuantityLots inventory_limit_;
};

lob::stochastic::SyntheticMarketConfig make_market() {
  lob::stochastic::SyntheticMarketConfig market;
  market.horizon_seconds = 8.0;
  market.hawkes.baseline_events_per_second = {2.0, 1.5, 0.7, 0.7, 2.0, 2.0};
  market.hawkes.alpha_branching.assign(6, std::vector<double>(6, 0.015));
  market.hawkes.beta_per_second.assign(6, std::vector<double>(6, 3.0));
  market.initial_book.best_bid_ticks = 999;
  market.initial_book.best_ask_ticks = 1001;
  market.initial_book.levels_per_side = 3;
  market.initial_book.orders_per_level = 2;
  market.initial_book.order_size = 5;
  return market;
}

void print_summary(const project3::MarketQualityScenarioSummary& result) {
  double gross_realized = 0.0;
  double unrealized = 0.0;
  double fees = 0.0;
  double charged_fees = 0.0;
  double rebates = 0.0;
  double turnover = 0.0;
  for (const auto& run : result.runs) {
    gross_realized += run.gross_realized_pnl_currency;
    unrealized += run.unrealized_pnl_currency;
    fees += run.fees_currency;
    charged_fees += run.charged_fees_currency;
    rebates += run.rebates_currency;
    turnover += run.gross_turnover_currency;
  }
  const double count = static_cast<double>(result.episodes);
  if (count > 0.0) {
    gross_realized /= count;
    unrealized /= count;
    fees /= count;
    charged_fees /= count;
    rebates /= count;
    turnover /= count;
  }
  std::cout << result.scenario_label << " / " << result.policy_name << '\n'
            << "  P&L mean=" << result.mean_net_pnl_currency
            << " (gross realized=" << gross_realized
            << ", unrealized=" << unrealized << ", net fees=" << fees
            << ", charged=" << charged_fees << ", rebates=" << rebates
            << ", turnover=" << turnover << ")\n"
            << "  Loss VaR/ES=" << result.value_at_risk_loss_currency << '/'
            << result.expected_shortfall_loss_currency
            << ", mean/worst drawdown="
            << result.mean_maximum_drawdown_currency << '/'
            << result.worst_maximum_drawdown_currency << '\n'
            << "  Fill probability=" << result.mean_fill_probability
            << ", cancellation rate=" << result.mean_cancellation_rate
            << ", mean queue wait="
            << result.mean_queue_waiting_time_seconds << " s\n"
            << "  Inventory duration="
            << result.mean_inventory_duration_lot_seconds
            << " lot-s, concentration="
            << result.mean_inventory_concentration_ratio
            << ", breach duration (summed)="
            << result.total_inventory_limit_breach_duration_seconds << " s\n"
            << "  Inventory-limit breach observed="
            << (result.inventory_limit_breached ? "yes" : "no") << '\n'
            << "  Effective spread / implementation shortfall (buy,sell)="
            << result.market_maker_buys.effective_spread_ticks << '/'
            << result.market_maker_sells.effective_spread_ticks << " / "
            << result.market_maker_buys.implementation_shortfall_ticks << '/'
            << result.market_maker_sells.implementation_shortfall_ticks
            << " ticks\n";
  if (!result.mean_markouts.empty()) {
    const auto& markout = result.mean_markouts.front();
    std::cout << "  AS@" << markout.horizon_seconds
              << " s, market-maker buy/sell="
              << markout.market_maker_buys.adverse_selection_ticks << '/'
              << markout.market_maker_sells.adverse_selection_ticks
              << " ticks (positive is favorable), realized spread="
              << markout.market_maker_buys.realized_spread_ticks << '/'
              << markout.market_maker_sells.realized_spread_ticks
              << ", price impact="
              << markout.market_maker_buys.price_impact_ticks << '/'
              << markout.market_maker_sells.price_impact_ticks << " ticks\n";
  }
}

}  // namespace

int main() {
  const auto baseline = make_market();
  const auto scenarios = project3::make_market_stress_scenarios(
      baseline, {7001, 7002, 7003});

  project3::AvellanedaStoikovConfig avellaneda;
  avellaneda.horizon_seconds = baseline.horizon_seconds;
  avellaneda.volatility_ticks_per_sqrt_second = 0.7;
  avellaneda.inventory_limit_lots = 5;
  avellaneda.max_quote_size_lots = 1;
  const std::vector<project3::MarketQualityPolicy> policies{
      {.name = "AvellanedaStoikov",
       .make_agent = [avellaneda] {
         return std::make_unique<project3::AvellanedaStoikovAgent>(
             avellaneda);
       }},
      {.name = "TopOfBookMarketMaker",
       .make_agent = [] {
         return std::make_unique<TopOfBookAgent>(5);
       }},
  };
  project3::MarketQualityConfig config;
  config.markout_horizons_seconds = {0.25, 1.0, 2.0};
  config.inventory_limit_lots = 5.0;
  config.evaluation.timer_interval_ns = 100'000'000;
  config.evaluation.broker_config.risk.max_position_lots = 10;

  const auto results =
      project3::run_market_quality_study(policies, scenarios, config);
  std::cout << std::fixed << std::setprecision(3)
            << "Post-shock market-quality stress study (paired seeds)\n"
            << "Adverse selection: side_sign * (future mid - execution price); "
               "positive favors the market maker.\n";
  for (const auto& result : results) print_summary(result);
}
