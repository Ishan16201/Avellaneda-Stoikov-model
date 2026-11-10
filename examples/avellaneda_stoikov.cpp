#include "project3/avellaneda_stoikov.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

lob::stochastic::SyntheticMarketConfig make_market() {
  lob::stochastic::SyntheticMarketConfig market;
  market.seed = 2026;
  market.horizon_seconds = 2.0;
  market.hawkes.baseline_events_per_second = {2.0, 1.0, 0.5, 0.5, 1.0, 1.0};
  market.hawkes.alpha_branching.assign(6, std::vector<double>(6, 0.02));
  market.hawkes.beta_per_second.assign(6, std::vector<double>(6, 3.0));
  market.initial_book.best_bid_ticks = 999;
  market.initial_book.best_ask_ticks = 1001;
  market.initial_book.levels_per_side = 3;
  market.initial_book.orders_per_level = 2;
  market.initial_book.order_size = 5;
  return market;
}

}  // namespace

int main() {
  try {
    project3::AvellanedaStoikovConfig policy;
    policy.volatility_ticks_per_sqrt_second = 1.5;
    policy.risk_aversion_per_tick_lot = 0.05;
    policy.arrival_rate_per_second = 1.2;
    policy.intensity_decay_per_tick = 0.75;
    policy.horizon_seconds = 2.0;
    policy.inventory_limit_lots = 5;
    policy.base_quote_size_lots = 1;
    policy.max_quote_size_lots = 2;

    lob::strategy::StrategyEvaluatorConfig evaluation;
    evaluation.timer_interval_ns = 100'000'000;
    const auto report =
        project3::evaluate_avellaneda_stoikov(policy, make_market(), evaluation);
    std::cout << report.to_formatted_report();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Avellaneda-Stoikov demo failed: " << error.what() << '\n';
    return 1;
  }
}
