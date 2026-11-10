#include "project3/hawkes_policy.hpp"

#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

namespace {

lob::stochastic::SyntheticMarketConfig make_market(const std::uint64_t seed) {
  lob::stochastic::SyntheticMarketConfig market;
  market.seed = seed;
  market.horizon_seconds = 3.0;
  market.hawkes.baseline_events_per_second = {2.0, 1.5, 0.7, 0.7, 1.2, 1.2};
  market.hawkes.alpha_branching.assign(6, std::vector<double>(6, 0.015));
  market.hawkes.beta_per_second.assign(6, std::vector<double>(6, 3.0));
  market.initial_book.best_bid_ticks = 999;
  market.initial_book.best_ask_ticks = 1001;
  market.initial_book.levels_per_side = 3;
  market.initial_book.orders_per_level = 2;
  market.initial_book.order_size = 5;
  return market;
}

void print_summary(const char* label, const project3::PolicyRunSummary& summary) {
  std::cout << "  " << label << ": mean P&L=" << summary.mean_total_pnl
            << " sd=" << summary.std_total_pnl
            << " fill rate=" << summary.mean_fill_rate
            << " max DD=" << summary.mean_max_drawdown
            << " turnover=" << summary.mean_turnover
            << " max inventory=" << summary.mean_max_inventory << '\n';
}

}  // namespace

int main() {
  std::vector<lob::stochastic::SyntheticMarketConfig> training;
  const std::vector<double> buy_flow_scales{0.6, 0.9, 1.2, 1.6,
                                            0.7, 1.1, 1.4, 1.8};
  const std::vector<double> sell_flow_scales{1.5, 1.1, 0.8, 0.6,
                                             1.3, 0.9, 0.7, 1.2};
  for (std::size_t i = 0; i < buy_flow_scales.size(); ++i) {
    auto run = make_market(1001 + static_cast<std::uint64_t>(i));
    run.hawkes.baseline_events_per_second[4] *= buy_flow_scales[i];
    run.hawkes.baseline_events_per_second[5] *= sell_flow_scales[i];
    run.hawkes.alpha_branching[4][4] *= buy_flow_scales[i];
    run.hawkes.alpha_branching[5][5] *= sell_flow_scales[i];
    training.push_back(std::move(run));
  }

  auto unseen = make_market(0);
  unseen.hawkes.baseline_events_per_second[4] *= 1.8;
  unseen.hawkes.baseline_events_per_second[5] *= 0.7;
  unseen.hawkes.alpha_branching[4][4] *= 0.5;
  project3::PolicyComparisonScenario scenario{
      .label = "held-out-Hawkes-configuration",
      .market = unseen,
      .evaluation_seeds = {2001, 2002, 2003, 2004},
  };

  project3::AvellanedaStoikovConfig as;
  as.horizon_seconds = unseen.horizon_seconds;
  as.volatility_ticks_per_sqrt_second = 0.5;
  as.inventory_limit_lots = 5;
  as.max_quote_size_lots = 2;
  project3::HawkesAwareConfig hawkes{
      .mean_market_order_size_lots = 3.0,
  };
  project3::SymmetricMarketMakerConfig symmetric{
      .quote_offset_ticks = 1,
      .quote_size_lots = 1,
      .inventory_limit_lots = 5,
  };
  project3::HawkesDirectionFitConfig fit;
  fit.forecast_horizon_seconds = 0.25;
  lob::strategy::StrategyEvaluatorConfig evaluation;
  evaluation.timer_interval_ns = 100'000'000;

  const auto results = project3::compare_market_making_policies(
      training, {scenario}, fit, as, hawkes, symmetric, evaluation);
  for (const auto& result : results) {
    std::cout << result.scenario_label << '\n';
    print_summary("Classical A-S", result.classical_summary);
    print_summary("Hawkes-aware", result.hawkes_summary);
    print_summary("Symmetric", result.symmetric_summary);
    std::cout << "  held-out direction log loss=" << result.direction_log_loss
              << " accuracy=" << result.direction_accuracy
              << " vs base-rate log loss="
              << result.direction_base_rate_log_loss
              << " accuracy=" << result.direction_base_rate_accuracy
              << " observations=" << result.direction_observations << '\n';
  }
}
