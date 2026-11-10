#include "project3/policy_optimization.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

lob::stochastic::SyntheticMarketConfig make_market(
    const std::uint64_t seed, const double buy_flow, const double sell_flow) {
  lob::stochastic::SyntheticMarketConfig market;
  market.seed = seed;
  market.horizon_seconds = 3.0;
  market.hawkes.baseline_events_per_second = {2.0, 1.5, 0.7, 0.7,
                                              8.0 * buy_flow,
                                              8.0 * sell_flow};
  market.hawkes.alpha_branching.assign(6, std::vector<double>(6, 0.015));
  market.hawkes.beta_per_second.assign(6, std::vector<double>(6, 3.0));
  market.hawkes.alpha_branching[4][4] *= buy_flow;
  market.hawkes.alpha_branching[5][5] *= sell_flow;
  market.marks.market_size.min_lots = 1;
  market.marks.market_size.max_lots = 4;
  market.initial_book.best_bid_ticks = 999;
  market.initial_book.best_ask_ticks = 1001;
  market.initial_book.levels_per_side = 2;
  market.initial_book.orders_per_level = 1;
  market.initial_book.order_size = 2;
  return market;
}

void print_evaluation(const project3::PolicyEvaluationSummary& row) {
  std::cout << "  " << row.scenario_label << " / " << row.policy_name
            << ": mean OOS P&L=" << row.mean_net_pnl_currency
            << " mean objective reward=" << row.mean_objective_reward_currency
            << " sd=" << row.std_net_pnl_currency
            << " fill rate=" << row.mean_fill_rate
            << " max drawdown=" << row.mean_max_drawdown_currency << '\n';
}

}  // namespace

int main() {
  project3::PolicyOptimizationStudyConfig config;
  config.training_scenarios = {
      {.label = "train-balanced",
       .market = make_market(11, 1.0, 1.0),
       .seeds = {1001, 1002}},
      {.label = "train-buy-heavy",
       .market = make_market(12, 1.4, 0.8),
       .seeds = {1101, 1102}},
  };
  config.validation_scenarios = {
      {.label = "validation-sell-heavy",
       .market = make_market(13, 0.8, 1.3),
       .seeds = {1201, 1202}},
  };
  auto wider_mark_scenario = make_market(15, 1.0, 1.0);
  wider_mark_scenario.marks.price_placement.max_offset_ticks += 3;
  config.out_of_sample_scenarios = {
      {.label = "OOS-high-flow",
       .market = make_market(14, 1.8, 1.6),
       .seeds = {1301, 1302}},
      {.label = "OOS-wider-mark-placement",
       .market = wider_mark_scenario,
       .seeds = {1401, 1402}},
  };

  config.policy_gradient.epochs = 4;
  config.policy_gradient.episodes_per_epoch = 4;
  config.policy_gradient.validation_patience = 3;
  config.policy_gradient.decision_interval_seconds = 0.25;
  config.policy_gradient.seed = 8128;
  config.policy_gradient.checkpoint_path =
      "build/policy-gradient-checkpoint.txt";
  config.policy_gradient.learning_curve_csv_path =
      "build/policy-gradient-learning.csv";

  config.surrogate.epochs = 5;
  config.surrogate.validation_patience = 3;
  config.surrogate.seed = 9341;
  config.surrogate.checkpoint_path =
      "build/surrogate-policy-checkpoint.txt";
  config.surrogate.learning_curve_csv_path =
      "build/surrogate-policy-learning.csv";

  config.avellaneda_stoikov.horizon_seconds = 3.0;
  config.avellaneda_stoikov.volatility_ticks_per_sqrt_second = 0.5;
  config.avellaneda_stoikov.inventory_limit_lots = 4;
  config.avellaneda_stoikov.max_quote_size_lots = 1;
  config.dynamic_programming.horizon_seconds = 3.0;
  config.dynamic_programming.time_steps = 300;
  config.dynamic_programming.inventory_limit_lots = 4;
  config.dynamic_programming.quote_offset_step_ticks = 0.5;

  config.policy_gradient.evaluation.timer_interval_ns = 100'000'000;

  const auto study = project3::run_policy_optimization_study(config);
  std::cout << std::fixed << std::setprecision(4)
            << "Policy optimization study\n"
            << "  REINFORCE best epoch: "
            << study.policy_gradient.best_epoch
            << " / " << study.policy_gradient.completed_epochs
            << ", training reward at selected policy: "
            << study.policy_gradient.training_reward_currency
            << ", validation reward: "
            << study.policy_gradient.best_validation_reward_currency << '\n'
            << "  Surrogate best epoch: " << study.surrogate.best_epoch
            << " / " << study.surrogate.learning_curve.size()
            << ", smooth training objective: "
            << study.surrogate.training_reward_currency
            << ", exact-market train reward: "
            << study.surrogate.training_exact_market_reward_currency
            << ", exact-market validation reward: "
            << study.surrogate.validation_exact_market_reward_currency << '\n'
            << "  Exact-market out-of-sample comparisons:\n";
  for (const auto& row : study.out_of_sample) {
    print_evaluation(row);
  }
}
