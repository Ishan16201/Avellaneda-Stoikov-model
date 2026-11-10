#include "project3/avellaneda_stoikov.hpp"
#include "project3/benchmark.hpp"
#include "project3/hawkes_policy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct AdaptiveMakerFeatures final {
  bool hawkes{true};
  bool queue{true};
  bool inventory_penalty{true};
  bool regime{true};
  bool dynamic_quotes{true};
};

class AdaptiveMarketMaker final : public project3::TradingAgent {
 public:
  AdaptiveMarketMaker(
      std::shared_ptr<const project3::HawkesDirectionModel> direction_model,
      AdaptiveMakerFeatures features)
      : direction_model_(std::move(direction_model)), features_(features) {}

  [[nodiscard]] std::string name() const override {
    return "HawkesForecastMarketMaker";
  }

  void on_start(const project3::MarketView& view,
                project3::OrderGateway& gateway) override {
    refresh(view, gateway);
  }

  void on_market_event(const project3::MarketView& view,
                       project3::OrderGateway& gateway) override {
    if (features_.dynamic_quotes) refresh(view, gateway);
  }

  void on_timer(const project3::MarketView& view,
                project3::OrderGateway& gateway) override {
    if (features_.dynamic_quotes) refresh(view, gateway);
  }

  void on_execution(const project3::ExecutionNotification& report,
                    const project3::MarketView&,
                    project3::OrderGateway&) override {
    using Status = lob::strategy::StrategyOrderStatus;
    if (report.status == Status::Filled || report.status == Status::Cancelled ||
        report.status == Status::Expired ||
        report.status == Status::Rejected) {
      const auto index =
          report.side == lob::core::Side::Buy ? std::size_t{0} : std::size_t{1};
      cancel_pending_[index] = false;
    }
  }

 private:
  struct Quote final {
    lob::core::PriceTicks price{};
    lob::core::QuantityLots size{};
  };

  [[nodiscard]] std::array<Quote, 2> targets(
      const project3::MarketView& view) const {
    if (!view.best_bid || !view.best_ask) return {};
    int common_shift = 0;
    int widening = 0;
    if (features_.hawkes && direction_model_) {
      const auto forecast = direction_model_->predict(view.hawkes_flow);
      common_shift = static_cast<int>(std::lround(
          forecast.expected_up_move_ticks() -
          forecast.expected_down_move_ticks()));
    }
    if (features_.regime) {
      const double total_intensity =
          std::accumulate(view.hawkes_intensities.begin(),
                          view.hawkes_intensities.end(), 0.0);
      if (total_intensity > 12.0) widening = 1;
    }
    const auto inventory =
        features_.inventory_penalty
            ? static_cast<int>(std::clamp<std::int64_t>(
                  view.inventory_lots, -5, 5))
            : 0;
    lob::core::PriceTicks bid =
        *view.best_bid + common_shift - inventory - widening;
    lob::core::PriceTicks ask =
        *view.best_ask + common_shift - inventory + widening;
    if (features_.queue) {
      for (const auto& queue : view.queue_positions) {
        if (queue.quantity_ahead <= 5) continue;
        if (queue.side == lob::core::Side::Buy) ++bid;
        else --ask;
      }
    }
    bid = std::min(bid, *view.best_ask - 1);
    ask = std::max(ask, *view.best_bid + 1);
    const bool may_buy = view.inventory_lots < 5;
    const bool may_sell = view.inventory_lots > -5;
    return {Quote{bid, may_buy ? lob::core::QuantityLots{1}
                               : lob::core::QuantityLots{0}},
            Quote{ask, may_sell ? lob::core::QuantityLots{1}
                                : lob::core::QuantityLots{0}}};
  }

  void refresh(const project3::MarketView& view,
               project3::OrderGateway& gateway) {
    if (!view.best_bid || !view.best_ask) return;
    const auto desired = targets(view);
    constexpr std::array sides{lob::core::Side::Buy, lob::core::Side::Sell};
    for (std::size_t i = 0; i < sides.size(); ++i) {
      const auto order = std::find_if(
          view.open_orders.begin(), view.open_orders.end(),
          [side = sides[i]](const lob::strategy::OpenStrategyOrder& candidate) {
            return candidate.side == side;
          });
      if (order != view.open_orders.end()) {
        if (order->price_ticks != desired[i].price &&
            desired[i].size > 0 && !cancel_pending_[i]) {
          gateway.cancel(order->client_order_id);
          cancel_pending_[i] = true;
        } else if (desired[i].size == 0 && !cancel_pending_[i]) {
          gateway.cancel(order->client_order_id);
          cancel_pending_[i] = true;
        }
        continue;
      }
      if (desired[i].size > 0 && !cancel_pending_[i]) {
        static_cast<void>(
            gateway.submit_limit(sides[i], desired[i].price, desired[i].size));
      }
    }
  }

  std::shared_ptr<const project3::HawkesDirectionModel> direction_model_;
  AdaptiveMakerFeatures features_;
  std::array<bool, 2> cancel_pending_{};
};

lob::stochastic::SyntheticMarketConfig make_market(
    const double flow_scale,
    const double excitation_scale,
    const lob::core::PriceTicks reference_price,
    const lob::core::QuantityLots book_order_size) {
  lob::stochastic::SyntheticMarketConfig market;
  market.horizon_seconds = 3.0;
  market.hawkes.baseline_events_per_second = {2.0, 1.5, 0.7, 0.7, 1.2, 1.2};
  market.hawkes.alpha_branching.assign(6, std::vector<double>(6, 0.015));
  market.hawkes.beta_per_second.assign(6, std::vector<double>(6, 3.0));
  for (auto& intensity : market.hawkes.baseline_events_per_second) {
    intensity *= flow_scale;
  }
  for (auto& row : market.hawkes.alpha_branching) {
    for (auto& value : row) value *= excitation_scale;
  }
  market.initial_book.best_bid_ticks = reference_price - 1;
  market.initial_book.best_ask_ticks = reference_price + 1;
  market.initial_book.levels_per_side = 3;
  market.initial_book.orders_per_level = 2;
  market.initial_book.order_size = book_order_size;
  market.marks.price_placement.reference_price_ticks = reference_price;
  return market;
}

project3::BenchmarkScenario make_scenario(
    const std::string& label,
    const std::string& volatility,
    const std::string& liquidity,
    const std::uint64_t first_seed,
    const double flow_scale,
    const double excitation_scale,
    const lob::core::PriceTicks reference_price,
    const lob::core::QuantityLots book_order_size,
    const bool unseen,
    const bool perturbation) {
  std::vector<std::uint64_t> seeds;
  seeds.reserve(4);
  for (std::uint64_t offset = 0; offset < 4; ++offset) {
    seeds.push_back(first_seed + offset);
  }
  return {.label = label,
          .volatility_regime = volatility,
          .liquidity_regime = liquidity,
          .market = make_market(flow_scale, excitation_scale, reference_price,
                                book_order_size),
          .seeds = std::move(seeds),
          .parameter_perturbation = perturbation,
          .unseen_configuration = unseen};
}

void print_final_results(const project3::BenchmarkExperimentResult& result) {
  std::cout << "Experiment: " << result.experiment_id << '\n'
            << "Final-test aggregate results (evaluation only):\n";
  for (const auto& summary : result.summaries) {
    if (summary.split != project3::BenchmarkSplit::FinalTest ||
        summary.scenario_label != "ALL") {
      continue;
    }
    std::cout << "  " << summary.policy_name
              << ": mean P&L=" << summary.mean_pnl_currency
              << " [" << summary.pnl_confidence_interval_low_currency << ", "
              << summary.pnl_confidence_interval_high_currency << ']'
              << ", sample SD=" << summary.pnl_standard_deviation_currency
              << ", ES=" << summary.expected_shortfall_loss_currency
              << ", Sharpe=" << summary.mean_sharpe_ratio
              << ", max drawdown=" << summary.worst_maximum_drawdown_currency
              << ", fill rate=" << summary.fill_rate
              << ", adverse selection=" << summary.adverse_selection_ticks
              << " ticks\n";
  }
  std::cout << "Artifacts:\n  " << result.manifest_path << "\n  "
            << result.runs_csv_path << "\n  " << result.summaries_csv_path
            << "\n  " << result.paired_csv_path << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    project3::BenchmarkExperimentConfig experiment;
    experiment.experiment_id = "project3-frozen-policy-benchmark-v1";
    experiment.output_prefix =
        argc > 1 ? argv[1] : "benchmark-results/project3";

    experiment.training_scenarios = {
        make_scenario("train-low-flow", "low-flow proxy", "deep book", 1001,
                      0.7, 0.8, 1000, 6, false, false),
        make_scenario("train-high-flow", "high-flow proxy", "medium book",
                      1101, 1.5, 1.2, 1010, 4, false, false)};
    experiment.validation_scenarios = {
        make_scenario("validation-medium", "medium-flow proxy", "deep book",
                      2001, 1.0, 1.0, 1020, 5, false, false),
        make_scenario("validation-bursty", "bursty-flow proxy", "thin book",
                      2101, 1.8, 1.4, 1030, 2, false, false)};
    experiment.final_test_scenarios = {
        make_scenario("test-unseen-excitation", "unseen burst proxy",
                      "thin book", 3001, 2.2, 1.8, 1100, 2, true, true),
        make_scenario("test-unseen-liquidity", "unseen flow proxy",
                      "shallow book", 3101, 2.8, 1.1, 1120, 1, true, true)};

    std::vector<lob::stochastic::SyntheticMarketConfig> training_runs;
    for (const auto& scenario : experiment.training_scenarios) {
      for (const auto seed : scenario.seeds) {
        auto training_run = scenario.market;
        training_run.seed = seed;
        training_runs.push_back(std::move(training_run));
      }
    }
    project3::HawkesDirectionFitConfig fit;
    fit.forecast_horizon_seconds = 0.25;
    fit.training_epochs = 100;
    const auto direction_model =
        std::make_shared<const project3::HawkesDirectionModel>(
            project3::HawkesDirectionModel::train(training_runs, fit));

    project3::AvellanedaStoikovConfig avellaneda;
    avellaneda.volatility_ticks_per_sqrt_second = 0.6;
    avellaneda.risk_aversion_per_tick_lot = 0.05;
    avellaneda.arrival_rate_per_second = 1.2;
    avellaneda.intensity_decay_per_tick = 0.8;
    avellaneda.horizon_seconds = 3.0;
    avellaneda.inventory_limit_lots = 5;
    avellaneda.max_quote_size_lots = 1;
    experiment.policies = {
        {.name = "classical-avellaneda-stoikov",
         .model_version = "as-v1",
         .configuration =
             "sigma=0.6 ticks/sqrt(s), gamma=0.05, A=1.2, k=0.8, "
             "horizon=3s, inventory_limit=5, max_quote_size=1",
         .make_agent = [avellaneda] {
           return std::make_unique<project3::AvellanedaStoikovAgent>(
               avellaneda);
         }},
        {.name = "hawkes-directional-maker",
         .model_version = "hawkes-softmax-v1",
         .configuration =
             "trained direction forecast (horizon=0.25s, epochs=100, "
             "learning_rate=0.05, l2=0.01) on 40 training runs; queue-aware, "
             "inventory-skewed, regime-aware dynamic quotes",
         .make_agent = [direction_model] {
           return std::make_unique<AdaptiveMarketMaker>(
               direction_model, AdaptiveMakerFeatures{});
         }},
        {.name = "symmetric-static-maker",
         .model_version = "symmetric-v1",
         .configuration =
             "no learned inputs; static best-price one-lot quotes; "
             "hard inventory cap=5",
         .make_agent = [] {
           return std::make_unique<AdaptiveMarketMaker>(
               nullptr, AdaptiveMakerFeatures{false, false, false, false,
                                              false});
         }},
    };

    constexpr std::array<std::pair<project3::AblationKind, const char*>, 5>
        ablation_definitions{{
            {project3::AblationKind::WithoutHawkesInformation, "no-hawkes"},
            {project3::AblationKind::WithoutQueueInformation, "no-queue"},
            {project3::AblationKind::WithoutInventoryPenalties,
             "no-inventory-penalty"},
            {project3::AblationKind::WithoutRegimeInformation, "no-regime"},
            {project3::AblationKind::WithoutDynamicQuoteAdjustment,
             "static-quotes"},
        }};
    for (std::size_t i = 0; i < ablation_definitions.size(); ++i) {
      AdaptiveMakerFeatures features;
      switch (ablation_definitions[i].first) {
        case project3::AblationKind::WithoutHawkesInformation:
          features.hawkes = false;
          break;
        case project3::AblationKind::WithoutQueueInformation:
          features.queue = false;
          break;
        case project3::AblationKind::WithoutInventoryPenalties:
          features.inventory_penalty = false;
          break;
        case project3::AblationKind::WithoutRegimeInformation:
          features.regime = false;
          break;
        case project3::AblationKind::WithoutDynamicQuoteAdjustment:
          features.dynamic_quotes = false;
          break;
      }
      const std::string name = ablation_definitions[i].second;
      experiment.ablations.push_back(
          {.kind = ablation_definitions[i].first,
           .baseline_policy_name = "hawkes-directional-maker",
           .ablated_policy =
               {.name = name,
                .model_version = "hawkes-ablation-v1",
                .configuration = "disabled feature: " + name,
                .make_agent = [direction_model, features] {
                  return std::make_unique<AdaptiveMarketMaker>(
                      direction_model, features);
                }}});
    }

    lob::strategy::StrategyEvaluatorConfig evaluation;
    evaluation.timer_interval_ns = 100'000'000;
    experiment.market_quality.evaluation = evaluation;
    experiment.market_quality.markout_horizons_seconds = {0.25, 1.0};
    experiment.primary_adverse_selection_horizon_seconds = 0.25;
    const auto result = project3::run_market_making_benchmark(experiment);
    print_final_results(result);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Benchmark experiment failed: " << error.what() << '\n';
    return 1;
  }
}
