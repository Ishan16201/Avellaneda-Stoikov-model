#pragma once

#include "project3/exchange_adapter.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace project3 {

enum class MarketStressKind {
  VolatilityIncrease,
  OneSidedOrderFlowBurst,
  HawkesExcitationIncrease,
  LiquidityWithdrawal,
  CancellationSurge,
  SpreadWidening,
  PriceJumps,
};

struct MarketStressConfig final {
  double volatility_flow_multiplier{2.0};
  double one_sided_flow_multiplier{3.0};
  double hawkes_excitation_multiplier{1.5};
  double liquidity_remaining_fraction{0.5};
  double cancellation_multiplier{3.0};
  lob::core::PriceTicks spread_widening_ticks{4};
  double price_jump_size_multiplier{4.0};
};

/// Stress configurations are static post-shock regimes: the current simulator
/// does not support parameters changing at an intrarun onset time.
struct MarketStressScenario final {
  std::string label;
  MarketStressKind kind{MarketStressKind::VolatilityIncrease};
  lob::stochastic::SyntheticMarketConfig market;
  std::vector<std::uint64_t> seeds;
};

[[nodiscard]] std::vector<MarketStressScenario> make_market_stress_scenarios(
    const lob::stochastic::SyntheticMarketConfig& baseline,
    const std::vector<std::uint64_t>& seeds,
    const MarketStressConfig& stress = {});

struct DirectionalExecutionQuality final {
  /// Fill notifications and orders with at least one fill, respectively.
  std::size_t execution_count{};
  std::size_t filled_order_count{};
  double executed_quantity_lots{};
  double shortfall_quantity_lots{};
  /// Volume-weighted effective spread/shortfall in ticks per lot.
  /// Shortfall uses only fills with an observed submission midpoint.
  double effective_spread_ticks{};
  double implementation_shortfall_ticks{};
  /// Mean submit-to-first-fill time among filled orders.
  double queue_waiting_time_seconds{};
};

/// Positive AS is favorable to the market maker; negative AS is adverse.
/// Buy means the market maker bought, and Sell means it sold.
struct DirectionalMarkout final {
  /// Marked executions and fill records without a later observed mid.
  std::size_t execution_count{};
  double executed_quantity_lots{};
  std::size_t right_censored_execution_count{};
  /// Volume-weighted markout/spread/impact in ticks and AS currency per lot.
  double adverse_selection_ticks{};
  double adverse_selection_currency{};
  double realized_spread_ticks{};
  double price_impact_ticks{};
};

struct HorizonMarkout final {
  double horizon_seconds{};
  DirectionalMarkout market_maker_buys;
  DirectionalMarkout market_maker_sells;
};

struct MarketQualityEpisode final {
  std::string policy_name;
  std::string scenario_label;
  std::uint64_t seed{};
  double net_pnl_currency{};
  double gross_realized_pnl_currency{};
  double unrealized_pnl_currency{};
  /// Net fees equal charged fees less rebates.
  double fees_currency{};
  double charged_fees_currency{};
  double rebates_currency{};
  double gross_turnover_currency{};
  double spread_capture_currency{};
  double sharpe_ratio{};
  double maximum_drawdown_currency{};
  double fill_probability{};
  double cancellation_rate{};
  double queue_waiting_time_seconds{};
  double inventory_duration_lot_seconds{};
  double inventory_risk_lot_squared_seconds{};
  double mean_absolute_inventory_lots{};
  double inventory_concentration_ratio{};
  double inventory_limit_breach_duration_seconds{};
  std::size_t inventory_limit_breach_intervals{};
  double maximum_inventory_lots{};
  DirectionalExecutionQuality market_maker_buys;
  DirectionalExecutionQuality market_maker_sells;
  std::vector<HorizonMarkout> markouts;
};

struct MarketQualityScenarioSummary final {
  std::string policy_name;
  std::string scenario_label;
  std::size_t episodes{};
  double mean_net_pnl_currency{};
  double standard_deviation_net_pnl_currency{};
  double worst_net_pnl_currency{};
  double value_at_risk_loss_currency{};
  double expected_shortfall_loss_currency{};
  double mean_maximum_drawdown_currency{};
  double worst_maximum_drawdown_currency{};
  double mean_sharpe_ratio{};
  double mean_fill_probability{};
  double mean_cancellation_rate{};
  double mean_queue_waiting_time_seconds{};
  double mean_inventory_duration_lot_seconds{};
  double mean_inventory_risk_lot_squared_seconds{};
  double mean_inventory_concentration_ratio{};
  double mean_turnover_currency{};
  double mean_transaction_costs_currency{};
  double mean_spread_capture_currency{};
  double mean_adverse_selection_ticks{};
  double total_inventory_limit_breach_duration_seconds{};
  std::size_t total_inventory_limit_breach_intervals{};
  bool inventory_limit_breached{};
  DirectionalExecutionQuality market_maker_buys;
  DirectionalExecutionQuality market_maker_sells;
  std::vector<HorizonMarkout> mean_markouts;
  std::vector<MarketQualityEpisode> runs;
};

struct MarketQualityConfig final {
  /// Markouts are sampled at the first observed midprice at/after each horizon.
  std::vector<double> markout_horizons_seconds{0.1, 0.5, 1.0};
  double expected_shortfall_confidence{0.95};
  double inventory_limit_lots{10.0};
  double currency_per_tick{1.0};
  double sharpe_capital_base_currency{100'000.0};
  double annualization_seconds{365.25 * 24.0 * 60.0 * 60.0};
  /// Zero selects `std::thread::hardware_concurrency()`. Independent
  /// (policy, seed) episodes are embarrassingly parallel.
  std::size_t worker_count{0};
  lob::strategy::StrategyEvaluatorConfig evaluation{};
};

struct MarketQualityPolicy final {
  std::string name;
  std::function<std::unique_ptr<TradingAgent>()> make_agent;
};

/// AS_h = sign(MM side) * (mid(t+h) - execution price), with buy=+1,
/// sell=-1. Positive values are favorable to the market maker.
[[nodiscard]] double adverse_selection_ticks(
    lob::core::Side market_maker_side,
    double future_midprice_ticks,
    double execution_price_ticks);

/// Evaluates each policy on identical seeds for each supplied stress scenario.
/// Effective spread and implementation shortfall are from the market-maker
/// perspective; inventory/P&L metrics use the exchange broker's ledger.
[[nodiscard]] std::vector<MarketQualityScenarioSummary>
run_market_quality_study(
    const std::vector<MarketQualityPolicy>& policies,
    const std::vector<MarketStressScenario>& scenarios,
    const MarketQualityConfig& config = {});

}  // namespace project3
