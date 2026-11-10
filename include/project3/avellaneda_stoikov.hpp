#pragma once

#include "project3/exchange_adapter.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace project3 {

class HawkesDirectionModel;

struct AvellanedaStoikovConfig final {
  double volatility_ticks_per_sqrt_second{1.0};
  double risk_aversion_per_tick_lot{0.01};
  double arrival_rate_per_second{1.0};
  double intensity_decay_per_tick{1.0};
  double horizon_seconds{60.0};
  lob::core::QuantityLots inventory_limit_lots{10};
  lob::core::QuantityLots base_quote_size_lots{1};
  lob::core::QuantityLots max_quote_size_lots{1};
  double currency_per_tick{1.0};
  double capital_base_currency{100'000.0};
};

struct AvellanedaStoikovQuotes final {
  double reservation_price_ticks{};
  double bid_offset_ticks{};
  double ask_offset_ticks{};
  lob::core::PriceTicks bid_price_ticks{};
  lob::core::PriceTicks ask_price_ticks{};
  double bid_intensity_per_second{};
  double ask_intensity_per_second{};
};

/// Classical Avellaneda-Stoikov first-order quotes for unit Poisson arrivals.
/// Volatility is in ticks/sqrt(second), inventory in lots, and gamma in
/// inverse tick-lots. Offsets are measured from the midprice.
[[nodiscard]] AvellanedaStoikovQuotes avellaneda_stoikov_quotes(
    double midprice_ticks,
    std::int64_t inventory_lots,
    double elapsed_seconds,
    const AvellanedaStoikovConfig& config);

struct IntensityComparison final {
  double theoretical_per_order_second{};
  double empirical_per_order_second{};
  double active_order_seconds{};
  double expected_fills{};
  std::size_t observed_fills{};
};

struct AvellanedaStoikovEvaluation final {
  std::string strategy_name{"AvellanedaStoikov"};
  double total_pnl_currency{};
  double spread_capture_currency{};
  double inventory_pnl_currency{};
  double fees_currency{};
  double fill_rate{};
  std::size_t submitted_orders{};
  std::size_t filled_orders{};
  std::map<std::int64_t, std::size_t> inventory_distribution;
  double maximum_drawdown_currency{};
  double sharpe_ratio{};
  double gross_turnover_currency{};
  IntensityComparison bid_intensity;
  IntensityComparison ask_intensity;
  lob::strategy::PortfolioState final_account;

  [[nodiscard]] std::string to_formatted_report() const;
};

/// Reproducible single-run baseline using the existing exchange evaluator.
[[nodiscard]] AvellanedaStoikovEvaluation evaluate_avellaneda_stoikov(
    AvellanedaStoikovConfig policy_config,
    const lob::stochastic::SyntheticMarketConfig& market_config,
    lob::strategy::StrategyEvaluatorConfig evaluation_config = {});

class AvellanedaStoikovAgent final : public TradingAgent {
 public:
  explicit AvellanedaStoikovAgent(
      AvellanedaStoikovConfig config,
      const HawkesDirectionModel* hawkes_model = nullptr,
      double mean_market_order_size_lots = 1.0);
  [[nodiscard]] std::string name() const override;
  void on_start(const MarketView&, OrderGateway&) override;
  void on_market_event(const MarketView&, OrderGateway&) override;
  void on_execution(const ExecutionNotification&,
                    const MarketView&,
                    OrderGateway&) override;
  void on_timer(const MarketView&, OrderGateway&) override;
  void on_stop(const MarketView&, OrderGateway&) override;

  [[nodiscard]] const AvellanedaStoikovEvaluation& results() const noexcept {
    return results_;
  }

 private:
  friend AvellanedaStoikovEvaluation evaluate_avellaneda_stoikov(
      AvellanedaStoikovConfig,
      const lob::stochastic::SyntheticMarketConfig&,
      lob::strategy::StrategyEvaluatorConfig);

  struct WorkingQuote final {
    lob::core::Side side{lob::core::Side::Buy};
    double theoretical_intensity_per_second{};
    lob::core::QuantityLots remaining_quantity{};
    lob::core::PriceTicks price_ticks{};
    bool cancel_requested{};
  };
  void observe(const MarketView&, bool sample_inventory);
  void refresh_quotes(const MarketView&, OrderGateway&);
  void refresh_side(const MarketView&,
                    OrderGateway&,
                    lob::core::Side,
                    lob::core::PriceTicks,
                    lob::core::QuantityLots);
  void advance_exposure(lob::core::TimestampNs timestamp_ns);
  void sample_equity(const MarketView&);

  AvellanedaStoikovConfig config_;
  const HawkesDirectionModel* hawkes_model_{};
  double mean_market_order_size_lots_{1.0};
  AvellanedaStoikovEvaluation results_;
  std::map<lob::core::OrderId, WorkingQuote> working_quotes_;
  std::map<lob::core::Side, lob::core::OrderId> side_order_ids_;
  std::map<lob::core::OrderId, bool> order_has_fill_;
  lob::core::TimestampNs last_observation_ns_{};
  std::int64_t last_inventory_lots_{};
  double last_midprice_ticks_{};
  std::vector<double> last_hawkes_intensities_;
  bool has_observation_{};
  std::vector<std::pair<lob::core::TimestampNs, double>> equity_samples_;
};

}  // namespace project3
