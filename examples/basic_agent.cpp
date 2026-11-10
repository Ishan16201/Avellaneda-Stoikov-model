#include "project3/exchange_adapter.hpp"

#include <iostream>
#include <stdexcept>

namespace {

class BasicTwoSidedAgent final : public project3::TradingAgent {
 public:
  [[nodiscard]] std::string name() const override { return "BasicTwoSidedAgent"; }

  void on_start(const project3::MarketView& view,
                project3::OrderGateway& gateway) override {
    if (view.best_bid) {
      bid_order_ = gateway.submit_limit(lob::core::Side::Buy, *view.best_bid - 1, 2);
    }
    if (view.best_ask) {
      ask_order_ = gateway.submit_limit(lob::core::Side::Sell, *view.best_ask + 1, 2);
    }
    observe(view);
  }

  void on_market_event(const project3::MarketView& view,
                       project3::OrderGateway&) override {
    observe(view);
  }

  void on_execution(const project3::ExecutionNotification& execution,
                    const project3::MarketView&,
                    project3::OrderGateway&) override {
    if (execution.filled_quantity > 0) {
      std::cout << "fill order=" << execution.order_id
                << " qty=" << execution.filled_quantity
                << " remaining=" << execution.remaining_quantity
                << " price=" << execution.price_ticks
                << " status=" << static_cast<int>(execution.status) << '\n';
    }
  }

  void on_stop(const project3::MarketView&,
               project3::OrderGateway& gateway) override {
    if (bid_order_) gateway.cancel(bid_order_);
    if (ask_order_) gateway.cancel(ask_order_);
  }

  [[nodiscard]] std::size_t intensity_observations() const noexcept {
    return intensity_observations_;
  }

 private:
  void observe(const project3::MarketView& view) {
    if (!view.hawkes_intensities.empty()) {
      last_intensity_total_ = 0.0;
      for (const double intensity : view.hawkes_intensities) {
        last_intensity_total_ += intensity;
      }
      ++intensity_observations_;
    }
  }

  lob::core::OrderId bid_order_{};
  lob::core::OrderId ask_order_{};
  std::size_t intensity_observations_{};
  double last_intensity_total_{};
};

lob::stochastic::SyntheticMarketConfig make_market() {
  lob::stochastic::SyntheticMarketConfig market;
  market.seed = 12345;
  market.horizon_seconds = 0.5;
  market.hawkes.baseline_events_per_second = {0.01, 0.01, 0.01, 0.01, 100.0, 0.01};
  market.hawkes.alpha_branching.assign(6, std::vector<double>(6, 0.01));
  market.hawkes.beta_per_second.assign(6, std::vector<double>(6, 5.0));
  market.initial_book.best_bid_ticks = 99;
  market.initial_book.best_ask_ticks = 101;
  market.initial_book.levels_per_side = 1;
  market.initial_book.orders_per_level = 1;
  market.initial_book.order_size = 1;
  market.marks.market_size = {.min_lots = 1, .max_lots = 1};
  return market;
}

}  // namespace

int main() {
  try {
    BasicTwoSidedAgent agent;
    const auto report = project3::ExchangeAdapter{}.run(agent, make_market());
    std::cout << report.to_formatted_report()
              << "\nHawkes intensity observations: "
              << agent.intensity_observations() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Project 3 demo failed: " << error.what() << '\n';
    return 1;
  }
}
