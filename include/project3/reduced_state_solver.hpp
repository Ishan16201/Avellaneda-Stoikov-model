#pragma once

#include "lob/core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace project3 {

struct ReducedStateConfig final {
  double horizon_seconds{10.0};
  std::size_t time_steps{500};
  std::int64_t inventory_limit_lots{10};
  std::int64_t inventory_step_lots{1};
  double terminal_inventory_penalty_currency_per_lot_squared{0.0};
  double running_inventory_penalty_currency_per_second_lot_squared{0.0};
  double arrival_rate_per_second{1.0};
  double intensity_decay_per_tick{1.0};
  double min_quote_offset_ticks{0.0};
  double max_quote_offset_ticks{4.0};
  double quote_offset_step_ticks{0.1};
  double currency_per_tick{1.0};
  bool include_hawkes_activity_state{false};
  double hawkes_baseline_activity_per_second{1.0};
  double hawkes_mean_reversion_per_second{1.0};
  double hawkes_jump_size_per_event{0.5};
  double hawkes_activity_max_per_second{5.0};
  std::size_t hawkes_activity_points{21};
};

struct ReducedStateAction final {
  double bid_offset_ticks{};
  double ask_offset_ticks{};
  bool bid_enabled{true};
  bool ask_enabled{true};
};

/// Relative value of active quoting over the finite horizon, measured in
/// currency units above terminal marked wealth X+qS.
class ReducedStateSolution final {
 public:
  [[nodiscard]] double value(double elapsed_seconds,
                             double inventory_lots,
                             double hawkes_activity_per_second = 0.0) const;
  [[nodiscard]] ReducedStateAction policy(
      double elapsed_seconds,
      double inventory_lots,
      double hawkes_activity_per_second = 0.0) const;

  [[nodiscard]] const ReducedStateConfig& config() const noexcept {
    return config_;
  }
  [[nodiscard]] std::size_t inventory_grid_points() const noexcept {
    return inventory_grid_.size();
  }
  [[nodiscard]] std::size_t activity_grid_points() const noexcept {
    return activity_grid_.size();
  }
  [[nodiscard]] std::size_t action_grid_points() const noexcept {
    return action_grid_.size();
  }
  [[nodiscard]] std::size_t estimated_memory_bytes() const noexcept {
    return estimated_memory_bytes_;
  }

 private:
  friend class ReducedStateSolver;
  [[nodiscard]] std::size_t value_index(std::size_t time_index,
                                        std::size_t activity_index,
                                        std::size_t inventory_index) const noexcept;

  ReducedStateConfig config_;
  std::vector<std::int64_t> inventory_grid_;
  std::vector<double> activity_grid_;
  std::vector<double> action_grid_;
  std::vector<double> values_;
  std::vector<double> bid_policy_;
  std::vector<double> ask_policy_;
  std::vector<std::uint8_t> bid_enabled_;
  std::vector<std::uint8_t> ask_enabled_;
  std::size_t estimated_memory_bytes_{};
};

class ReducedStateSolver final {
 public:
  [[nodiscard]] static ReducedStateSolution solve(
      const ReducedStateConfig& config);
};

struct GridConvergenceRow final {
  std::size_t time_steps{};
  std::size_t inventory_points{};
  std::size_t activity_points{};
  std::size_t action_points{};
  double initial_value_currency{};
  double initial_bid_offset_ticks{};
  double initial_ask_offset_ticks{};
  double analytical_zero_risk_offset_error_ticks{};
  double value_change_from_previous_currency{};
  double policy_change_from_previous_ticks{};
  double boundary_value_effect_currency{};
  double runtime_milliseconds{};
  std::size_t estimated_memory_bytes{};
};

/// Solves each supplied grid and compares successive solutions at flat
/// inventory and baseline activity. Boundary effect doubles the inventory
/// limit and measures the resulting change at the same initial state.
[[nodiscard]] std::vector<GridConvergenceRow> run_grid_convergence_study(
    const std::vector<ReducedStateConfig>& grids,
    bool compare_to_zero_risk_avellaneda_stoikov = false);

}  // namespace project3
