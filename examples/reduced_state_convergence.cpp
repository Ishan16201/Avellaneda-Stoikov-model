#include "project3/reduced_state_solver.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

int main() {
  project3::ReducedStateConfig coarse;
  coarse.horizon_seconds = 2.0;
  coarse.time_steps = 250;
  coarse.inventory_limit_lots = 4;
  coarse.arrival_rate_per_second = 0.5;
  coarse.intensity_decay_per_tick = 1.0;
  coarse.min_quote_offset_ticks = 0.0;
  coarse.max_quote_offset_ticks = 3.0;
  coarse.quote_offset_step_ticks = 0.4;

  auto medium = coarse;
  medium.time_steps = 500;
  medium.inventory_limit_lots = 6;
  medium.quote_offset_step_ticks = 0.2;

  auto fine = medium;
  fine.time_steps = 1000;
  fine.inventory_limit_lots = 8;
  fine.quote_offset_step_ticks = 0.1;

  const auto rows =
      project3::run_grid_convergence_study({coarse, medium, fine}, true);
  std::cout << std::fixed << std::setprecision(6)
            << "time_steps,inventory_points,action_points,value,delta_value,"
               "bid,ask,policy_change,zero_risk_offset_error,"
               "boundary_effect,runtime_ms,memory_bytes\n";
  for (const auto& row : rows) {
    std::cout << row.time_steps << ',' << row.inventory_points << ','
              << row.action_points << ',' << row.initial_value_currency << ','
              << row.value_change_from_previous_currency << ','
              << row.initial_bid_offset_ticks << ','
              << row.initial_ask_offset_ticks << ','
              << row.policy_change_from_previous_ticks << ','
              << row.analytical_zero_risk_offset_error_ticks << ','
              << row.boundary_value_effect_currency << ','
              << row.runtime_milliseconds << ',' << row.estimated_memory_bytes
              << '\n';
  }
}
