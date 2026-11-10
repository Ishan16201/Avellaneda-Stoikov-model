#include "project3/reduced_state_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace project3 {
namespace {

void validate(const ReducedStateConfig& config) {
  if (!std::isfinite(config.horizon_seconds) || config.horizon_seconds <= 0.0 ||
      config.time_steps == 0 || config.inventory_limit_lots <= 0 ||
      config.inventory_step_lots <= 0 ||
      config.inventory_limit_lots % config.inventory_step_lots != 0 ||
      !std::isfinite(config.terminal_inventory_penalty_currency_per_lot_squared) ||
      config.terminal_inventory_penalty_currency_per_lot_squared < 0.0 ||
      !std::isfinite(config.running_inventory_penalty_currency_per_second_lot_squared) ||
      config.running_inventory_penalty_currency_per_second_lot_squared < 0.0 ||
      !std::isfinite(config.arrival_rate_per_second) ||
      config.arrival_rate_per_second <= 0.0 ||
      !std::isfinite(config.intensity_decay_per_tick) ||
      config.intensity_decay_per_tick <= 0.0 ||
      !std::isfinite(config.min_quote_offset_ticks) ||
      config.min_quote_offset_ticks < 0.0 ||
      !std::isfinite(config.max_quote_offset_ticks) ||
      config.max_quote_offset_ticks < config.min_quote_offset_ticks ||
      !std::isfinite(config.quote_offset_step_ticks) ||
      config.quote_offset_step_ticks <= 0.0 ||
      !std::isfinite(config.currency_per_tick) ||
      config.currency_per_tick <= 0.0 ||
      !std::isfinite(config.hawkes_baseline_activity_per_second) ||
      config.hawkes_baseline_activity_per_second <= 0.0 ||
      !std::isfinite(config.hawkes_mean_reversion_per_second) ||
      config.hawkes_mean_reversion_per_second <= 0.0 ||
      !std::isfinite(config.hawkes_jump_size_per_event) ||
      config.hawkes_jump_size_per_event < 0.0 ||
      !std::isfinite(config.hawkes_activity_max_per_second) ||
      config.hawkes_activity_max_per_second <
          config.hawkes_baseline_activity_per_second ||
      config.hawkes_activity_points < 2) {
    throw std::invalid_argument{"invalid reduced-state solver configuration"};
  }
  const auto inventory_steps =
      static_cast<std::uint64_t>(config.inventory_limit_lots /
                                 config.inventory_step_lots);
  if (inventory_steps >
      static_cast<std::uint64_t>((std::numeric_limits<std::size_t>::max() - 1) /
                                 2)) {
    throw std::length_error{"reduced-state inventory grid is too large"};
  }
  const double action_intervals =
      (config.max_quote_offset_ticks - config.min_quote_offset_ticks) /
      config.quote_offset_step_ticks;
  if (!std::isfinite(action_intervals) ||
      action_intervals >
          static_cast<double>(std::numeric_limits<std::size_t>::max() - 1)) {
    throw std::length_error{"reduced-state action grid is too large"};
  }
  if (config.time_steps == std::numeric_limits<std::size_t>::max()) {
    throw std::length_error{"reduced-state time grid is too large"};
  }
}

std::pair<std::size_t, double> bracket(const std::vector<double>& grid,
                                       const double value) {
  if (grid.size() == 1 || value <= grid.front()) return {0, 0.0};
  if (value >= grid.back()) return {grid.size() - 2, 1.0};
  const auto upper = std::upper_bound(grid.begin(), grid.end(), value);
  const auto lower_index =
      static_cast<std::size_t>(std::distance(grid.begin(), upper) - 1);
  const double fraction =
      (value - grid[lower_index]) / (grid[lower_index + 1] - grid[lower_index]);
  return {lower_index, fraction};
}

template <typename Getter>
double interpolate(double time_position,
                   double activity_position,
                   double inventory_position,
                   const std::size_t time_steps,
                   const std::size_t activity_points,
                   const std::size_t inventory_points,
                   Getter&& get) {
  const auto t0 = static_cast<std::size_t>(std::floor(time_position));
  const auto t1 = std::min(time_steps, t0 + 1);
  const double ft = time_position - static_cast<double>(t0);
  const auto x0 = static_cast<std::size_t>(std::floor(activity_position));
  const auto x1 = std::min(activity_points - 1, x0 + 1);
  const double fx = activity_position - static_cast<double>(x0);
  const auto q0 = static_cast<std::size_t>(std::floor(inventory_position));
  const auto q1 = std::min(inventory_points - 1, q0 + 1);
  const double fq = inventory_position - static_cast<double>(q0);
  const auto at_time = [&](const std::size_t t) {
    const double lower =
        get(t, x0, q0) * (1.0 - fq) + get(t, x0, q1) * fq;
    const double upper =
        get(t, x1, q0) * (1.0 - fq) + get(t, x1, q1) * fq;
    return lower * (1.0 - fx) + upper * fx;
  };
  return at_time(t0) * (1.0 - ft) + at_time(t1) * ft;
}

}  // namespace

std::size_t ReducedStateSolution::value_index(
    const std::size_t time_index,
    const std::size_t activity_index,
    const std::size_t inventory_index) const noexcept {
  return (time_index * activity_grid_.size() + activity_index) *
             inventory_grid_.size() +
         inventory_index;
}

double ReducedStateSolution::value(const double elapsed_seconds,
                                   const double inventory_lots,
                                   const double hawkes_activity) const {
  if (!std::isfinite(elapsed_seconds) || !std::isfinite(inventory_lots) ||
      !std::isfinite(hawkes_activity)) {
    throw std::invalid_argument{"reduced-state interpolation input is not finite"};
  }
  const double tau = std::clamp(
      config_.horizon_seconds - elapsed_seconds, 0.0, config_.horizon_seconds);
  const double time_position =
      tau / config_.horizon_seconds * static_cast<double>(config_.time_steps);
  const double inventory_position = std::clamp(
      (inventory_lots + static_cast<double>(config_.inventory_limit_lots)) /
          static_cast<double>(config_.inventory_step_lots),
      0.0, static_cast<double>(inventory_grid_.size() - 1));
  std::size_t activity_index = 0;
  double activity_fraction = 0.0;
  if (activity_grid_.size() > 1) {
    const auto [lower, fraction] = bracket(activity_grid_, hawkes_activity);
    activity_index = lower;
    activity_fraction = fraction;
  }
  const double activity_position =
      static_cast<double>(activity_index) + activity_fraction;
  return interpolate(
      time_position, activity_position, inventory_position, config_.time_steps,
      activity_grid_.size(), inventory_grid_.size(),
      [&](const std::size_t t, const std::size_t x, const std::size_t q) {
        return values_[value_index(t, x, q)];
      });
}

ReducedStateAction ReducedStateSolution::policy(
    const double elapsed_seconds,
    const double inventory_lots,
    const double hawkes_activity) const {
  if (!std::isfinite(elapsed_seconds) || !std::isfinite(inventory_lots) ||
      !std::isfinite(hawkes_activity)) {
    throw std::invalid_argument{"reduced-state policy input is not finite"};
  }
  const double tau = std::clamp(
      config_.horizon_seconds - elapsed_seconds, 0.0, config_.horizon_seconds);
  const double time_position =
      tau / config_.horizon_seconds * static_cast<double>(config_.time_steps);
  const double inventory_position = std::clamp(
      (inventory_lots + static_cast<double>(config_.inventory_limit_lots)) /
          static_cast<double>(config_.inventory_step_lots),
      0.0, static_cast<double>(inventory_grid_.size() - 1));
  std::size_t activity_index = 0;
  double activity_fraction = 0.0;
  if (activity_grid_.size() > 1) {
    const auto [lower, fraction] = bracket(activity_grid_, hawkes_activity);
    activity_index = lower;
    activity_fraction = fraction;
  }
  const double activity_position =
      static_cast<double>(activity_index) + activity_fraction;
  const auto interpolated = [&](const std::vector<double>& policy_values) {
    return interpolate(
        time_position, activity_position, inventory_position,
        config_.time_steps, activity_grid_.size(), inventory_grid_.size(),
        [&](const std::size_t t, const std::size_t x, const std::size_t q) {
          return policy_values[value_index(t, x, q)];
        });
  };
  const auto interpolated_enabled = [&](const std::vector<std::uint8_t>& enabled) {
    const auto t = static_cast<std::size_t>(std::llround(time_position));
    const auto [x0, xf] =
        activity_grid_.size() == 1
            ? std::pair<std::size_t, double>{0, 0.0}
            : bracket(activity_grid_, hawkes_activity);
    const auto x = xf < 0.5 ? x0 : std::min(x0 + 1, activity_grid_.size() - 1);
    const auto q = static_cast<std::size_t>(std::llround(inventory_position));
    return enabled[value_index(std::min(t, config_.time_steps), x, q)] != 0;
  };
  return {
      .bid_offset_ticks = interpolated(bid_policy_),
      .ask_offset_ticks = interpolated(ask_policy_),
      .bid_enabled = interpolated_enabled(bid_enabled_),
      .ask_enabled = interpolated_enabled(ask_enabled_),
  };
}

ReducedStateSolution ReducedStateSolver::solve(const ReducedStateConfig& config) {
  validate(config);
  ReducedStateSolution solution;
  solution.config_ = config;

  const auto inventory_radius = static_cast<std::size_t>(
      config.inventory_limit_lots / config.inventory_step_lots);
  solution.inventory_grid_.reserve(2 * inventory_radius + 1);
  for (std::int64_t index = -static_cast<std::int64_t>(inventory_radius);
       index <= static_cast<std::int64_t>(inventory_radius); ++index) {
    solution.inventory_grid_.push_back(index * config.inventory_step_lots);
  }

  const double action_intervals =
      (config.max_quote_offset_ticks - config.min_quote_offset_ticks) /
      config.quote_offset_step_ticks;
  const auto action_steps =
      static_cast<std::size_t>(std::floor(action_intervals + 1.0e-10));
  solution.action_grid_.reserve(action_steps + 1);
  for (std::size_t i = 0; i <= action_steps; ++i) {
    solution.action_grid_.push_back(
        config.min_quote_offset_ticks +
        static_cast<double>(i) * config.quote_offset_step_ticks);
  }
  if (solution.action_grid_.back() < config.max_quote_offset_ticks - 1.0e-10) {
    solution.action_grid_.push_back(config.max_quote_offset_ticks);
  }

  if (config.include_hawkes_activity_state) {
    solution.activity_grid_.reserve(config.hawkes_activity_points);
    for (std::size_t i = 0; i < config.hawkes_activity_points; ++i) {
      const double fraction =
          static_cast<double>(i) /
          static_cast<double>(config.hawkes_activity_points - 1);
      solution.activity_grid_.push_back(
          fraction * config.hawkes_activity_max_per_second);
    }
  } else {
    solution.activity_grid_.push_back(config.hawkes_baseline_activity_per_second);
  }

  const auto checked_multiply = [](const std::size_t left,
                                   const std::size_t right) {
    if (right != 0 &&
        left > std::numeric_limits<std::size_t>::max() / right) {
      throw std::length_error{"reduced-state value grid size overflows"};
    }
    return left * right;
  };
  const std::size_t grid_size = checked_multiply(
      checked_multiply(config.time_steps + 1, solution.activity_grid_.size()),
      solution.inventory_grid_.size());
  solution.values_.assign(grid_size, 0.0);
  solution.bid_policy_.assign(grid_size, 0.0);
  solution.ask_policy_.assign(grid_size, 0.0);
  solution.bid_enabled_.assign(grid_size, 0);
  solution.ask_enabled_.assign(grid_size, 0);
  constexpr std::size_t bytes_per_cell =
      3 * sizeof(double) + 2 * sizeof(std::uint8_t);
  if (grid_size >
      std::numeric_limits<std::size_t>::max() / bytes_per_cell) {
    throw std::length_error{"reduced-state memory estimate overflows"};
  }
  solution.estimated_memory_bytes_ = grid_size * bytes_per_cell;

  const auto activity_count = solution.activity_grid_.size();
  const auto inventory_count = solution.inventory_grid_.size();
  const auto action_count = solution.action_grid_.size();
  for (std::size_t x = 0; x < activity_count; ++x) {
    for (std::size_t q = 0; q < inventory_count; ++q) {
      const double inventory =
          static_cast<double>(solution.inventory_grid_[q]);
      solution.values_[solution.value_index(0, x, q)] =
          -config.terminal_inventory_penalty_currency_per_lot_squared *
          inventory * inventory;
    }
  }

  const double dt =
      config.horizon_seconds / static_cast<double>(config.time_steps);
  const double dx = config.include_hawkes_activity_state
                        ? solution.activity_grid_[1] - solution.activity_grid_[0]
                        : 1.0;
  const double max_activity = config.include_hawkes_activity_state
                                  ? config.hawkes_activity_max_per_second
                                  : 0.0;
  const double max_fill_scale =
      config.include_hawkes_activity_state
          ? max_activity / config.hawkes_baseline_activity_per_second
          : 1.0;
  const double max_quote_intensity =
      config.arrival_rate_per_second * max_fill_scale *
      std::exp(-config.intensity_decay_per_tick *
               config.min_quote_offset_ticks);
  double max_transport_rate = 0.0;
  if (config.include_hawkes_activity_state) {
    for (const double activity : solution.activity_grid_) {
      max_transport_rate =
          std::max(max_transport_rate,
                   std::abs(config.hawkes_mean_reversion_per_second *
                            (config.hawkes_baseline_activity_per_second -
                             activity)) /
                       dx);
    }
  }
  const double cfl_rate =
      max_activity + 2.0 * max_quote_intensity + max_transport_rate;
  if (dt * cfl_rate > 1.0 + 1.0e-12) {
    throw std::invalid_argument{
        "explicit HJB time step violates the monotone CFL condition; increase time_steps"};
  }

  const auto interpolate_activity_grid =
      [&](const std::size_t time_index, const std::size_t q_index,
          const double activity) {
        if (activity_count == 1) {
          return solution.values_[solution.value_index(time_index, 0, q_index)];
        }
        const auto [lower, fraction] = bracket(solution.activity_grid_, activity);
        const double lower_value =
            solution.values_[solution.value_index(time_index, lower, q_index)];
        const double upper_value = solution.values_[solution.value_index(
            time_index, std::min(lower + 1, activity_count - 1), q_index)];
        return lower_value * (1.0 - fraction) + upper_value * fraction;
      };

  for (std::size_t time = 1; time <= config.time_steps; ++time) {
    const auto previous_time = time - 1;
    for (std::size_t x = 0; x < activity_count; ++x) {
      const double activity = solution.activity_grid_[x];
      const double drift =
          config.include_hawkes_activity_state
              ? config.hawkes_mean_reversion_per_second *
                    (config.hawkes_baseline_activity_per_second - activity)
              : 0.0;
      for (std::size_t q = 0; q < inventory_count; ++q) {
        const std::size_t index = solution.value_index(time, x, q);
        const std::size_t previous_index =
            solution.value_index(previous_time, x, q);
        const double inventory =
            static_cast<double>(solution.inventory_grid_[q]);
        const double current = solution.values_[previous_index];
        double flow_generator = 0.0;
        if (config.include_hawkes_activity_state) {
          const double flow_transition = interpolate_activity_grid(
              previous_time, q,
              std::min(config.hawkes_activity_max_per_second,
                       activity + config.hawkes_jump_size_per_event));
          const double upwind_derivative =
              drift > 0.0
                  ? (interpolate_activity_grid(
                         previous_time, q,
                         std::min(config.hawkes_activity_max_per_second,
                                  activity + dx)) -
                     current) /
                        dx
                  : (current -
                     interpolate_activity_grid(
                         previous_time, q,
                         std::max(0.0, activity - dx))) /
                        dx;
          flow_generator =
              drift * upwind_derivative +
              activity * (flow_transition - current);
        }
        const double bid_fill_scale =
            config.include_hawkes_activity_state
                ? activity / config.hawkes_baseline_activity_per_second
                : 1.0;
        double best_bid_hamiltonian = 0.0;
        double best_ask_hamiltonian = 0.0;
        double best_bid_offset = solution.action_grid_.front();
        double best_ask_offset = solution.action_grid_.front();
        bool bid_allowed =
            q + 1 < inventory_count;
        bool ask_allowed = q > 0;
        if (bid_allowed) {
          const double next_value =
              solution.values_[solution.value_index(previous_time, x, q + 1)];
          for (std::size_t action = 0; action < action_count; ++action) {
            const double offset = solution.action_grid_[action];
            const double rate =
                config.arrival_rate_per_second * bid_fill_scale *
                std::exp(-config.intensity_decay_per_tick * offset);
            const double hamiltonian =
                rate * (offset *
                            static_cast<double>(config.inventory_step_lots) *
                            config.currency_per_tick +
                        next_value - current);
            if (hamiltonian > best_bid_hamiltonian) {
              best_bid_hamiltonian = hamiltonian;
              best_bid_offset = offset;
            }
          }
        }
        if (ask_allowed) {
          const double next_value =
              solution.values_[solution.value_index(previous_time, x, q - 1)];
          for (std::size_t action = 0; action < action_count; ++action) {
            const double offset = solution.action_grid_[action];
            const double rate =
                config.arrival_rate_per_second * bid_fill_scale *
                std::exp(-config.intensity_decay_per_tick * offset);
            const double hamiltonian =
                rate * (offset *
                            static_cast<double>(config.inventory_step_lots) *
                            config.currency_per_tick +
                        next_value - current);
            if (hamiltonian > best_ask_hamiltonian) {
              best_ask_hamiltonian = hamiltonian;
              best_ask_offset = offset;
            }
          }
        }
        const double running_cost =
            config.running_inventory_penalty_currency_per_second_lot_squared *
            inventory * inventory;
        solution.values_[index] =
            current + dt * (flow_generator - running_cost +
                            best_bid_hamiltonian + best_ask_hamiltonian);
        if (!std::isfinite(solution.values_[index])) {
          throw std::overflow_error{"reduced-state value iteration diverged"};
        }
        solution.bid_policy_[index] = best_bid_offset;
        solution.ask_policy_[index] = best_ask_offset;
        solution.bid_enabled_[index] =
            static_cast<std::uint8_t>(bid_allowed &&
                                      best_bid_hamiltonian > 0.0);
        solution.ask_enabled_[index] =
            static_cast<std::uint8_t>(ask_allowed &&
                                      best_ask_hamiltonian > 0.0);
      }
    }
  }
  return solution;
}

std::vector<GridConvergenceRow> run_grid_convergence_study(
    const std::vector<ReducedStateConfig>& grids,
    const bool compare_to_zero_risk_avellaneda_stoikov) {
  if (grids.empty()) {
    throw std::invalid_argument{"grid convergence study requires at least one grid"};
  }
  std::vector<GridConvergenceRow> rows;
  rows.reserve(grids.size());
  double previous_value = 0.0;
  double previous_bid = 0.0;
  double previous_ask = 0.0;
  bool has_previous = false;
  for (const auto& config : grids) {
    const auto start = std::chrono::steady_clock::now();
    const auto solution = ReducedStateSolver::solve(config);
    const auto stop = std::chrono::steady_clock::now();
    const double baseline_activity =
        config.include_hawkes_activity_state
            ? config.hawkes_baseline_activity_per_second
            : 0.0;
    const double initial_value = solution.value(0.0, 0.0, baseline_activity);
    const auto initial_policy =
        solution.policy(0.0, 0.0, baseline_activity);
    double boundary_effect = 0.0;
    if (config.inventory_limit_lots <=
        std::numeric_limits<std::int64_t>::max() / 2) {
      auto wider_config = config;
      wider_config.inventory_limit_lots *= 2;
      const auto wider_solution = ReducedStateSolver::solve(wider_config);
      boundary_effect =
          std::abs(wider_solution.value(0.0, 0.0, baseline_activity) -
                   initial_value);
    }
    const auto elapsed =
        std::chrono::duration<double, std::milli>(stop - start).count();
    double analytical_error = 0.0;
    if (compare_to_zero_risk_avellaneda_stoikov &&
        config.terminal_inventory_penalty_currency_per_lot_squared == 0.0 &&
        config.running_inventory_penalty_currency_per_second_lot_squared == 0.0 &&
        !config.include_hawkes_activity_state) {
      const double zero_risk_offset =
          1.0 / config.intensity_decay_per_tick;
      analytical_error =
          std::max(std::abs(initial_policy.bid_offset_ticks - zero_risk_offset),
                   std::abs(initial_policy.ask_offset_ticks - zero_risk_offset));
    }
    rows.push_back({
        .time_steps = config.time_steps,
        .inventory_points = solution.inventory_grid_points(),
        .activity_points = solution.activity_grid_points(),
        .action_points = solution.action_grid_points(),
        .initial_value_currency = initial_value,
        .initial_bid_offset_ticks = initial_policy.bid_offset_ticks,
        .initial_ask_offset_ticks = initial_policy.ask_offset_ticks,
        .analytical_zero_risk_offset_error_ticks = analytical_error,
        .value_change_from_previous_currency =
            has_previous ? std::abs(initial_value - previous_value) : 0.0,
        .policy_change_from_previous_ticks =
            has_previous
                ? std::max(std::abs(initial_policy.bid_offset_ticks - previous_bid),
                           std::abs(initial_policy.ask_offset_ticks - previous_ask))
                : 0.0,
        .boundary_value_effect_currency = boundary_effect,
        .runtime_milliseconds = elapsed,
        .estimated_memory_bytes = solution.estimated_memory_bytes(),
    });
    previous_value = initial_value;
    previous_bid = initial_policy.bid_offset_ticks;
    previous_ask = initial_policy.ask_offset_ticks;
    has_previous = true;
  }
  return rows;
}

}  // namespace project3
