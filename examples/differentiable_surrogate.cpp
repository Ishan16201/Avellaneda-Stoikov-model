#include "project3/differentiable_surrogate.hpp"

#include <iomanip>
#include <iostream>

int main() {
  const auto paths =
      project3::make_surrogate_training_batch(8128, 128, 40);
  project3::SurrogateParameters initial{
      .values = {-0.5, -0.5, 0.1, 0.0, 0.0, 0.0}};
  const auto gradient_check = project3::check_surrogate_gradient(
      initial, paths, {}, 1.0e-5, 1.0e-12);
  project3::SurrogateOptimizerConfig optimizer;
  optimizer.iterations = 400;
  optimizer.learning_rate = 0.03;
  optimizer.minimum_learning_rate = 0.002;
  optimizer.seed = 8128;
  const auto result = project3::optimize_differentiable_surrogate(
      initial, paths, {}, optimizer);

  std::cout << std::fixed << std::setprecision(8)
            << "Differentiable market-making surrogate (not exchange fills)\n"
            << "  Seed:                    " << result.seed << '\n'
            << "  Initial expected reward: "
            << result.initial_reward_currency << '\n'
            << "  Final expected reward:   "
            << result.final_reward_currency << '\n'
            << "  Max gradient rel. error: "
            << gradient_check.maximum_relative_error << '\n'
            << "  Parameters:";
  for (const double value : result.parameters.values) {
    std::cout << ' ' << value;
  }
  std::cout << '\n';
}
