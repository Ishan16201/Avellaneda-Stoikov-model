#pragma once

#include "project3/market_agent.hpp"
#include "lob/strategy/strategy_evaluator.hpp"

#include <cstdint>

namespace project3 {

/// Event-driven facade over the completed Hawkes LOB strategy evaluator.
///
/// The underlying evaluator advances the exchange event by event and invokes
/// the agent only with the current causal view. It owns the event tape, RNG,
/// matching engine, and broker.
class ExchangeAdapter final {
 public:
  [[nodiscard]] lob::strategy::StrategyEvaluationReport run(
      TradingAgent& agent,
      const lob::stochastic::SyntheticMarketConfig& market,
      const lob::strategy::StrategyEvaluatorConfig& evaluation = {}) const;
};

}  // namespace project3
