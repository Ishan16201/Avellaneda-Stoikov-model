#include "project3/benchmark.hpp"
#include "project3/detail/parallel.hpp"
#include "project3/version.hpp"

#include "lob/stochastic/hawkes_parameters.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace project3 {
namespace {

struct NamedRun final {
  std::string scenario_label;
  MarketQualityEpisode episode;
};

struct ScenarioResults final {
  BenchmarkSplit split;
  BenchmarkScenario scenario;
  std::vector<MarketQualityScenarioSummary> policies;
};

const char* split_name(const BenchmarkSplit split) noexcept {
  switch (split) {
    case BenchmarkSplit::Training: return "training";
    case BenchmarkSplit::Validation: return "validation";
    case BenchmarkSplit::FinalTest: return "final_test";
  }
  return "unknown";
}

const char* ablation_name(const AblationKind kind) noexcept {
  switch (kind) {
    case AblationKind::WithoutHawkesInformation:
      return "without_hawkes_information";
    case AblationKind::WithoutQueueInformation:
      return "without_queue_information";
    case AblationKind::WithoutInventoryPenalties:
      return "without_inventory_penalties";
    case AblationKind::WithoutRegimeInformation:
      return "without_regime_information";
    case AblationKind::WithoutDynamicQuoteAdjustment:
      return "without_dynamic_quote_adjustment";
  }
  return "unknown";
}

double average(const std::vector<double>& values) {
  return values.empty()
             ? 0.0
             : std::accumulate(values.begin(), values.end(), 0.0) /
                   static_cast<double>(values.size());
}

double inverse_standard_normal(const double p) {
  constexpr std::array<double, 6> a{
      -3.969683028665376e+01, 2.209460984245205e+02,
      -2.759285104469687e+02, 1.383577518672690e+02,
      -3.066479806614716e+01, 2.506628277459239e+00};
  constexpr std::array<double, 5> b{
      -5.447609879822406e+01, 1.615858368580409e+02,
      -1.556989798598866e+02, 6.680131188771972e+01,
      -1.328068155288572e+01};
  constexpr std::array<double, 6> c{
      -7.784894002430293e-03, -3.223964580411365e-01,
      -2.400758277161838e+00, -2.549732539343734e+00,
      4.374664141464968e+00, 2.938163982698783e+00};
  constexpr std::array<double, 4> d{
      7.784695709041462e-03, 3.224671290700398e-01,
      2.445134137142996e+00, 3.754408661907416e+00};
  constexpr double low = 0.02425;
  constexpr double high = 1.0 - low;
  if (p < low) {
    const double q = std::sqrt(-2.0 * std::log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) *
                q +
            c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  }
  if (p > high) {
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) *
                  q +
              c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  }
  const double q = p - 0.5;
  const double r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) *
              r +
          a[5]) *
         q /
         (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) *
              r +
          1.0);
}

double beta_continued_fraction(const double a,
                                const double b,
                                const double x) {
  constexpr int max_iterations = 300;
  constexpr double epsilon = 3.0e-14;
  constexpr double fp_min = std::numeric_limits<double>::min() / epsilon;
  const double qab = a + b;
  const double qap = a + 1.0;
  const double qam = a - 1.0;
  double c = 1.0;
  double d = 1.0 - qab * x / qap;
  if (std::abs(d) < fp_min) d = fp_min;
  d = 1.0 / d;
  double h = d;
  for (int m = 1; m <= max_iterations; ++m) {
    const double m_double = static_cast<double>(m);
    const double m2 = 2.0 * m_double;
    double aa = m_double * (b - m_double) * x /
                 ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (std::abs(d) < fp_min) d = fp_min;
    c = 1.0 + aa / c;
    if (std::abs(c) < fp_min) c = fp_min;
    d = 1.0 / d;
    h *= d * c;
    aa = -(a + m_double) * (qab + m_double) * x /
          ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (std::abs(d) < fp_min) d = fp_min;
    c = 1.0 + aa / c;
    if (std::abs(c) < fp_min) c = fp_min;
    d = 1.0 / d;
    const double delta = d * c;
    h *= delta;
    if (std::abs(delta - 1.0) <= epsilon) return h;
  }
  throw std::runtime_error{
      "incomplete beta continued fraction did not converge"};
}

double regularized_incomplete_beta(const double x,
                                   const double a,
                                   const double b) {
  if (x <= 0.0) return 0.0;
  if (x >= 1.0) return 1.0;
  const double log_factor =
      std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
      a * std::log(x) + b * std::log1p(-x);
  const double factor = std::exp(log_factor);
  if (x < (a + 1.0) / (a + b + 2.0)) {
    return factor * beta_continued_fraction(a, b, x) / a;
  }
  return 1.0 - factor * beta_continued_fraction(b, a, 1.0 - x) / b;
}

double student_t_critical(const double confidence,
                           const std::size_t degrees_of_freedom) {
  const double df = static_cast<double>(degrees_of_freedom);
  const double target_tail = 1.0 - confidence;
  const auto beta_tail = [df](const double t) {
    return regularized_incomplete_beta(df / (df + t * t), 0.5 * df, 0.5);
  };
  double lower = 0.0;
  const double normal_probability =
      std::min(0.5 + confidence / 2.0,
               std::nextafter(1.0, 0.0));
  double upper =
      std::max(1.0, inverse_standard_normal(normal_probability));
  while (beta_tail(upper) > target_tail) upper *= 2.0;
  for (int iteration = 0; iteration < 100; ++iteration) {
    const double middle = lower + (upper - lower) / 2.0;
    if (beta_tail(middle) > target_tail) lower = middle;
    else upper = middle;
  }
  return lower + (upper - lower) / 2.0;
}

std::pair<double, double> mean_ci(const std::vector<double>& values,
                                  const double confidence) {
  if (values.empty()) return {0.0, 0.0};
  const double mean = average(values);
  if (values.size() < 2) return {mean, mean};
  double squared_error = 0.0;
  for (const double value : values) {
    const double residual = value - mean;
    squared_error += residual * residual;
  }
  const double standard_error =
      std::sqrt(squared_error /
                (static_cast<double>(values.size() - 1) *
                 static_cast<double>(values.size())));
  const double critical =
      student_t_critical(confidence, values.size() - 1);
  return {mean - critical * standard_error,
          mean + critical * standard_error};
}

double mean_adverse_selection(const MarketQualityEpisode& episode,
                              const double horizon_seconds) {
  const auto found = std::find_if(
      episode.markouts.begin(), episode.markouts.end(),
      [horizon_seconds](const HorizonMarkout& markout) {
        return std::abs(markout.horizon_seconds - horizon_seconds) <= 1.0e-9;
      });
  if (found == episode.markouts.end()) {
    throw std::logic_error{"benchmark primary markout horizon is unavailable"};
  }
  const auto& buy = found->market_maker_buys;
  const auto& sell = found->market_maker_sells;
  const double quantity =
      buy.executed_quantity_lots + sell.executed_quantity_lots;
  if (quantity == 0.0) return 0.0;
  return (buy.adverse_selection_ticks * buy.executed_quantity_lots +
          sell.adverse_selection_ticks * sell.executed_quantity_lots) /
         quantity;
}

std::string volatility_proxy_signature(
    const lob::stochastic::SyntheticMarketConfig& market) {
  const auto market_buy =
      lob::stochastic::event_type_index(lob::stochastic::EventType::MarketBuy);
  const auto market_sell =
      lob::stochastic::event_type_index(lob::stochastic::EventType::MarketSell);
  std::ostringstream signature;
  signature << std::setprecision(17)
            << market.hawkes.baseline_events_per_second[market_buy] << ','
            << market.hawkes.baseline_events_per_second[market_sell] << ';'
            << market.marks.market_size.min_lots << ','
            << market.marks.market_size.max_lots << ';';
  for (const auto event : {market_buy, market_sell}) {
    for (const auto value : market.hawkes.alpha_branching[event]) {
      signature << value << ',';
    }
    signature << ';';
    for (const auto value : market.hawkes.beta_per_second[event]) {
      signature << value << ',';
    }
    signature << ';';
  }
  return signature.str();
}

std::string liquidity_signature(
    const lob::stochastic::SyntheticMarketConfig& market) {
  std::ostringstream signature;
  signature << std::setprecision(17)
            << (market.initial_book.best_ask_ticks -
                market.initial_book.best_bid_ticks)
            << ','
            << market.initial_book.levels_per_side << ','
            << market.initial_book.orders_per_level << ','
            << market.initial_book.tick_spacing << ','
            << market.initial_book.order_size << ','
            << market.marks.limit_size.min_lots << ','
            << market.marks.limit_size.max_lots << ';';
  for (const auto& order : market.initial_book.explicit_orders) {
    signature << order.id << ',' << static_cast<int>(order.side) << ','
              << order.price_ticks << ',' << order.quantity << ';';
  }
  return signature.str();
}

BenchmarkMetricSummary summarize(
    const BenchmarkSplit split,
    const std::string& scenario_label,
    const bool perturbation,
    const bool unseen,
    const std::string& policy_name,
    const std::string& model_version,
    const std::vector<MarketQualityEpisode>& episodes,
    const double confidence,
    const double markout_horizon) {
  BenchmarkMetricSummary result;
  result.split = split;
  result.scenario_label = scenario_label;
  result.parameter_perturbation = perturbation;
  result.unseen_configuration = unseen;
  result.policy_name = policy_name;
  result.model_version = model_version;
  result.sample_count = episodes.size();
  std::vector<double> pnl;
  std::vector<double> losses;
  pnl.reserve(episodes.size());
  losses.reserve(episodes.size());
  for (const auto& episode : episodes) {
    pnl.push_back(episode.net_pnl_currency);
    losses.push_back(-episode.net_pnl_currency);
    result.mean_sharpe_ratio += episode.sharpe_ratio;
    result.mean_maximum_drawdown_currency +=
        episode.maximum_drawdown_currency;
    result.worst_maximum_drawdown_currency =
        std::max(result.worst_maximum_drawdown_currency,
                 episode.maximum_drawdown_currency);
    result.fill_rate += episode.fill_probability;
    result.spread_capture_currency += episode.spread_capture_currency;
    result.inventory_risk_lot_squared_seconds +=
        episode.inventory_risk_lot_squared_seconds;
    result.inventory_concentration_ratio +=
        episode.inventory_concentration_ratio;
    result.turnover_currency += episode.gross_turnover_currency;
    result.transaction_costs_currency += episode.fees_currency;
    result.adverse_selection_ticks +=
        mean_adverse_selection(episode, markout_horizon);
  }
  result.mean_pnl_currency = average(pnl);
  const auto [low, high] = mean_ci(pnl, confidence);
  result.pnl_confidence_interval_low_currency = low;
  result.pnl_confidence_interval_high_currency = high;
  if (pnl.size() > 1) {
    double sum_squares = 0.0;
    for (const double value : pnl) {
      const double residual = value - result.mean_pnl_currency;
      sum_squares += residual * residual;
    }
    result.pnl_standard_deviation_currency =
        std::sqrt(sum_squares / static_cast<double>(pnl.size() - 1));
  }
  if (!losses.empty()) {
    std::sort(losses.begin(), losses.end());
    const auto index = static_cast<std::size_t>(
                           std::ceil(confidence *
                                     static_cast<double>(losses.size()))) -
                       1;
    result.value_at_risk_loss_currency = losses[index];
    std::vector<double> tail;
    for (const double loss : losses) {
      if (loss >= result.value_at_risk_loss_currency) tail.push_back(loss);
    }
    result.expected_shortfall_loss_currency = average(tail);
  }
  if (!episodes.empty()) {
    const double count = static_cast<double>(episodes.size());
    result.mean_sharpe_ratio /= count;
    result.mean_maximum_drawdown_currency /= count;
    result.fill_rate /= count;
    result.spread_capture_currency /= count;
    result.adverse_selection_ticks /= count;
    result.inventory_risk_lot_squared_seconds /= count;
    result.inventory_concentration_ratio /= count;
    result.turnover_currency /= count;
    result.transaction_costs_currency /= count;
  }
  return result;
}

void validate_scenarios(const std::vector<BenchmarkScenario>& scenarios,
                        const char* split,
                        const bool must_be_unseen) {
  if (scenarios.size() < 2) {
    throw std::invalid_argument{
        std::string{split} +
        " requires multiple scenarios to test parameter/regime variation"};
  }
  std::set<std::string> labels;
  std::set<std::string> volatility_regimes;
  std::set<std::string> liquidity_regimes;
  std::set<std::string> volatility_proxies;
  std::set<std::string> liquidity_conditions;
  std::set<std::string> hawkes_configurations;
  bool has_parameter_perturbation = false;
  for (const auto& scenario : scenarios) {
    if (scenario.label.empty() || scenario.volatility_regime.empty() ||
        scenario.liquidity_regime.empty() ||
        !labels.insert(scenario.label).second || scenario.seeds.size() < 2) {
      throw std::invalid_argument{
          std::string{split} +
          " scenarios need unique labels, volatility/liquidity labels, and "
          "at least two seeds"};
    }
    if (must_be_unseen && !scenario.unseen_configuration) {
      throw std::invalid_argument{
          "every final-test scenario must be explicitly marked unseen"};
    }
    if (!must_be_unseen && scenario.unseen_configuration) {
      throw std::invalid_argument{
          "training and validation scenarios cannot be marked unseen"};
    }
    std::set<std::uint64_t> seeds;
    for (const auto seed : scenario.seeds) {
      if (!seeds.insert(seed).second) {
        throw std::invalid_argument{
            std::string{split} + " scenario repeats a random seed"};
      }
    }
    lob::stochastic::validate_stationary(scenario.market.hawkes);
    std::ostringstream hawkes_signature;
    hawkes_signature << std::setprecision(17);
    for (const auto value :
         scenario.market.hawkes.baseline_events_per_second) {
      hawkes_signature << value << ',';
    }
    hawkes_signature << ';';
    for (const auto& row : scenario.market.hawkes.alpha_branching) {
      for (const auto value : row) hawkes_signature << value << ',';
      hawkes_signature << ';';
    }
    for (const auto& row : scenario.market.hawkes.beta_per_second) {
      for (const auto value : row) hawkes_signature << value << ',';
      hawkes_signature << ';';
    }
    hawkes_configurations.insert(hawkes_signature.str());
    volatility_regimes.insert(scenario.volatility_regime);
    liquidity_regimes.insert(scenario.liquidity_regime);
    volatility_proxies.insert(
        volatility_proxy_signature(scenario.market));
    liquidity_conditions.insert(liquidity_signature(scenario.market));
    has_parameter_perturbation =
        has_parameter_perturbation || scenario.parameter_perturbation;
  }
  if (hawkes_configurations.size() < 2 ||
      volatility_regimes.size() < 2 || liquidity_regimes.size() < 2 ||
      volatility_proxies.size() < 2 || liquidity_conditions.size() < 2) {
    throw std::invalid_argument{
        std::string{split} +
        " requires multiple Hawkes configurations, labeled volatility "
        "proxies, and actual market-flow/liquidity perturbations"};
  }
  if (must_be_unseen && !has_parameter_perturbation) {
    throw std::invalid_argument{
        "final-test scenarios must include a marked parameter perturbation"};
  }
}

std::string market_signature(
    const lob::stochastic::SyntheticMarketConfig& market) {
  std::ostringstream stream;
  stream << std::setprecision(17);
  const auto append_vector = [&stream](const std::vector<double>& values) {
    for (const auto value : values) stream << value << ',';
  };
  append_vector(market.hawkes.baseline_events_per_second);
  stream << ';';
  for (const auto& row : market.hawkes.alpha_branching) {
    append_vector(row);
    stream << ';';
  }
  for (const auto& row : market.hawkes.beta_per_second) {
    append_vector(row);
    stream << ';';
  }
  stream << market.horizon_seconds << ';'
         << market.initial_book.best_bid_ticks << ';'
         << market.initial_book.best_ask_ticks << ';'
         << market.initial_book.levels_per_side << ';'
         << market.initial_book.orders_per_level << ';'
         << market.initial_book.tick_spacing << ';'
         << market.initial_book.order_size << ';'
         << market.marks.price_placement.reference_price_ticks << ';'
         << market.marks.price_placement.min_offset_ticks << ';'
         << market.marks.price_placement.max_offset_ticks << ';'
         << market.marks.limit_size.min_lots << ';'
         << market.marks.limit_size.max_lots << ';'
         << market.marks.market_size.min_lots << ';'
         << market.marks.market_size.max_lots << ';'
         << static_cast<int>(market.marks.cancellation.policy);
  for (const auto& order : market.initial_book.explicit_orders) {
    stream << ';' << order.id << ',' << static_cast<int>(order.side) << ','
           << order.price_ticks << ',' << order.quantity;
  }
  return stream.str();
}

void ensure_disjoint_splits(const BenchmarkExperimentConfig& config) {
  const std::array<const std::vector<BenchmarkScenario>*, 3> groups{
      &config.training_scenarios, &config.validation_scenarios,
      &config.final_test_scenarios};
  std::set<std::string> development_configurations;
  std::array<std::set<std::uint64_t>, 3> split_seeds;
  for (std::size_t g = 0; g < groups.size(); ++g) {
    for (const auto& scenario : *groups[g]) {
      const auto signature = market_signature(scenario.market);
      if (g < 2) {
        development_configurations.insert(signature);
      } else if (development_configurations.contains(signature)) {
        throw std::invalid_argument{
            "final-test configuration duplicates a training/validation "
            "configuration"};
      }
      for (const auto seed : scenario.seeds) {
        split_seeds[g].insert(seed);
      }
    }
  }
  for (std::size_t left = 0; left < split_seeds.size(); ++left) {
    for (std::size_t right = left + 1; right < split_seeds.size(); ++right) {
      for (const auto seed : split_seeds[left]) {
        if (split_seeds[right].contains(seed)) {
          throw std::invalid_argument{
              "training, validation, and final-test seeds must be disjoint"};
        }
      }
    }
  }
}

std::string json_escape(const std::string& value) {
  std::string result;
  result.reserve(value.size() + 2);
  for (const char character : value) {
    switch (character) {
      case '"': result += "\\\""; break;
      case '\\': result += "\\\\"; break;
      case '\n': result += "\\n"; break;
      case '\r': result += "\\r"; break;
      case '\t': result += "\\t"; break;
      default: result += character; break;
    }
  }
  return result;
}

std::string csv_escape(const std::string& value) {
  std::string result{"\""};
  for (const char character : value) {
    if (character == '"') result += "\"\"";
    else result += character;
  }
  result += '"';
  return result;
}

void write_manifest(const std::string& path,
                    const BenchmarkExperimentConfig& config,
                    const std::vector<BenchmarkPolicy>& policies) {
  std::ofstream output{path};
  if (!output) throw std::runtime_error{"cannot open benchmark manifest"};
  output << std::setprecision(17)
         << "{\n  \"experiment_id\": \"" << json_escape(config.experiment_id)
         << "\",\n  \"runner_version\": \"project3-benchmark-v2\""
         << ",\n  \"library_version\": \"" << json_escape(library_version())
         << "\",\n  \"git_revision\": \""
         << json_escape(config.git_revision.empty() ? git_revision()
                                                    : config.git_revision)
         << "\",\n  \"worker_count\": "
         << detail::resolve_worker_count(config.worker_count)
         << ",\n  \"confidence_level\": " << config.confidence_level
         << ",\n  \"primary_adverse_selection_horizon_seconds\": "
         << config.primary_adverse_selection_horizon_seconds
         << ",\n  \"markout_horizons_seconds\": [";
  for (std::size_t i = 0;
       i < config.market_quality.markout_horizons_seconds.size(); ++i) {
    if (i > 0) output << ',';
    output << config.market_quality.markout_horizons_seconds[i];
  }
  output << "],\n  \"inventory_limit_lots\": "
         << config.market_quality.inventory_limit_lots
         << ",\n  \"currency_per_tick\": "
         << config.market_quality.currency_per_tick
         << ",\n  \"sharpe_capital_base_currency\": "
         << config.market_quality.sharpe_capital_base_currency
         << ",\n  \"annualization_seconds\": "
         << config.market_quality.annualization_seconds
         << ",\n  \"expected_shortfall_confidence\": "
         << config.market_quality.expected_shortfall_confidence
         << ",\n  \"evaluation\": {\"timer_interval_ns\": "
         << config.market_quality.evaluation.timer_interval_ns
         << ", \"evaluate_parent_order\": "
         << (config.market_quality.evaluation.evaluate_parent_order ? "true"
                                                                     : "false")
         << ", \"parent_order_target\": "
         << config.market_quality.evaluation.parent_order_target
         << ", \"parent_order_side\": "
         << static_cast<int>(config.market_quality.evaluation.parent_order_side)
         << ", \"broker\": {\"currency_per_tick\": "
         << config.market_quality.evaluation.broker_config.currency_per_tick
         << ", \"maker_fee_per_lot\": "
         << config.market_quality.evaluation.broker_config.fees
                .maker_fee_per_lot
         << ", \"taker_fee_per_lot\": "
         << config.market_quality.evaluation.broker_config.fees
                .taker_fee_per_lot
         << ", \"commission_per_order\": "
         << config.market_quality.evaluation.broker_config.fees
                .commission_per_order
         << ", \"max_position_lots\": "
         << config.market_quality.evaluation.broker_config.risk
                .max_position_lots
         << ", \"max_order_lots\": "
         << config.market_quality.evaluation.broker_config.risk.max_order_lots
         << ", \"allow_short_selling\": "
         << (config.market_quality.evaluation.broker_config.risk
                     .allow_short_selling
                 ? "true"
                 : "false")
         << ", \"max_drawdown_currency\": "
         << config.market_quality.evaluation.broker_config.risk
                .max_drawdown_currency
         << ", \"outbound_latency_ns\": "
         << config.market_quality.evaluation.broker_config.latency
                .outbound_delay_ns
         << ", \"inbound_latency_ns\": "
         << config.market_quality.evaluation.broker_config.latency
                .inbound_delay_ns
         << "}},\n  \"policies\": [";
  for (std::size_t i = 0; i < policies.size(); ++i) {
    if (i > 0) output << ',';
    output << "\n    {\"name\": \"" << json_escape(policies[i].name)
           << "\", \"model_version\": \""
           << json_escape(policies[i].model_version)
           << "\", \"configuration\": \""
           << json_escape(policies[i].configuration) << "\"}";
  }
  output << "\n  ],\n  \"splits\": {";
  const std::array<std::pair<const char*, const std::vector<BenchmarkScenario>*>,
                   3>
      split_scenarios{{{"training", &config.training_scenarios},
                       {"validation", &config.validation_scenarios},
                       {"final_test", &config.final_test_scenarios}}};
  for (std::size_t i = 0; i < split_scenarios.size(); ++i) {
    const auto& [label, scenarios] = split_scenarios[i];
    if (i > 0) output << ',';
    output << "\n    \"" << label << "\": [";
    for (std::size_t j = 0; j < scenarios->size(); ++j) {
      const auto& scenario = (*scenarios)[j];
      if (j > 0) output << ',';
      output << "\n      {\"label\": \"" << json_escape(scenario.label)
             << "\", \"volatility_regime\": \""
             << json_escape(scenario.volatility_regime)
             << "\", \"liquidity_regime\": \""
             << json_escape(scenario.liquidity_regime)
             << "\", \"unseen\": "
             << (scenario.unseen_configuration ? "true" : "false")
             << ", \"parameter_perturbation\": "
             << (scenario.parameter_perturbation ? "true" : "false")
             << ", \"horizon_seconds\": " << scenario.market.horizon_seconds
             << ", \"hawkes_baseline\": [";
      for (std::size_t k = 0;
           k < scenario.market.hawkes.baseline_events_per_second.size(); ++k) {
        if (k > 0) output << ',';
        output << scenario.market.hawkes.baseline_events_per_second[k];
      }
      output << "], \"hawkes_alpha\": [";
      for (std::size_t row = 0;
           row < scenario.market.hawkes.alpha_branching.size(); ++row) {
        if (row > 0) output << ',';
        output << '[';
        for (std::size_t column = 0;
             column < scenario.market.hawkes.alpha_branching[row].size();
             ++column) {
          if (column > 0) output << ',';
          output << scenario.market.hawkes.alpha_branching[row][column];
        }
        output << ']';
      }
      output << "], \"hawkes_beta\": [";
      for (std::size_t row = 0;
           row < scenario.market.hawkes.beta_per_second.size(); ++row) {
        if (row > 0) output << ',';
        output << '[';
        for (std::size_t column = 0;
             column < scenario.market.hawkes.beta_per_second[row].size();
             ++column) {
          if (column > 0) output << ',';
          output << scenario.market.hawkes.beta_per_second[row][column];
        }
        output << ']';
      }
      output << "], \"book\": {\"best_bid\": "
             << scenario.market.initial_book.best_bid_ticks
             << ", \"best_ask\": "
             << scenario.market.initial_book.best_ask_ticks
             << ", \"levels\": "
             << scenario.market.initial_book.levels_per_side
             << ", \"orders_per_level\": "
             << scenario.market.initial_book.orders_per_level
             << ", \"order_size\": "
             << scenario.market.initial_book.order_size
             << ", \"tick_spacing\": "
             << scenario.market.initial_book.tick_spacing
             << ", \"explicit_orders\": [";
      for (std::size_t k = 0;
           k < scenario.market.initial_book.explicit_orders.size(); ++k) {
        const auto& order = scenario.market.initial_book.explicit_orders[k];
        if (k > 0) output << ',';
        output << "{\"id\": " << order.id << ", \"side\": "
               << static_cast<int>(order.side) << ", \"price_ticks\": "
               << order.price_ticks << ", \"quantity\": " << order.quantity
               << '}';
      }
      output << "]}, \"marks\": {\"price_reference\": "
             << scenario.market.marks.price_placement.reference_price_ticks
             << ", \"min_price_offset\": "
             << scenario.market.marks.price_placement.min_offset_ticks
             << ", \"max_price_offset\": "
             << scenario.market.marks.price_placement.max_offset_ticks
             << ", \"limit_min_lots\": "
             << scenario.market.marks.limit_size.min_lots
             << ", \"limit_max_lots\": "
             << scenario.market.marks.limit_size.max_lots
             << ", \"market_min_lots\": "
             << scenario.market.marks.market_size.min_lots
             << ", \"market_max_lots\": "
             << scenario.market.marks.market_size.max_lots
             << ", \"cancellation_policy\": "
             << static_cast<int>(scenario.market.marks.cancellation.policy)
             << "}, \"seeds\": [";
      for (std::size_t k = 0; k < scenario.seeds.size(); ++k) {
        if (k > 0) output << ',';
        output << scenario.seeds[k];
      }
      output << "]}";
    }
    output << "\n    ]";
  }
  output << "\n  },\n  \"ablations\": [";
  for (std::size_t i = 0; i < config.ablations.size(); ++i) {
    if (i > 0) output << ',';
    output << "\n    {\"kind\": \""
           << ablation_name(config.ablations[i].kind)
           << "\", \"baseline_policy\": \""
           << json_escape(config.ablations[i].baseline_policy_name)
           << "\", \"ablated_policy\": \""
           << json_escape(config.ablations[i].ablated_policy.name)
           << "\", \"configuration\": \""
           << json_escape(config.ablations[i].ablated_policy.configuration)
           << "\"}";
  }
  output << "\n  ]\n}\n";
  if (!output) throw std::runtime_error{"failed writing benchmark manifest"};
}

void write_runs_csv(const std::string& path,
                    const std::vector<ScenarioResults>& results,
                    const std::map<std::string, std::string>& versions,
                    const double horizon) {
  std::ofstream output{path};
  if (!output) throw std::runtime_error{"cannot open benchmark runs CSV"};
  output << "split,scenario,volatility_regime,liquidity_regime,"
            "parameter_perturbation,unseen,policy,model_version,"
            "seed,pnl_currency,sharpe,max_drawdown_currency,fill_rate,"
            "spread_capture_currency,adverse_selection_ticks,inventory_duration_"
            "lot_seconds,inventory_risk_lot_squared_seconds,"
            "inventory_concentration,turnover_currency,"
            "net_transaction_costs_currency,charged_fees_currency,rebates_currency,"
            "inventory_limit_breach_duration_seconds,inventory_limit_breach_"
            "intervals\n"
         << std::setprecision(17);
  for (const auto& group : results) {
    for (const auto& policy : group.policies) {
      const auto version = versions.at(policy.policy_name);
      for (const auto& run : policy.runs) {
        output << split_name(group.split) << ','
               << csv_escape(group.scenario.label) << ','
               << csv_escape(group.scenario.volatility_regime) << ','
               << csv_escape(group.scenario.liquidity_regime) << ','
               << (group.scenario.parameter_perturbation ? "true" : "false")
               << ',' << (group.scenario.unseen_configuration ? "true" : "false")
               << ',' << csv_escape(policy.policy_name) << ','
               << csv_escape(version) << ',' << run.seed << ','
               << run.net_pnl_currency << ',' << run.sharpe_ratio << ','
               << run.maximum_drawdown_currency << ','
               << run.fill_probability << ',' << run.spread_capture_currency
               << ',' << mean_adverse_selection(run, horizon) << ','
               << run.inventory_duration_lot_seconds << ','
               << run.inventory_risk_lot_squared_seconds << ','
               << run.inventory_concentration_ratio << ','
               << run.gross_turnover_currency << ',' << run.fees_currency
               << ','
               << run.charged_fees_currency << ',' << run.rebates_currency
               << ',' << run.inventory_limit_breach_duration_seconds << ','
               << run.inventory_limit_breach_intervals << '\n';
      }
    }
  }
  if (!output) throw std::runtime_error{"failed writing benchmark runs CSV"};
}

void write_summaries_csv(
    const std::string& path,
    const std::vector<BenchmarkMetricSummary>& summaries) {
  std::ofstream output{path};
  if (!output) throw std::runtime_error{"cannot open benchmark summaries CSV"};
  output << "split,scenario,parameter_perturbation,unseen,policy,model_version,"
            "sample_count,mean_pnl_currency,pnl_std_currency,ci_low_currency,"
            "ci_high_currency,mean_sharpe,mean_max_drawdown_currency,"
            "worst_max_drawdown_currency,var_loss_currency,"
            "expected_shortfall_loss_currency,fill_rate,"
            "spread_capture_currency,adverse_selection_ticks,"
            "inventory_risk_lot_squared_seconds,inventory_concentration,"
            "turnover_currency,transaction_costs_currency\n"
         << std::setprecision(17);
  for (const auto& summary : summaries) {
    output << split_name(summary.split) << ','
           << csv_escape(summary.scenario_label) << ','
           << (summary.parameter_perturbation ? "true" : "false") << ','
           << (summary.unseen_configuration ? "true" : "false") << ','
           << csv_escape(summary.policy_name) << ','
           << csv_escape(summary.model_version) << ',' << summary.sample_count
           << ',' << summary.mean_pnl_currency << ','
           << summary.pnl_standard_deviation_currency << ','
           << summary.pnl_confidence_interval_low_currency << ','
           << summary.pnl_confidence_interval_high_currency << ','
           << summary.mean_sharpe_ratio << ','
           << summary.mean_maximum_drawdown_currency << ','
           << summary.worst_maximum_drawdown_currency << ','
           << summary.value_at_risk_loss_currency << ','
           << summary.expected_shortfall_loss_currency << ','
           << summary.fill_rate << ',' << summary.spread_capture_currency
           << ',' << summary.adverse_selection_ticks << ','
           << summary.inventory_risk_lot_squared_seconds << ','
           << summary.inventory_concentration_ratio << ','
           << summary.turnover_currency << ','
           << summary.transaction_costs_currency << '\n';
  }
  if (!output) throw std::runtime_error{"failed writing benchmark summaries CSV"};
}

void write_paired_csv(const std::string& path,
                      const std::vector<PairedBenchmarkComparison>& results) {
  std::ofstream output{path};
  if (!output) throw std::runtime_error{"cannot open paired benchmark CSV"};
  output << "split,scenario,parameter_perturbation,unseen,reference_policy,"
            "comparison_policy,paired_samples,mean_pnl_difference_currency,"
            "difference_std_currency,ci_low_currency,ci_high_currency\n"
         << std::setprecision(17);
  for (const auto& row : results) {
    output << split_name(row.split) << ',' << csv_escape(row.scenario_label)
           << ',' << (row.parameter_perturbation ? "true" : "false") << ','
           << (row.unseen_configuration ? "true" : "false") << ','
           << csv_escape(row.reference_policy) << ','
           << csv_escape(row.comparison_policy) << ',' << row.paired_samples
           << ',' << row.mean_pnl_difference_currency << ','
           << row.pnl_difference_standard_deviation_currency << ','
           << row.difference_confidence_interval_low_currency << ','
           << row.difference_confidence_interval_high_currency << '\n';
  }
  if (!output) throw std::runtime_error{"failed writing paired benchmark CSV"};
}

PairedBenchmarkComparison paired_comparison(
    const BenchmarkSplit split,
    const BenchmarkScenario& scenario,
    const MarketQualityScenarioSummary& reference,
    const MarketQualityScenarioSummary& comparison,
    const double confidence) {
  std::map<std::uint64_t, double> reference_pnl;
  for (const auto& run : reference.runs) {
    reference_pnl.emplace(run.seed, run.net_pnl_currency);
  }
  std::vector<double> differences;
  for (const auto& run : comparison.runs) {
    const auto found = reference_pnl.find(run.seed);
    if (found == reference_pnl.end()) {
      throw std::logic_error{"paired policy runs do not share identical seeds"};
    }
    differences.push_back(run.net_pnl_currency - found->second);
  }
  if (differences.size() != reference.runs.size()) {
    throw std::logic_error{"paired policy seed sets have different sizes"};
  }
  PairedBenchmarkComparison result;
  result.split = split;
  result.scenario_label = scenario.label;
  result.parameter_perturbation = scenario.parameter_perturbation;
  result.unseen_configuration = scenario.unseen_configuration;
  result.reference_policy = reference.policy_name;
  result.comparison_policy = comparison.policy_name;
  result.paired_samples = differences.size();
  result.mean_pnl_difference_currency = average(differences);
  const auto [low, high] = mean_ci(differences, confidence);
  result.difference_confidence_interval_low_currency = low;
  result.difference_confidence_interval_high_currency = high;
  if (differences.size() > 1) {
    double squared_error = 0.0;
    for (const double value : differences) {
      const double residual = value - result.mean_pnl_difference_currency;
      squared_error += residual * residual;
    }
    result.pnl_difference_standard_deviation_currency =
        std::sqrt(squared_error /
                  static_cast<double>(differences.size() - 1));
  }
  return result;
}

void add_split_results(BenchmarkExperimentResult& output,
                       std::vector<ScenarioResults>& evaluated,
                       const BenchmarkSplit split,
                       const std::vector<BenchmarkScenario>& scenarios,
                       const std::vector<BenchmarkPolicy>& policies,
                       const MarketQualityConfig& market_quality,
                       const double confidence,
                       const double markout_horizon) {
  const bool aggregate_perturbation = std::any_of(
      scenarios.begin(), scenarios.end(),
      [](const BenchmarkScenario& scenario) {
        return scenario.parameter_perturbation;
      });
  const bool aggregate_unseen =
      split == BenchmarkSplit::FinalTest &&
      std::all_of(scenarios.begin(), scenarios.end(),
                  [](const BenchmarkScenario& scenario) {
                    return scenario.unseen_configuration;
                  });
  for (const auto& scenario : scenarios) {
    MarketStressScenario quality_scenario{
        .label = scenario.label,
        .kind = MarketStressKind::VolatilityIncrease,
        .market = scenario.market,
        .seeds = scenario.seeds};
    std::vector<MarketQualityPolicy> quality_policies;
    quality_policies.reserve(policies.size());
    for (const auto& policy : policies) {
      quality_policies.push_back(
          {.name = policy.name, .make_agent = policy.make_agent});
    }
    auto summaries =
        run_market_quality_study(quality_policies, {quality_scenario},
                                 market_quality);
    for (const auto& summary : summaries) {
      output.summaries.push_back(summarize(
          split, scenario.label, scenario.parameter_perturbation,
          scenario.unseen_configuration, summary.policy_name,
          std::find_if(policies.begin(), policies.end(),
                       [&summary](const BenchmarkPolicy& policy) {
                         return policy.name == summary.policy_name;
                       })->model_version,
          summary.runs, confidence, markout_horizon));
    }
    for (std::size_t i = 0; i < summaries.size(); ++i) {
      for (std::size_t j = i + 1; j < summaries.size(); ++j) {
        output.paired_comparisons.push_back(paired_comparison(
            split, scenario, summaries[i], summaries[j], confidence));
      }
    }
    evaluated.push_back({.split = split,
                         .scenario = scenario,
                         .policies = std::move(summaries)});
  }

  for (const auto& policy : policies) {
    std::vector<MarketQualityEpisode> pooled;
    for (const auto& scenario_result : evaluated) {
      if (scenario_result.split != split) continue;
      const auto found = std::find_if(
          scenario_result.policies.begin(), scenario_result.policies.end(),
          [&policy](const MarketQualityScenarioSummary& summary) {
            return summary.policy_name == policy.name;
          });
      if (found != scenario_result.policies.end()) {
        pooled.insert(pooled.end(), found->runs.begin(), found->runs.end());
      }
    }
    output.summaries.push_back(summarize(
        split, "ALL", aggregate_perturbation, aggregate_unseen, policy.name,
        policy.model_version, pooled, confidence, markout_horizon));
  }
}

}  // namespace

const char* to_string(const BenchmarkSplit split) noexcept {
  return split_name(split);
}

const char* to_string(const AblationKind kind) noexcept {
  return ablation_name(kind);
}

BenchmarkExperimentResult run_market_making_benchmark(
    const BenchmarkExperimentConfig& config) {
  if (config.experiment_id.empty() ||
      !std::isfinite(config.confidence_level) ||
      config.confidence_level <= 0.0 || config.confidence_level >= 1.0 ||
      !std::isfinite(config.primary_adverse_selection_horizon_seconds) ||
      config.primary_adverse_selection_horizon_seconds <= 0.0) {
    throw std::invalid_argument{"invalid market benchmark configuration"};
  }
  validate_scenarios(config.training_scenarios, "training", false);
  validate_scenarios(config.validation_scenarios, "validation", false);
  validate_scenarios(config.final_test_scenarios, "final-test", true);
  ensure_disjoint_splits(config);
  auto market_quality = config.market_quality;
  market_quality.worker_count = config.worker_count;
  std::set<std::string> names;
  std::vector<BenchmarkPolicy> policies = config.policies;
  if (policies.empty()) {
    throw std::invalid_argument{"benchmark requires at least one frozen policy"};
  }
  for (const auto& policy : policies) {
    if (policy.name.empty() || policy.model_version.empty() ||
        policy.configuration.empty() ||
        !policy.make_agent || !names.insert(policy.name).second) {
      throw std::invalid_argument{
          "benchmark policies need unique names, versions, and factories"};
    }
  }
  std::set<AblationKind> ablation_kinds;
  for (const auto& ablation : config.ablations) {
    if (ablation.baseline_policy_name.empty() ||
        !names.contains(ablation.baseline_policy_name) ||
        ablation.ablated_policy.name.empty() ||
        ablation.ablated_policy.model_version.empty() ||
        ablation.ablated_policy.configuration.empty() ||
        !ablation.ablated_policy.make_agent ||
        !ablation_kinds.insert(ablation.kind).second ||
        !names.insert(ablation.ablated_policy.name).second) {
      throw std::invalid_argument{
          "benchmark ablations need a baseline and unique variant per ablation"};
    }
    policies.push_back(ablation.ablated_policy);
  }
  constexpr std::array<AblationKind, 5> required_ablations{
      AblationKind::WithoutHawkesInformation,
      AblationKind::WithoutQueueInformation,
      AblationKind::WithoutInventoryPenalties,
      AblationKind::WithoutRegimeInformation,
      AblationKind::WithoutDynamicQuoteAdjustment};
  for (const auto kind : required_ablations) {
    if (!ablation_kinds.contains(kind)) {
      throw std::invalid_argument{
          std::string{"benchmark is missing required ablation: "} +
          ablation_name(kind)};
    }
  }
  const auto has_horizon = std::any_of(
      config.market_quality.markout_horizons_seconds.begin(),
      config.market_quality.markout_horizons_seconds.end(),
      [&config](const double horizon) {
        return std::abs(
                   horizon -
                   config.primary_adverse_selection_horizon_seconds) <=
               1.0e-9;
      });
  if (!has_horizon) {
    throw std::invalid_argument{
        "primary markout horizon must be included in market-quality horizons"};
  }

  BenchmarkExperimentResult result;
  result.experiment_id = config.experiment_id;
  std::vector<ScenarioResults> evaluated;
  add_split_results(result, evaluated, BenchmarkSplit::Training,
                    config.training_scenarios, policies,
                    market_quality, config.confidence_level,
                    config.primary_adverse_selection_horizon_seconds);
  add_split_results(result, evaluated, BenchmarkSplit::Validation,
                    config.validation_scenarios, policies,
                    market_quality, config.confidence_level,
                    config.primary_adverse_selection_horizon_seconds);
  add_split_results(result, evaluated, BenchmarkSplit::FinalTest,
                    config.final_test_scenarios, policies,
                    market_quality, config.confidence_level,
                    config.primary_adverse_selection_horizon_seconds);

  if (!config.output_prefix.empty()) {
    const std::filesystem::path prefix{config.output_prefix};
    const auto parent = prefix.parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    result.manifest_path = config.output_prefix + ".manifest.json";
    result.runs_csv_path = config.output_prefix + ".runs.csv";
    result.paired_csv_path = config.output_prefix + ".paired.csv";
    result.summaries_csv_path = config.output_prefix + ".summaries.csv";
    std::map<std::string, std::string> versions;
    for (const auto& policy : policies) {
      versions.emplace(policy.name, policy.model_version);
    }
    write_manifest(result.manifest_path, config, policies);
    write_runs_csv(result.runs_csv_path, evaluated, versions,
                   config.primary_adverse_selection_horizon_seconds);
    write_summaries_csv(result.summaries_csv_path, result.summaries);
    write_paired_csv(result.paired_csv_path, result.paired_comparisons);
  }
  return result;
}

}  // namespace project3
