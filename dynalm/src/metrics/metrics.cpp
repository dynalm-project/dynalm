#include "metrics/metrics.h"

#include <algorithm>
#include <cstdio>

namespace engine::metrics {

Histogram::Histogram(std::vector<double> upper_bounds)
    : bounds_(std::move(upper_bounds)), buckets_(std::make_unique<std::atomic<uint64_t>[]>(bounds_.size() + 1)) {
  std::sort(bounds_.begin(), bounds_.end());
}

void Histogram::observe(double v) {
  const auto idx = static_cast<size_t>(std::lower_bound(bounds_.begin(), bounds_.end(), v) - bounds_.begin());
  buckets_[idx].fetch_add(1, std::memory_order_relaxed);
  count_.fetch_add(1, std::memory_order_relaxed);
  double cur = sum_.load(std::memory_order_relaxed);
  while (!sum_.compare_exchange_weak(cur, cur + v, std::memory_order_relaxed)) {
  }
}

double Histogram::quantile(double q) const {
  const uint64_t n = count();
  if (n == 0) return 0;
  const double target = q * static_cast<double>(n);
  uint64_t seen = 0;
  for (size_t i = 0; i <= bounds_.size(); ++i) {
    const uint64_t c = buckets_[i].load(std::memory_order_relaxed);
    if (static_cast<double>(seen + c) >= target && c > 0) {
      const double lo = i == 0 ? 0.0 : bounds_[i - 1];
      const double hi = i < bounds_.size() ? bounds_[i] : bounds_.back();
      return lo + (hi - lo) * (target - static_cast<double>(seen)) / static_cast<double>(c);
    }
    seen += c;
  }
  return bounds_.empty() ? 0 : bounds_.back();
}

void Histogram::render(std::string_view name, std::string_view help, std::string& out) const {
  char buf[256];
  std::snprintf(buf, sizeof(buf), "# HELP %.*s %.*s\n# TYPE %.*s histogram\n", static_cast<int>(name.size()),
                name.data(), static_cast<int>(help.size()), help.data(), static_cast<int>(name.size()), name.data());
  out += buf;
  uint64_t cumulative = 0;
  for (size_t i = 0; i <= bounds_.size(); ++i) {
    cumulative += buckets_[i].load(std::memory_order_relaxed);
    if (i < bounds_.size()) {
      std::snprintf(buf, sizeof(buf), "%.*s_bucket{le=\"%g\"} %llu\n", static_cast<int>(name.size()), name.data(),
                    bounds_[i], static_cast<unsigned long long>(cumulative));
    } else {
      std::snprintf(buf, sizeof(buf), "%.*s_bucket{le=\"+Inf\"} %llu\n", static_cast<int>(name.size()), name.data(),
                    static_cast<unsigned long long>(cumulative));
    }
    out += buf;
  }
  std::snprintf(buf, sizeof(buf), "%.*s_sum %g\n%.*s_count %llu\n", static_cast<int>(name.size()), name.data(), sum(),
                static_cast<int>(name.size()), name.data(), static_cast<unsigned long long>(count()));
  out += buf;
}

std::vector<double> latency_buckets_ms() {
  return {1, 2.5, 5, 10, 25, 50, 75, 100, 150, 250, 500, 750, 1000, 2500, 5000, 10000, 30000, 60000};
}

namespace {
void render_scalar(std::string_view type, std::string_view name, std::string_view help, double value,
                   std::string& out) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), "# HELP %.*s %.*s\n# TYPE %.*s %.*s\n%.*s %.17g\n", static_cast<int>(name.size()),
                name.data(), static_cast<int>(help.size()), help.data(), static_cast<int>(name.size()), name.data(),
                static_cast<int>(type.size()), type.data(), static_cast<int>(name.size()), name.data(), value);
  out += buf;
}
}  // namespace

void render_counter(std::string_view name, std::string_view help, double value, std::string& out) {
  render_scalar("counter", name, help, value, out);
}
void render_gauge(std::string_view name, std::string_view help, double value, std::string& out) {
  render_scalar("gauge", name, help, value, out);
}

}  // namespace engine::metrics
