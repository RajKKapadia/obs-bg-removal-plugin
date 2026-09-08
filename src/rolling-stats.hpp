#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <vector>

namespace rmbg {
struct MetricSummary {
    size_t count = 0;
    double mean = 0, p95 = 0, rate = 0;
};
// Caller provides synchronization and a monotonic clock. Bounded even when
// invoked by an unusually high-rate OBS output. Values are milliseconds.
class RollingMetric {
public:
    static constexpr uint64_t window_ns = 5000000000ULL;
    static constexpr size_t capacity = 4096;
    void reset(uint64_t now) { samples_.clear(); start_ = now; }
    void add(uint64_t now, double value = 0)
    {
        if (!start_ || now < start_ || (!samples_.empty() && now < samples_.back().time)) reset(now);
        prune(now);
        if (samples_.size() == capacity) samples_.pop_front();
        samples_.push_back({now, value});
    }
    MetricSummary summary(uint64_t now) const
    {
        MetricSummary s;
        if (!start_ || now < start_) return s;
        std::vector<double> values;
        for (const auto &sample : samples_)
            if (sample.time <= now && now - sample.time < window_ns) {
                values.push_back(sample.value); s.mean += sample.value;
            }
        s.count = values.size();
        if (!s.count) return s;
        s.mean /= s.count;
        std::sort(values.begin(), values.end());
        s.p95 = values[size_t(std::ceil(0.95 * values.size())) - 1];
        const double seconds = double(std::min(window_ns, now - start_)) / 1e9;
        s.rate = seconds > 0 ? s.count / seconds : 0;
        return s;
    }
    size_t stored() const { return samples_.size(); }
private:
    struct Sample { uint64_t time; double value; };
    void prune(uint64_t now)
    {
        while (!samples_.empty() && now - samples_.front().time >= window_ns) samples_.pop_front();
    }
    std::deque<Sample> samples_;
    uint64_t start_ = 0;
};
}
