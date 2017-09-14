#pragma once

#include "timer.h"

class bandwidth_counter : public timer {
  public:
    bandwidth_counter(long long bytes) : m_bytes(bytes) {}

    virtual result_array total() const override {
        auto durations = m_state.thread_max();
        result_array result;
        std::transform(durations.begin(), durations.end(), std::back_inserter(result), [this](clock::duration d) {
            return m_bytes / (1024 * 1024 * 1.024 * to_ms(d));
        });
        return result;
    }

    virtual result_array imbalance() const override {
        auto durations_sum = m_state.thread_sum();
        auto durations_max = m_state.thread_max();
        int threads = m_state.active_threads();
        result_array result;

        std::transform(durations_sum.begin(),
            durations_sum.end(),
            durations_max.begin(),
            std::back_inserter(result),
            [this, threads](timer::clock::duration dsum, clock::duration dmax) {
                double sum = m_bytes / (1024 * 1024 * 1.024 * to_ms(dsum));
                double max = m_bytes / (1024 * 1024 * 1.024 * to_ms(dmax));
                return max * threads / sum - 1.0;
            });
        return result;
    }

  private:
    long long m_bytes;
};
