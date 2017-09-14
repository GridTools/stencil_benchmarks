#pragma once

#include <chrono>
#include <vector>

#include <omp.h>

#include "counter.h"
#include "counter_state.h"
#include "except.h"

class timer final : public counter {
    using clock = std::chrono::high_resolution_clock;
    using state = omp_counter_state<clock::duration, clock::time_point>;

  public:
    virtual ~timer() {}

    void start() override { m_state.start(clock::duration(0), clock::now()); }

    void pause() override { m_state.pause(clock::now()); }

    void resume() override { m_state.resume(clock::now()); }

    void stop() override { m_state.stop(clock::now()); }

    void clear() override { m_state.clear(); }

    result_array total() const override {
        auto durations = m_state.thread_max();
        result_array result;
        std::transform(durations.begin(), durations.end(), std::back_inserter(result), to_ms);
        return result;
    }

    result_array imbalance() const override {
        auto durations_sum = m_state.thread_sum();
        auto durations_max = m_state.thread_max();
        int threads = m_state.active_threads();
        std::vector<double> imbalances;

        std::transform(durations_sum.begin(),
            durations_sum.end(),
            durations_max.begin(),
            std::back_inserter(imbalances),
            [threads](clock::duration sum, clock::duration max) { return to_ms(max) * threads / to_ms(sum) - 1.0; });
        return imbalances;
    }

  private:
    static double to_ms(clock::duration d) { return std::chrono::duration<double>(d).count() * 1000.0; }

    state m_state;
};
