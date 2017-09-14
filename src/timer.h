#pragma once

#include <chrono>
#include <vector>

#include <omp.h>

#include "counter.h"
#include "counter_state.h"
#include "except.h"

class timer : public counter {
  protected:
    using clock = std::chrono::high_resolution_clock;
    using state = omp_counter_state<clock::duration, clock::time_point>;

  public:
    virtual ~timer() {}

    virtual void start() override { m_state.start(clock::duration(0), clock::now()); }

    virtual void pause() override { m_state.pause(clock::now()); }

    virtual void resume() override { m_state.resume(clock::now()); }

    virtual void stop() override { m_state.stop(clock::now()); }

    virtual void clear() override { m_state.clear(); }

    virtual result_array total() const override {
        auto durations = m_state.thread_max();
        result_array result;
        std::transform(durations.begin(), durations.end(), std::back_inserter(result), to_ms);
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
            [threads](clock::duration sum, clock::duration max) { return to_ms(max) * threads / to_ms(sum) - 1.0; });
        return result;
    }

    virtual int threads() const override { return m_state.active_threads(); }

    virtual result_array thread_total(int thread) const override {
        auto durations = m_state.thread_values(thread);
        result_array result;
        std::transform(durations.begin(), durations.end(), std::back_inserter(result), to_ms);
        return result;
    }

  protected:
    static double to_ms(clock::duration d) { return std::chrono::duration<double>(d).count() * 1000.0; }

    state m_state;
};
