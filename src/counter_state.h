#pragma once

#include <algorithm>
#include <vector>

#include <omp.h>

#include "except.h"

template <class Duration, class TimePoint = Duration>
class alignas(64) counter_state {
    enum class state { stopped, running, paused };

  public:
    counter_state() : m_state(state::stopped) { clear(); }

    ~counter_state() {
        if (m_state == state::running || m_state == state::paused)
            throw ERROR("destroyed active counter");
    }

    void start(const Duration &initial_value, const TimePoint &current) {
        if (m_state != state::stopped)
            throw ERROR("can only start a stopped counter");
        m_totals.push_back(initial_value);
        m_start = current;
        m_state = state::running;
        m_run = true;
    }

    void pause(const TimePoint &current) {
        if (m_state != state::running)
            throw ERROR("can only pause a running counter");
        m_totals.back() += current - m_start;
        m_state = state::paused;
    }

    void resume(const TimePoint &current) {
        if (m_state != state::paused)
            throw ERROR("can only resume a paused timer");
        m_start = current;
        m_state = state::running;
    }

    void stop(const TimePoint &current) {
        if (m_state != state::running)
            throw ERROR("can only stop a running counter");
        m_totals.back() += current - m_start;
        m_state = state::stopped;
    }

    const std::vector<Duration> &values() const {
        if (m_state != state::stopped)
            throw ERROR("can only get value of a stopped counter");
        if (!m_run)
            throw ERROR("counter was never running");
        return m_totals;
    }

    bool has_run() const { return m_run; }

    void clear() {
        if (m_state != state::stopped)
            throw ERROR("can only clear a stopped counter");
        m_totals.clear();
        m_totals.reserve(30);
        m_run = false;
    }

  private:
    std::vector<Duration> m_totals;
    TimePoint m_start;
    state m_state;
    bool m_run;
};

template <class Duration, class TimePoint = Duration>
class omp_counter_state {
    using thread_state = counter_state<Duration, TimePoint>;

  public:
    omp_counter_state() : m_thread_states(omp_get_max_threads()) {}

    void start(const Duration &initial_value, const TimePoint &current) { state().start(initial_value, current); }

    void pause(const TimePoint &current) { state().pause(current); }

    void resume(const TimePoint &current) { state().resume(current); }

    void stop(const TimePoint &current) { state().stop(current); }

    int active_threads() const {
        int r = 0;
        for (auto &s : m_thread_states)
            r += s.has_run();
        return r;
    }

    void clear() {
        for (auto &s : m_thread_states)
            s.clear();
    }

    std::vector<Duration> thread_sum() const {
        std::vector<Duration> result(m_thread_states.at(0).values().size());
        for (auto &s : m_thread_states) {
            if (s.has_run()) {
                auto &v = s.values();
                std::transform(v.begin(), v.end(), result.begin(), result.begin(), std::plus<Duration>());
            }
        }
        return result;
    }

    std::vector<Duration> thread_max() const {
        std::vector<Duration> result(m_thread_states.at(0).values().size());
        for (auto &s : m_thread_states) {
            if (s.has_run()) {
                auto &v = s.values();
                std::transform(v.begin(), v.end(), result.begin(), result.begin(), [](Duration a, Duration b) {
                    return a > b ? a : b;
                });
            }
        }
        return result;
    }

  private:
    thread_state &state() { return m_thread_states.at(omp_get_thread_num()); }

    std::vector<thread_state> m_thread_states;
};
