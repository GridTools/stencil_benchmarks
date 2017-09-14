#pragma once

#include <limits>
#include <string>
#include <vector>

#include <omp.h>

#ifdef WITH_PAPI
#include <papi.h>
#endif

#include "counter.h"
#include "counter_state.h"
#include "except.h"

class papi_counter final : public counter {
    using state = omp_counter_state<long long>;

  public:
    papi_counter(const std::string &event) {
#ifdef WITH_PAPI
        if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
            throw ERROR("PAPI error: initialization failed");
        int ret = PAPI_event_name_to_code(const_cast<char *>(event.c_str()), &m_event_code);
        if (ret != PAPI_OK) {
            char *msg = PAPI_strerror(ret);
            if (msg != nullptr)
                throw ERROR("PAPI error, " + std::string(msg));
            else
                throw ERROR("unknown PAPI error");
        }
        if (PAPI_num_counters() <= PAPI_OK)
            throw ERROR("no PAPI counter available");

        if (PAPI_thread_init(pthread_self) != PAPI_OK)
            throw ERROR("PAPI thread init error");

        for (thread_data &td : m_data)
            td = {0, false, false, false};
#else
        throw ERROR("executable built without PAPI support");
#endif
    }

    virtual ~papi_counter() {}

    void start() override {
        m_state.start(0, 0);
#ifdef WITH_PAPI
        if (PAPI_register_thread() != PAPI_OK)
            throw ERROR("PAPI error, could not register thread");
        if (PAPI_start_counters(&m_event_code, 1) != PAPI_OK)
            throw ERROR("could not start PAPI counters");
#endif
    }

    void pause() override {
        long long ctr = 0;
#ifdef WITH_PAPI
        if (PAPI_stop_counters(&ctr, 1) != PAPI_OK)
            throw ERROR("could not stop PAPI counters");
#endif
        m_state.pause(ctr);
    }

    void resume() override {
        m_state.resume(0);
#ifdef WITH_PAPI
        if (PAPI_start_counters(&m_event_code, 1) != PAPI_OK)
            throw ERROR("could not start PAPI counters");
#endif
    }

    void stop() override {
        long long ctr = 0;
#ifdef WITH_PAPI
        if (PAPI_stop_counters(&ctr, 1) != PAPI_OK)
            throw ERROR("could not stop PAPI counters");
        if (PAPI_unregister_thread() != PAPI_OK)
            throw ERROR("PAPI error, could not unregister thread");
#endif
        m_state.stop(ctr);
    }

    void clear() override { m_state.clear(); }

    result_array total() const override {
        auto counts = m_state.thread_sum();
        result_array result;
        std::transform(
            counts.begin(), counts.end(), std::back_inserter(result), [](long long count) { return double(count); });
        return result;
    }

    result_array imbalance() const override {
        auto counts_sum = m_state.thread_sum();
        auto counts_max = m_state.thread_max();
        int threads = m_state.active_threads();
        std::vector<double> imbalances;

        std::transform(counts_sum.begin(),
            counts_sum.end(),
            counts_max.begin(),
            std::back_inserter(imbalances),
            [threads](long long sum, long long max) { return double(max * threads) / sum - 1.0; });
        return imbalances;
    }

  private:
    int m_event_code;
    state m_state;
};
