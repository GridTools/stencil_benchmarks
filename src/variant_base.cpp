#include "variant_base.h"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <stdexcept>

#ifdef WITH_PAPI
#include <omp.h>
#include <papi.h>
#endif

#include "arguments.h"
#include "except.h"
#include "result.h"

namespace platform {

    variant_base::variant_base(const arguments_map &args)
        : m_halo(args.get<int>("halo")), m_alignment(args.get<int>("alignment")), m_isize(args.get<int>("i-size")),
          m_jsize(args.get<int>("j-size")), m_ksize(args.get<int>("k-size")), m_ilayout(args.get<int>("i-layout")),
          m_jlayout(args.get<int>("j-layout")), m_klayout(args.get<int>("k-layout")),
          m_data_offset(((m_halo + m_alignment - 1) / m_alignment) * m_alignment - m_halo),
          m_runs(args.get<int>("runs")) {
        if (m_isize <= 0 || m_jsize <= 0 || m_ksize <= 0)
            throw ERROR("invalid domain size");
        if (m_halo < 0)
            throw ERROR("invalid halo size");
        if (m_alignment <= 0)
            throw ERROR("invalid alignment");
        if (m_runs <= 0)
            throw ERROR("invalid number of runs");

        int ish = m_isize + 2 * m_halo;
        int jsh = m_jsize + 2 * m_halo;
        int ksh = m_ksize + 2 * m_halo;

        int s = 1;
        if (m_ilayout == 2) {
            m_istride = s;
            s *= ish;
        } else if (m_jlayout == 2) {
            m_jstride = s;
            s *= jsh;
        } else if (m_klayout == 2) {
            m_kstride = s;
            s *= ksh;
        } else {
            throw ERROR("invalid layout");
        }

        s = ((s + m_alignment - 1) / m_alignment) * m_alignment;

        if (m_ilayout == 1) {
            m_istride = s;
            s *= ish;
        } else if (m_jlayout == 1) {
            m_jstride = s;
            s *= jsh;
        } else if (m_klayout == 1) {
            m_kstride = s;
            s *= ksh;
        } else {
            throw ERROR("invalid layout");
        }

        if (m_ilayout == 0) {
            m_istride = s;
            s *= ish;
        } else if (m_jlayout == 0) {
            m_jstride = s;
            s *= jsh;
        } else if (m_klayout == 0) {
            m_kstride = s;
            s *= ksh;
        } else {
            throw ERROR("invalid layout");
        }

        m_storage_size = m_data_offset + s;

#ifdef WITH_PAPI
        if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
            throw ERROR("PAPI error: initialization failed");
        int ret = PAPI_event_name_to_code(const_cast<char *>(args.get("papi-event").c_str()), &m_papi_event_code);
        if (ret != PAPI_OK) {
            char *msg = PAPI_strerror(ret);
            if (msg != nullptr)
                throw ERROR("PAPI error, " + std::string(msg));
            else
                throw ERROR("unknown PAPI error");
        }
#endif
    }

    std::vector<result> variant_base::run(const std::string &stencil) {
        using clock = std::chrono::high_resolution_clock;
        constexpr int dry = 2;

#ifdef WITH_PAPI
        if (PAPI_num_counters() <= PAPI_OK)
            throw ERROR("PAPI not available");

        if (PAPI_thread_init(reinterpret_cast<unsigned long (*)()>(omp_get_thread_num)) != PAPI_OK)
            throw ERROR("PAPI thread init error");
#endif

        std::vector<std::string> stencils;
        if (stencil == "all")
            stencils = stencil_list();
        else
            stencils = {stencil};

        std::vector<result> results;

        for (const std::string &s : stencils) {
            auto f = stencil_function(s);
            result res(s);
            res.add_data("time", "ms");
            res.add_data("bandwidth", "GB/s");
#ifdef WITH_PAPI
            res.add_data("counter", "-");
            res.add_data("counter-imbalance", "-");
#endif

            for (int i = 0; i < dry; ++i) {
                prerun();
                f(i);
                postrun();

                if (i == 0) {
                    if (!verify(s))
                        throw ERROR("result of stencil '" + s + "' is wrong");
                }
            }

            setup();
            auto tstart = clock::now();

            for (int i = 0; i < m_runs; ++i) {
                prerun();

#ifdef WITH_PAPI
#pragma omp parallel
                {
                    if (PAPI_start_counters(&m_papi_event_code, 1) != PAPI_OK)
                        throw ERROR("PAPI error, could not start counters");
                }
#endif
                f(i);
#ifdef WITH_PAPI
                std::vector<long long> ctrs;
#pragma omp parallel shared(ctrs)
                {
                    long long ctr;
                    if (PAPI_stop_counters(&ctr, 1) != PAPI_OK)
                        throw ERROR("PAPI error, could not stop counters");
#pragma omp single
                    ctrs.resize(omp_get_num_threads());
#pragma omp barrier
                    ctrs[omp_get_thread_num()] = ctr;
                }
#endif
                postrun();
            }
            teardown();
            auto tend = clock::now();

            double t = std::chrono::duration<double>(tend - tstart).count() / (double)m_runs;
            double gb = touched_bytes(s) / 1.0e9;

#ifdef WITH_PAPI
            double ctrs_sum = std::accumulate(ctrs.begin(), ctrs.end(), 0ll);
            double ctr = ctrs_sum / ctrs.size();
            double ctr_imb = *std::max_element(ctrs.begin(), ctrs.end()) / (ctrs_sum / ctrs.size()) - 1.0;

            res.push_back(t * 1000.0, gb / t, ctr, ctr_imb);
#else
            res.push_back(t * 1000.0, gb / t);
#endif
            results.push_back(res);
        }
        return results;
    }

} // namespace platform
