#include "variant_base.h"

#include <chrono>
#include <stdexcept>

#include "arguments.h"
#include "except.h"
#include "result.h"

namespace platform {

    variant_base::variant_base(const arguments_map &args)
        : m_halo(args.get<int>("halo")), m_alignment(args.get<int>("alignment")), m_isize(args.get<int>("i-size")),
          m_jsize(args.get<int>("j-size")), m_ksize(args.get<int>("k-size")), m_ilayout(args.get<int>("i-layout")),
          m_jlayout(args.get<int>("j-layout")), m_klayout(args.get<int>("k-layout")),
          m_data_offset(((m_halo + m_alignment - 1) / m_alignment) * m_alignment - m_halo) {
        if (m_isize <= 0 || m_jsize <= 0 || m_ksize <= 0)
            throw ERROR("invalid domain size");
        if (m_halo <= 0)
            throw ERROR("invalid m_halo size");
        if (m_alignment <= 0)
            throw ERROR("invalid alignment");

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
    }

    std::vector<result> variant_base::run(const std::string &stencil, int runs) {
        using clock = std::chrono::high_resolution_clock;
        constexpr int dry = 2;

        std::vector<std::string> stencils;
        if (stencil == "all")
            stencils = stencil_list();
        else
            stencils = {stencil};

        std::vector<result> results;

        for (const std::string &s : stencils) {
            auto f = stencil_function(s);
            result res(s);

            for (int i = 0; i < runs + dry; ++i) {
                prerun();

                auto tstart = clock::now();
                f();
                auto tend = clock::now();

                postrun();

                if (i == 0) {
                    if (!verify(s))
                        throw ERROR("result of stencil '" + s + "' is wrong");
                } else if (i >= dry) {
                    double t = std::chrono::duration<double>(tend - tstart).count();
                    res.push_back(t, touched_bytes(s) / (1024.0 * 1024.0 * 1024.0));
                }
            }

            results.push_back(res);
        }
        return results;
    }

} // namespace platform
