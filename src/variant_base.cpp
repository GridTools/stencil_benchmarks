#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "except.h"
#include "variant_base.h"

namespace platform {

    variant_base::variant_base(const arguments_map &args)
        : m_halo(args.get<int>("halo")), m_alignment(args.get<int>("alignment")), m_isize(args.get<int>("i-size")),
          m_jsize(args.get<int>("j-size")), m_ksize(args.get<int>("k-size")), m_ilayout(args.get<int>("i-layout")),
          m_jlayout(args.get<int>("j-layout")), m_klayout(args.get<int>("k-layout")),
          m_data_offset(((m_halo + m_alignment - 1) / m_alignment) * m_alignment - m_halo),
          m_runs(args.get<int>("runs")) {
        if (m_isize <= 0 || m_jsize <= 0 || m_ksize <= 0)
            throw ERROR("invalid domain size");
        if (m_halo <= 0)
            throw ERROR("invalid m_halo size");
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
    }

    void variant_base::run(const std::string &stencil, counter &ctr) {
        constexpr int dry = 2;

        auto f = stencil_function(stencil);

        for (int i = 0; i < m_runs + dry; ++i) {
            prerun();

            if (i == dry)
                ctr.clear();

            f(ctr);

            postrun();

            if (i == 0) {
                if (!verify(stencil))
                    throw ERROR("result of stencil '" + stencil + "' is wrong");
            }
        }
    }

} // namespace platform
