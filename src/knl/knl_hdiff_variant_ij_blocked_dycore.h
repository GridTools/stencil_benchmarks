#pragma once

#include <algorithm>
#include <random>

#include <omp.h>

#include "knl/knl_hdiff_stencil_variant.h"
#include "knl/knl_platform.h"

namespace platform {

    namespace knl {

        template <class ValueType>
        class hdiff_variant_ij_blocked_dycore final : public knl_hdiff_stencil_variant<ValueType> {
          public:
            using value_type = ValueType;
            using allocator = typename knl::allocator<value_type>;

            hdiff_variant_ij_blocked_dycore(const arguments_map &args)
                : knl_hdiff_stencil_variant<ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
 
                m_caches.resize(omp_get_max_threads());

                const int alignment = args.get<int>("alignment");
                const int icachesize = ((m_iblocksize + 2 + alignment - 1) / alignment) * alignment;
                const int jcachesize = m_jblocksize + 2;
                m_jcachestride = icachesize;

                for (auto& cache : m_caches)
                    cache.resize(icachesize * jcachesize);

                m_crlato.resize(args.get<int>("j-size"));
                m_crlatu.resize(args.get<int>("j-size"));
                std::minstd_rand eng;
                std::uniform_real_distribution<value_type> dist(-100, 100);
                auto rand = std::bind(dist, std::ref(eng));
                std::generate(m_crlato.begin(), m_crlato.end(), rand);
                std::generate(m_crlatu.begin(), m_crlatu.end(), rand);
            }

            void hdiff() override;

          private:
            int m_iblocksize, m_jblocksize;
            int m_jcachestride;
            std::vector<std::vector<value_type, allocator>> m_caches;
            std::vector<value_type, allocator> m_crlato, m_crlatu;
        };

        extern template class hdiff_variant_ij_blocked_dycore<float>;
        extern template class hdiff_variant_ij_blocked_dycore<double>;

    } // namespace knl

} // namespace platform
