#pragma once

#include "knl/knl_hdiff_stencil_variant.h"
#include "knl/knl_platform.h"
#include "omp.h"

namespace platform {

    namespace knl {

        template <class ValueType>
        class hdiff_variant_ij_private_blocks final : public knl_hdiff_stencil_variant<ValueType> {
          public:
            using value_type = ValueType;

            hdiff_variant_ij_private_blocks(const arguments_map &args)
                : knl_hdiff_stencil_variant<ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
                int num_threads;
#pragma omp parallel
                { num_threads = omp_get_num_threads(); }
                lap_data.resize(num_threads * (m_iblocksize + 2) * (m_jblocksize + 2));
                flx_data.resize(num_threads * (m_iblocksize + 1) * (m_jblocksize));
                fly_data.resize(num_threads * (m_iblocksize) * (m_jblocksize + 1));
            }

            void hdiff() override;

          private:
            int m_iblocksize, m_jblocksize;
            std::vector<value_type> lap_data, flx_data, fly_data;
        };

        extern template class hdiff_variant_ij_private_blocks<float>;
        extern template class hdiff_variant_ij_private_blocks<double>;

    } // namespace knl

} // namespace platform
