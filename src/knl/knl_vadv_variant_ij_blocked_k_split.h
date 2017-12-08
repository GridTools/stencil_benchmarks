#pragma once

#include "knl/knl_platform.h"
#include "knl/knl_vadv_variant.h"

namespace platform {

    namespace knl {

        template <class ValueType>
        class vadv_variant_ij_blocked_k_split final : public knl_vadv_stencil_variant<ValueType> {
          public:
            using value_type = ValueType;
            using platform = knl;

            vadv_variant_ij_blocked_k_split(const arguments_map &args)
                : knl_vadv_stencil_variant<ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
            }
            ~vadv_variant_ij_blocked_k_split() {}

            void vadv() override;

          private:
            int m_iblocksize, m_jblocksize;
        };

        extern template class vadv_variant_ij_blocked_k_split<float>;
        extern template class vadv_variant_ij_blocked_k_split<double>;

    } // knl

} // namespace platform
