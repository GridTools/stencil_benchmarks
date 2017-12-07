#pragma once

#include "knl/knl_platform.h"
#include "knl/knl_vadv_variant.h"

namespace platform {

    namespace knl {

        template <class Platform, class ValueType>
        class vadv_variant_ij_blocked_split final : public knl_vadv_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;
            using platform = Platform;

            vadv_variant_ij_blocked_split(const arguments_map &args)
                : knl_vadv_stencil_variant<Platform, ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
            }
            ~vadv_variant_ij_blocked_split() {}

            void vadv() override;

          private:
            int m_iblocksize, m_jblocksize;
        };

        extern template class vadv_variant_ij_blocked_split<flat, float>;
        extern template class vadv_variant_ij_blocked_split<flat, double>;
        extern template class vadv_variant_ij_blocked_split<cache, float>;
        extern template class vadv_variant_ij_blocked_split<cache, double>;

    } // knl

} // namespace platform
