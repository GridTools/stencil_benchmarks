#pragma once

#include <omp.h>

#include "knl/knl_hdiff_stencil_variant.h"
#include "knl/knl_platform.h"

namespace platform {

    namespace knl {

        template <class Platform, class ValueType>
        class hdiff_variant_ij_blocked_ddfused final : public knl_hdiff_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;

            hdiff_variant_ij_blocked_ddfused(const arguments_map &args)
                : knl_hdiff_stencil_variant<Platform, ValueType>(args), m_iblocks(args.get<int>("i-blocks")),
                  m_jblocks(args.get<int>("j-blocks")), m_kblocks(args.get<int>("k-blocks")) {}

            void hdiff() override;

          private:
            int m_iblocks, m_jblocks, m_kblocks;
        };

        extern template class hdiff_variant_ij_blocked_ddfused<flat, float>;
        extern template class hdiff_variant_ij_blocked_ddfused<flat, double>;
        extern template class hdiff_variant_ij_blocked_ddfused<cache, float>;
        extern template class hdiff_variant_ij_blocked_ddfused<cache, double>;

    } // namespace knl

} // namespace platform
