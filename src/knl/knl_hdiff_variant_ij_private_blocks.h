#pragma once

#include "knl/knl_hdiff_stencil_variant.h"
#include "knl/knl_platform.h"

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
            }

            void hdiff() override;
			void prerun_init() override;

          private:
            int m_iblocksize, m_jblocksize;
        };

        extern template class hdiff_variant_ij_private_blocks<float>;
        extern template class hdiff_variant_ij_private_blocks<double>;

    } // namespace knl

} // namespace platform
