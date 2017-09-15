#pragma once

#include "x86/x86_fast_waves_uv_stencil_variant.h"

namespace platform {

    namespace x86 {

        template <class Platform, class ValueType>
        class x86_fast_waves_uv_variant_ij_blocked final : public x86_fast_waves_uv_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;

            x86_fast_waves_uv_variant_ij_blocked(const arguments_map &args)
                : x86_fast_waves_uv_stencil_variant<Platform, ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
            }

            void fast_waves_uv() override {
                throw ERROR("implement me");
            }

          private:
            int m_iblocksize, m_jblocksize;
        };

    } // namespace x86

} // namespace platform
