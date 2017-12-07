#pragma once

#include "knl/knl_basic_multifield_variant.h"
#include "knl/knl_platform.h"

namespace platform {

    namespace knl {

        template <class Platform, class ValueType>
        class multifield_variant_ij_blocked final : public knl_basic_multifield_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;

            multifield_variant_ij_blocked(const arguments_map &args)
                : knl_basic_multifield_variant<Platform, ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
                if (this->fields() > 10)
                    throw ERROR("multifield variant supports only up to 10 fields");
            }

            void copy() override;
            void copyi() override;
            void copyj() override;
            void copyk() override;
            void avgi() override;
            void avgj() override;
            void avgk() override;
            void sumi() override;
            void sumj() override;
            void sumk() override;
            void lapij() override;

          private:
            int m_iblocksize, m_jblocksize;
        };

        extern template class multifield_variant_ij_blocked<flat, float>;
        extern template class multifield_variant_ij_blocked<flat, double>;
        extern template class multifield_variant_ij_blocked<cache, float>;
        extern template class multifield_variant_ij_blocked<cache, double>;

    } // namespace knl

} // namespace platform
