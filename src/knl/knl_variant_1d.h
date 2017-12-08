#pragma once

#include "knl/knl_basic_stencil_variant.h"
#include "knl/knl_platform.h"

namespace platform {

    namespace knl {

        template <class ValueType>
        class variant_1d final : public knl_basic_stencil_variant<ValueType> {
          public:
            using value_type = ValueType;

            variant_1d(const arguments_map &args) : knl_basic_stencil_variant<ValueType>(args) {}

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
        };

        extern template class variant_1d<float>;
        extern template class variant_1d<double>;

    } // namespace knl

} // namespace platform
