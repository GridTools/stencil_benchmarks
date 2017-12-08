#pragma once

#include "basic_multifield_variant.h"
#include "knl_platform.h"

namespace platform {

    namespace knl {

        template <class ValueType>
        class knl_basic_multifield_variant : public basic_multifield_variant<knl, ValueType> {
          public:
            knl_basic_multifield_variant(const arguments_map &args) : basic_multifield_variant<knl, ValueType>(args) {
                knl::check_cache_conflicts("i-stride offsets", this->istride() * this->bytes_per_element());
                knl::check_cache_conflicts("j-stride offsets", this->jstride() * this->bytes_per_element());
                knl::check_cache_conflicts("k-stride offsets", this->kstride() * this->bytes_per_element());
                knl::check_cache_conflicts("2 * i-stride offsets", 2 * this->istride() * this->bytes_per_element());
                knl::check_cache_conflicts("2 * j-stride offsets", 2 * this->jstride() * this->bytes_per_element());
                knl::check_cache_conflicts("2 * k-stride offsets", 2 * this->kstride() * this->bytes_per_element());
            }
            virtual ~knl_basic_multifield_variant() {}

            void prerun() override {
                basic_multifield_variant<knl, ValueType>::prerun();
                knl::flush_cache();
            }
        };

    } // namespace knl

} // namespace platform
