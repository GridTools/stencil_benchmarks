#pragma once

#include <thread>

#include "hdiff_stencil_variant.h"

namespace platform {

    namespace x86 {

        template <class Platform, class ValueType>
        class x86_hdiff_stencil_variant : public hdiff_stencil_variant<Platform, ValueType> {
          public:
            x86_hdiff_stencil_variant(const arguments_map &args) : hdiff_stencil_variant<Platform, ValueType>(args) {
                Platform::check_cache_conflicts("i-stride offsets", this->istride() * this->bytes_per_element());
                Platform::check_cache_conflicts("j-stride offsets", this->jstride() * this->bytes_per_element());
                Platform::check_cache_conflicts("k-stride offsets", this->kstride() * this->bytes_per_element());
                Platform::check_cache_conflicts(
                    "2 * i-stride offsets", 2 * this->istride() * this->bytes_per_element());
                Platform::check_cache_conflicts(
                    "2 * j-stride offsets", 2 * this->jstride() * this->bytes_per_element());
                Platform::check_cache_conflicts(
                    "2 * k-stride offsets", 2 * this->kstride() * this->bytes_per_element());
            }
            virtual ~x86_hdiff_stencil_variant() {}

            void prerun() override {
                hdiff_stencil_variant<Platform, ValueType>::prerun();
                Platform::flush_cache();
            }
        };

    } // namespace x86

} // namespace platform
