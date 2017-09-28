#pragma once

#include <thread>

#include "fast_waves_uv_variant.h"

namespace platform {

    namespace x86 {

        template <class Platform, class ValueType>
        class x86_fast_waves_uv_stencil_variant : public fast_waves_uv_variant<Platform, ValueType> {
          public:
            x86_fast_waves_uv_stencil_variant(const arguments_map &args)
                : fast_waves_uv_variant<Platform, ValueType>(args) {
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
            virtual ~x86_fast_waves_uv_stencil_variant() {}

            void prerun() override {
                fast_waves_uv_variant<Platform, ValueType>::prerun();
                Platform::flush_cache();
            }
        };

    } // namespace x86

} // namespace platform
