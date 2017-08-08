#pragma once

#include <thread>

#include "variant.h"

namespace platform {

    namespace knl {

        template <class Platform, class ValueType>
        class knl_variant : public variant<Platform, ValueType> {
          public:
            knl_variant(const arguments_map &args) : variant<Platform, ValueType>(args) {}
            virtual ~knl_variant() {}

            void prerun() override { flush_cache(); }

            void postrun() override {}

          private:
            void flush_cache() {
#pragma omp parallel
                { std::this_thread::sleep_for(std::chrono::duration<double>(0.02)); }
            }
        };

    } // namespace knl

} // namespace platform
