#pragma once

#include <cuda_runtime.h>

#include "except.h"
#include "variant.h"

namespace platform {

    namespace cuda {

        template <class Platform, class ValueType>
        class cuda_variant : public variant<Platform, ValueType> {
          public:
            using value_type = ValueType;

            cuda_variant(const arguments_map &args) : variant<Platform, ValueType>(args) {}

            virtual ~cuda_variant() {}

            void prerun() override {}

            void postrun() override {}
        };

    } // namespace cuda

} // namespace platform
