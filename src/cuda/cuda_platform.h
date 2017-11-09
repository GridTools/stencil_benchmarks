#pragma once

#include "arguments.h"
#include "cuda/cuda_allocator.h"
#include "variant_base.h"

namespace platform {

    namespace cuda {

        struct cuda {
            static constexpr const char *name = "cuda";

            template <class ValueType>
            using allocator = managed_allocator<ValueType>;

            static void setup(arguments &args);

            static variant_base *create_variant(const arguments_map &args);

            static void limit_blocksize(int &iblocksize, int &jblocksize);
            static void limit_blocksize(int &iblocksize, int &jblocksize, int &kblocksize);
        };

    } // namespace cuda

} // namespace platform
