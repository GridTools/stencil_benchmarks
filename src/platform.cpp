#include "platform.h"
#include "except.h"
#include "platform_list.h"

#ifdef PLATFORM_KNL
#include "knl/knl_platform.h"
#endif

#ifdef PLATFORM_CUDA
#include "cuda/cuda_platform.h"
#endif

#ifdef PLATFORM_X86
#include "x86/x86_platform.h"
#endif

namespace platform {

    struct setuper {
        template <class Platform>
        static void execute(arguments &args) {
            Platform::setup(args);
        }
    };

    struct creator {
        template <class Platform>
        static void execute(const arguments_map &args, variant_base *&variant) {
            if (variant == nullptr)
                variant = Platform::create_variant(args);
        }
    };

#ifdef PLATFORM_X86
    using x86_pls = platform_list<x86::x86_standard>;
#else
    using x86_pls = platform_list<>;
#endif

#ifdef PLATFORM_KNL
    using knl_pls = platform_list<knl::flat, knl::cache>;
#else
    using knl_pls = platform_list<>;
#endif

#ifdef PLATFORM_CUDA
    using cuda_pls = platform_list<cuda::cuda>;
#else
    using cuda_pls = platform_list<>;
#endif

    using pls_tmp = merge_platform_list<knl_pls, cuda_pls>::type;
    using pls = merge_platform_list<pls_tmp, x86_pls>::type;

    void setup(arguments &args) { pls::loop<setuper>(args); }

    std::unique_ptr<variant_base> create_variant(const arguments_map &args) {
        variant_base *variant = nullptr;
        pls::loop<creator>(args, variant);
        if (!variant)
            throw ERROR("Error: variant '" + args.get("variant") + "' not found");
        return std::unique_ptr<variant_base>(variant);
    }

} // namespace platform
