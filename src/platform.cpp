#include "platform.h"
#include "except.h"
#include "platform_list.h"

#ifdef PLATFORM_KNL
#include "knl/knl_platform.h"
#endif

#ifdef PLATFORM_CUDA
#include "cuda/cuda_platform.h"
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

#ifdef PLATFORM_KNL
    using knl_pls = platform_list<knl::knl>;
#else
    using knl_pls = platform_list<>;
#endif

#ifdef PLATFORM_CUDA
    using cuda_pls = platform_list<cuda::cuda>;
#else
    using cuda_pls = platform_list<>;
#endif

    using pls = merge_platform_list<knl_pls, cuda_pls>::type;

    void setup(arguments &args) { pls::loop<setuper>(args); }

    std::unique_ptr<variant_base> create_variant(const arguments_map &args) {
        variant_base *variant = nullptr;
        pls::loop<creator>(args, variant);
        if (!variant)
            throw ERROR("Error: variant '" + args.get("variant") + "' not found");
        return std::unique_ptr<variant_base>(variant);
    }

} // namespace platform
