#include "cuda/cuda_platform.h"

#include "cuda/cuda_variant_ij_blocked.h"

namespace platform {

    namespace cuda {

        void cuda::setup(arguments &args) {
            arguments &pargs = args.command(name, "variant");
            pargs.command("ij-blocked")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
        }

        variant_base *cuda::create_variant(const arguments_map &args) {
            if (args.get("platform") != name)
                return nullptr;

            std::string prec = args.get("precision");
            std::string var = args.get("variant");

            if (prec == "single") {
                if (var == "ij-blocked")
                    return new variant_ij_blocked< cuda, float >(args);
            } else if (prec == "double") {
                if (var == "ij-blocked")
                    return new variant_ij_blocked< cuda, double >(args);
            }

            return nullptr;
        }

    } // namespace cuda

} // namespace platform
