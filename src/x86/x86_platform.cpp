#include <chrono>

#include "x86/x86_platform.h"

#include "x86/x86_variant_1d.h"
#include "x86/x86_hdiff_variant_simple.h"
#include "x86/x86_hdiff_variant_ij_blocked.h"
#include "x86/x86_hdiff_variant_k_outermost.h"

namespace platform {

    namespace x86 {

        void x86_platform_base::flush_cache() {
#pragma omp parallel
            { std::this_thread::sleep_for(std::chrono::duration<double>(0.02)); }
        }

        void x86_platform_base::check_cache_conflicts(const std::string &stride_name, std::ptrdiff_t byte_stride) {
            //TODO: implement cache conflict check
        }

        void x86_standard::setup(arguments &args) {
            arguments &pargs = args.command(name, "variant");
            pargs.command("1d");
            pargs.command("hdiff-simple");
            pargs.command("hdiff-ij-blocked")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
            pargs.command("hdiff-k-outermost")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
        }

        variant_base *x86_standard::create_variant(const arguments_map &args) {
            if (args.get("platform") != name)
                return nullptr;
        
            std::string prec = args.get("precision");
            std::string var = args.get("variant");

            if (prec == "single") {
                if (var == "1d")
                    return new variant_1d<x86_standard, float>(args);
                if (var == "hdiff-simple")
                    return new hdiff_variant_simple<x86_standard, float>(args);
                if (var == "hdiff-ij-blocked")
                    return new hdiff_variant_ij_blocked<x86_standard, float>(args);
                if (var == "hdiff-k-outermost")
                    return new hdiff_variant_k_outermost<x86_standard, float>(args);
            } else if (prec == "double") {
                if (var == "1d")
                    return new variant_1d<x86_standard, double>(args);
                if (var == "hdiff-simple")
                    return new hdiff_variant_simple<x86_standard, double>(args);
                if (var == "hdiff-ij-blocked")
                    return new hdiff_variant_ij_blocked<x86_standard, double>(args);
                if (var == "hdiff-k-outermost")
                    return new hdiff_variant_k_outermost<x86_standard, double>(args);
            }

            return nullptr;
        }

    } // namespace x86

} // namespace platform
