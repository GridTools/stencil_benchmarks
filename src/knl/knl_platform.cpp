#include <chrono>

#include "knl/knl_platform.h"

#include "knl/knl_hdiff_variant_ij_blocked_k_innermost.h"
#include "knl/knl_hdiff_variant_ij_blocked_k_outermost.h"
#include "knl/knl_hdiff_variant_ij_blocked_non_red.h"
#include "knl/knl_multifield_variant_1d_nontemporal.h"
#include "knl/knl_multifield_variant_ij_blocked.h"
#include "knl/knl_variant_1d.h"
#include "knl/knl_variant_1d_nontemporal.h"
#include "knl/knl_variant_ij_blocked.h"
#include "knl/knl_variant_ijk_blocked.h"

namespace platform {

    namespace knl {

        void knl_platform_base::flush_cache() {
#pragma omp parallel
            { std::this_thread::sleep_for(std::chrono::duration<double>(0.02)); }
        }

        void knl_platform_base::check_cache_conflicts(const std::string &stride_name, std::ptrdiff_t byte_stride) {
            // no conflicts if stride is smaller than cache line
            auto line_diff = byte_stride & ~std::ptrdiff_t(0x3f);
            if (line_diff != 0) {
                // check bits 11:6 for L1 set conflicts
                auto l1_set_diff = byte_stride & std::ptrdiff_t(0xfc0);
                if (l1_set_diff == 0)
                    std::cerr << "Warning: possible L1 set conflicts for " << stride_name << std::endl;

                // check bits 16:6 for L2 set conflicts
                auto l2_set_diff = byte_stride & std::ptrdiff_t(0xffc0);
                if (l2_set_diff == 0)
                    std::cerr << "Warning: possible L2 set conflicts for " << stride_name << std::endl;
            }

            // no TLB conflicts for byte_stride smaller than page size
            auto page_diff = byte_stride & ~std::ptrdiff_t(0xfff);
            if (page_diff != 0) {
                // check bits 16:12 for 4KB page TLB conflicts
                auto page_set_diff = byte_stride & std::ptrdiff_t(0x1f000);
                if (page_set_diff == 0)
                    std::cerr << "Warning: possible TLB set conflicts for " << stride_name << std::endl;
            }
        }

        namespace {
            template <class Platform>
            void common_setup(arguments &args) {
                arguments &pargs = args.command(Platform::name, "variant");
                pargs.command("1d");
                pargs.command("1d-nontemporal");
                pargs.command("ij-blocked")
                    .add("i-blocksize", "block size in i-direction", "32")
                    .add("j-blocksize", "block size in j-direction", "8");
                pargs.command("ijk-blocked")
                    .add("i-blocksize", "block size in i-direction", "32")
                    .add("j-blocksize", "block size in j-direction", "8")
                    .add("k-blocksize", "block size in k-direction", "8");
                pargs.command("hdiff-ij-blocked-k-innermost")
                    .add("i-blocksize", "block size in i-direction", "32")
                    .add("j-blocksize", "block size in j-direction", "8");
                pargs.command("hdiff-ij-blocked-k-outermost")
                    .add("i-blocksize", "block size in i-direction", "32")
                    .add("j-blocksize", "block size in j-direction", "8");
                pargs.command("hdiff-ij-blocked-non-red")
                    .add("i-blocksize", "block size in i-direction", "32")
                    .add("j-blocksize", "block size in j-direction", "8");
                pargs.command("multifield-1d-nontemporal").add("fields", "number of fields", "5");
                pargs.command("multifield-ij-blocked")
                    .add("fields", "number of fields", "5")
                    .add("i-blocksize", "block size in i-direction", "32")
                    .add("j-blocksize", "block size in j-direction", "8");
            }

            template <class Platform>
            variant_base *common_create_variant(const arguments_map &args) {
                if (args.get("platform") != Platform::name)
                    return nullptr;

                std::string prec = args.get("precision");
                std::string var = args.get("variant");

                if (prec == "single") {
                    if (var == "1d")
                        return new variant_1d<Platform, float>(args);
                    if (var == "1d-nontemporal")
                        return new variant_1d_nontemporal<Platform, float>(args);
                    if (var == "ij-blocked")
                        return new variant_ij_blocked<Platform, float>(args);
                    if (var == "ijk-blocked")
                        return new variant_ijk_blocked<Platform, float>(args);
                    if (var == "hdiff-ij-blocked-k-innermost")
                        return new knl_hdiff_variant_ij_blocked_k_innermost<Platform, float>(args);
                    if (var == "hdiff-ij-blocked-k-outermost")
                        return new knl_hdiff_variant_ij_blocked_k_outermost<Platform, float>(args);
                    if (var == "hdiff-ij-blocked-non-red")
                        return new knl_hdiff_variant_ij_blocked_non_red<Platform, float>(args);
                    if (var == "multifield-1d-nontemporal")
                        return new multifield_variant_1d_nontemporal<Platform, float>(args);
                    if (var == "multifield-ij-blocked")
                        return new multifield_variant_ij_blocked<Platform, float>(args);
                } else if (prec == "double") {
                    if (var == "1d")
                        return new variant_1d<Platform, double>(args);
                    if (var == "1d-nontemporal")
                        return new variant_1d_nontemporal<Platform, double>(args);
                    if (var == "ij-blocked")
                        return new variant_ij_blocked<Platform, double>(args);
                    if (var == "ijk-blocked")
                        return new variant_ijk_blocked<Platform, double>(args);
                    if (var == "hdiff-ij-blocked-k-innermost")
                        return new knl_hdiff_variant_ij_blocked_k_innermost<Platform, double>(args);
                    if (var == "hdiff-ij-blocked-k-outermost")
                        return new knl_hdiff_variant_ij_blocked_k_outermost<Platform, double>(args);
                    if (var == "hdiff-ij-blocked-non-red")
                        return new knl_hdiff_variant_ij_blocked_non_red<Platform, double>(args);
                    if (var == "multifield-1d-nontemporal")
                        return new multifield_variant_1d_nontemporal<Platform, double>(args);
                    if (var == "multifield-ij-blocked")
                        return new multifield_variant_ij_blocked<Platform, double>(args);
                }

                return nullptr;
            }
        }

        void flat::setup(arguments &args) { common_setup<flat>(args); }

        variant_base *flat::create_variant(const arguments_map &args) { return common_create_variant<flat>(args); }

        void cache::setup(arguments &args) { common_setup<cache>(args); }

        variant_base *cache::create_variant(const arguments_map &args) { return common_create_variant<cache>(args); }

    } // namespace knl

} // namespace platform
