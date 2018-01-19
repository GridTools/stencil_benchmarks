#include <chrono>
#include <thread>

#include "knl/knl_platform.h"

#include "knl/knl_hdiff_variant_ij_blocked_ddfused.h"
#include "knl/knl_hdiff_variant_ij_blocked_fused.h"
#include "knl/knl_hdiff_variant_ij_blocked_k_innermost.h"
#include "knl/knl_hdiff_variant_ij_blocked_k_outermost.h"
#include "knl/knl_hdiff_variant_ij_blocked_non_red.h"
#include "knl/knl_hdiff_variant_ij_blocked_private_halo.h"
#include "knl/knl_hdiff_variant_ij_blocked_stacked_layout.h"
#include "knl/knl_multifield_variant_1d_nontemporal.h"
#include "knl/knl_multifield_variant_ij_blocked.h"
#include "knl/knl_vadv_variant_2d.h"
#include "knl/knl_vadv_variant_ij_blocked.h"
#include "knl/knl_vadv_variant_ij_blocked_colopt.h"
#include "knl/knl_vadv_variant_ij_blocked_k.h"
#include "knl/knl_vadv_variant_ij_blocked_k_ring.h"
#include "knl/knl_vadv_variant_ij_blocked_k_split.h"
#include "knl/knl_vadv_variant_ij_blocked_split.h"
#include "knl/knl_variant_1d.h"
#include "knl/knl_variant_1d_nontemporal.h"
#include "knl/knl_variant_ij_blocked.h"
#include "knl/knl_variant_ijk_blocked.h"

namespace platform {

    namespace knl {

        void knl::flush_cache() {
#ifdef KNL_CLASSIC_CFLUSHER
            constexpr int cache_size = 1 * 1024 * 1024;
            constexpr int n = cache_size / sizeof(float);
            std::vector<float, flat_allocator<float>> a(n), b(n);
            volatile float *a_ptr = a.data();
            volatile float *b_ptr = b.data();
#pragma omp parallel
            {
                std::minstd_rand eng(13 * omp_get_thread_num());
                std::uniform_int_distribution<int> dist(0, n - 1);
                const int offset = dist(eng);
#pragma vector nontemporal
                for (int i = 0; i < n; ++i) {
                    b_ptr[i] = a_ptr[offset];
                }

                _mm_mfence();
            }
#else
#pragma omp parallel
            { std::this_thread::sleep_for(std::chrono::duration<double>(0.02)); }
#endif
        }

        void knl::check_cache_conflicts(const std::string &stride_name, std::ptrdiff_t byte_stride) {
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

            template <class ValueType>
            variant_base *create_variant_by_prec(const arguments_map &args) {
                std::string grp = args.get("group");
                std::string var = args.get("variant");
                if (grp == "basic") {
                    if (var == "1d")
                        return new variant_1d<ValueType>(args);
                    if (var == "1d-nontemporal")
                        return new variant_1d_nontemporal<ValueType>(args);
                    if (var == "ij-blocked")
                        return new variant_ij_blocked<ValueType>(args);
                    if (var == "ijk-blocked")
                        return new variant_ijk_blocked<ValueType>(args);
                }
                if (grp == "hdiff") {
                    if (var == "ij-blocked-k-innermost")
                        return new hdiff_variant_ij_blocked_k_innermost<ValueType>(args);
                    if (var == "ij-blocked-k-outermost")
                        return new hdiff_variant_ij_blocked_k_outermost<ValueType>(args);
                    if (var == "ij-blocked-non-red")
                        return new hdiff_variant_ij_blocked_non_red<ValueType>(args);
                    if (var == "ij-blocked-private-halo")
                        return new hdiff_variant_ij_blocked_private_halo<ValueType>(args);
                    if (var == "ij-blocked-stacked-layout")
                        return new hdiff_variant_ij_blocked_stacked_layout<ValueType>(args);
                    if (var == "ij-blocked-fused")
                        return new hdiff_variant_ij_blocked_fused<ValueType>(args);
                    if (var == "ij-blocked-ddfused")
                        return new hdiff_variant_ij_blocked_ddfused<ValueType>(args);
                }
                if (grp == "multifield") {
                    if (var == "1d-nontemporal")
                        return new multifield_variant_1d_nontemporal<ValueType>(args);
                    if (var == "ij-blocked")
                        return new multifield_variant_ij_blocked<ValueType>(args);
                }
                if (grp == "vadv") {
                    if (var == "2d")
                        return new vadv_variant_2d<ValueType>(args);
                    if (var == "ij-blocked")
                        return new vadv_variant_ij_blocked<ValueType>(args);
                    if (var == "ij-blocked-split")
                        return new vadv_variant_ij_blocked_split<ValueType>(args);
                    if (var == "ij-blocked-colopt")
                        return new vadv_variant_ij_blocked_colopt<ValueType>(args);
                    if (var == "ij-blocked-k")
                        return new vadv_variant_ij_blocked_k<ValueType>(args);
                    if (var == "ij-blocked-k-ring")
                        return new vadv_variant_ij_blocked_k_ring<ValueType>(args);
                    if (var == "ij-blocked-k-split")
                        return new vadv_variant_ij_blocked_k_split<ValueType>(args);
                }
                return nullptr;
            }
        }

        void knl::setup(arguments &args) {
            auto &basic = args.command("basic", "variant");
            basic.command("1d");
            basic.command("1d-nontemporal");
            basic.command("ij-blocked")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
            basic.command("ijk-blocked")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8")
                .add("k-blocksize", "block size in k-direction", "8");
            auto &hdiff = args.command("hdiff", "variant");
            hdiff.command("ij-blocked-k-innermost")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
            hdiff.command("ij-blocked-k-outermost")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
            hdiff.command("ij-blocked-non-red")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
            hdiff.command("ij-blocked-private-halo")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
            hdiff.command("ij-blocked-stacked-layout")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
            hdiff.command("ij-blocked-fused")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
            hdiff.command("ij-blocked-ddfused")
                .add("i-blocks", "blocks in i-direction", "16")
                .add("j-blocks", "blocks in j-direction", "8")
                .add("k-blocks", "blocks in k-direction", "1");
            auto &multifield = args.command("multifield", "variant");
            multifield.command("1d-nontemporal").add("fields", "number of fields", "5");
            multifield.command("ij-blocked")
                .add("fields", "number of fields", "5")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
            auto &vadv = args.command("vadv", "variant");
            vadv.command("2d");
            vadv.command("ij-blocked")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
            vadv.command("ij-blocked-split")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
            vadv.command("ij-blocked-colopt")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
            vadv.command("ij-blocked-k")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
            vadv.command("ij-blocked-k-ring")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
            vadv.command("ij-blocked-k-split")
                .add("i-blocksize", "block size in i-direction", "32")
                .add("j-blocksize", "block size in j-direction", "8");
        }

        variant_base *knl::create_variant(const arguments_map &args) {
            std::string prec = args.get("precision");

            if (prec == "single") {
                return create_variant_by_prec<float>(args);
            } else if (prec == "double") {
                return create_variant_by_prec<double>(args);
            }

            return nullptr;
        }

    } // namespace knl

} // namespace platform
