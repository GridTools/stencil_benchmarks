#pragma once

#include "knl/knl_platform.h"
#include "knl/knl_vadv_variant.h"
#include "omp.h"

namespace platform {

    namespace knl {

        template <class ValueType>
        class vadv_variant_ik_blocked_j final : public knl_vadv_stencil_variant<ValueType> {
          public:
            using value_type = ValueType;
            using platform = knl;

            vadv_variant_ik_blocked_j(const arguments_map &args)
                : knl_vadv_stencil_variant<ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
				//set sizes of caches 
				int num_threads; 
#pragma omp parallel 
				{
					num_threads = omp_get_num_threads(); 
				}
				m_ccol_cache.resize(num_threads * m_iblocksize * m_jblocksize);
				m_dcol_cache.resize(num_threads * m_iblocksize * m_jblocksize);
            }
            ~vadv_variant_ik_blocked_j() {}

            void vadv() override;

          private:
            static constexpr value_type dtr_stage = 3.0 / 20.0;
            static constexpr value_type beta_v = 0;
            static constexpr value_type bet_m = 0.5 * (1.0 - beta_v);
            static constexpr value_type bet_p = 0.5 * (1.0 + beta_v);
			//allocate per-thread private ccol_cache and dcol_cache
			std::vector<value_type> m_ccol_cache, m_dcol_cache; 

#pragma omp declare simd linear(i) uniform( \
    j, ccol, dcol, datacol, upos, utensstage, isize, jsize, ksize, istride, jstride, kstride)
            __attribute__((always_inline)) inline void backward_sweep_kmax(const int i,
                const int j,
                const value_type *__restrict__ ccol,
                const value_type *__restrict__ dcol,
                value_type *__restrict__ datacol,
                const value_type *__restrict__ upos,
                value_type *__restrict__ utensstage,
                const int isize,
                const int jsize,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride) {

                const int k = ksize - 1;
                const int index = i * istride + j * jstride + k * kstride;
                const int datacol_index = i * istride + j * jstride;
                datacol[datacol_index] = dcol[index];
                utensstage[index] = dtr_stage * (datacol[datacol_index] - upos[index]);
            }

#pragma omp declare simd linear(i) uniform( \
    j, k, ccol, dcol, datacol, upos, utensstage, isize, jsize, ksize, istride, jstride, kstride)
            __attribute__((always_inline)) inline void backward_sweep_kbody(const int i,
                const int j,
                const int k,
                const value_type *__restrict__ ccol,
                const value_type *__restrict__ dcol,
                value_type *__restrict__ datacol,
                const value_type *__restrict__ upos,
                value_type *__restrict__ utensstage,
                const int isize,
                const int jsize,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride) {

                const int index = i * istride + j * jstride + k * kstride;
                const int datacol_index = i * istride + j * jstride;
                datacol[datacol_index] = dcol[index] - ccol[index] * datacol[datacol_index];
                utensstage[index] = dtr_stage * (datacol[datacol_index] - upos[index]);
            }

#pragma omp declare simd linear(i) uniform( \
    j, k, ccol, dcol, datacol, upos, utensstage, isize, jsize, ksize, istride, jstride, kstride)
            __attribute__((always_inline)) inline void backward_sweep_k(const int i,
                const int j,
                const int k,
                const value_type *__restrict__ ccol,
                const value_type *__restrict__ dcol,
                value_type *__restrict__ datacol,
                const value_type *__restrict__ upos,
                value_type *__restrict__ utensstage,
                const int isize,
                const int jsize,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride) {
                constexpr value_type dtr_stage = 3.0 / 20.0;

                if (k == ksize - 1) {
                    backward_sweep_kmax(
                        i, j, ccol, dcol, datacol, upos, utensstage, isize, jsize, ksize, istride, jstride, kstride);
                } else {
                    backward_sweep_kbody(
                        i, j, k, ccol, dcol, datacol, upos, utensstage, isize, jsize, ksize, istride, jstride, kstride);
                }
            }

#pragma omp declare simd linear(i) uniform(j,          \
                                           ishift,     \
                                           jshift,     \
                                           ccol,       \
                                           ccol_cache, \
                                           dcol,       \
                                           dcol_cache, \
                                           wcon,       \
                                           ustage,     \
                                           upos,       \
                                           utens,      \
                                           utensstage, \
                                           isize,      \
                                           jsize,      \
                                           ksize,      \
                                           istride,    \
                                           jstride,    \
                                           kstride)
            __attribute__((always_inline)) inline void forward_sweep_kmin(const int i,
                const int j,
                const int ishift,
                const int jshift,
                value_type *__restrict__ ccol,
                value_type *__restrict__ ccol_cache,
                value_type *__restrict__ dcol,
                value_type *__restrict__ dcol_cache,
                const value_type *__restrict__ wcon,
                const value_type *__restrict__ ustage,
                const value_type *__restrict__ upos,
                const value_type *__restrict__ utens,
                const value_type *__restrict__ utensstage,
                const int isize,
                const int jsize,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride, 
				const int ib, 
				const int jb) {

                const int k = 0;
                const int index = i * istride + j * jstride + k * kstride;
                const int cache_index = (i - ib) + (j - jb) * m_iblocksize;
                //const int cache_index = i * istride + j * jstride;
                value_type gcv = value_type(0.25) *
                                 (wcon[index + ishift * istride + jshift * jstride + kstride] + wcon[index + kstride]);
                value_type cs = gcv * bet_m;

                value_type ccoln = gcv * bet_p;
                value_type bcol = dtr_stage - ccoln;

                value_type correction_term = -cs * (ustage[index + kstride] - ustage[index]);
                value_type dcoln = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                value_type divided = value_type(1.0) / bcol;
                ccoln = ccoln * divided;
                dcoln = dcoln * divided;

                ccol_cache[cache_index] = ccoln;
                dcol_cache[cache_index] = dcoln;
                ccol[index] = ccoln;
                dcol[index] = dcoln;
            }

#pragma omp declare simd linear(i) uniform(j,          \
                                           k,          \
                                           ishift,     \
                                           jshift,     \
                                           ccol,       \
                                           ccol_cache, \
                                           dcol,       \
                                           dcol_cache, \
                                           wcon,       \
                                           ustage,     \
                                           upos,       \
                                           utens,      \
                                           utensstage, \
                                           isize,      \
                                           jsize,      \
                                           ksize,      \
                                           istride,    \
                                           jstride,    \
                                           kstride)
            __attribute__((always_inline)) inline void forward_sweep_kbody(const int i,
                const int j,
                const int k,
                const int ishift,
                const int jshift,
                value_type *__restrict__ ccol,
                value_type *__restrict__ ccol_cache,
                value_type *__restrict__ dcol,
                value_type *__restrict__ dcol_cache,
                const value_type *__restrict__ wcon,
                const value_type *__restrict__ ustage,
                const value_type *__restrict__ upos,
                const value_type *__restrict__ utens,
                const value_type *__restrict__ utensstage,
                const int isize,
                const int jsize,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride, 
				const int ib, 
				const int jb) {

                const int index = i * istride + j * jstride + k * kstride;
                const int cache_index = (i - ib) + (j - jb) * m_iblocksize;
                //const int cache_index = i * istride + j * jstride;
                value_type gav = value_type(-0.25) * (wcon[index + ishift * istride + jshift * jstride] + wcon[index]);
                value_type gcv = value_type(0.25) *
                                 (wcon[index + ishift * istride + jshift * jstride + kstride] + wcon[index + kstride]);

                value_type as = gav * bet_m;
                value_type cs = gcv * bet_m;

                value_type acol = gav * bet_p;
                value_type ccoln = gcv * bet_p;
                value_type bcol = dtr_stage - acol - ccoln;

                value_type correction_term =
                    -as * (ustage[index - kstride] - ustage[index]) - cs * (ustage[index + kstride] - ustage[index]);
                value_type dcoln = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                value_type divided = value_type(1.0) / (bcol - ccol_cache[cache_index] * acol);
                ccoln = ccoln * divided;
                dcoln = (dcoln - dcol_cache[cache_index] * acol) * divided;

                ccol_cache[cache_index] = ccoln;
                dcol_cache[cache_index] = dcoln;
                ccol[index] = ccoln;
                dcol[index] = dcoln;
            }

#pragma omp declare simd linear(i) uniform(j,          \
                                           ishift,     \
                                           jshift,     \
                                           ccol,       \
                                           ccol_cache, \
                                           dcol,       \
                                           dcol_cache, \
                                           wcon,       \
                                           ustage,     \
                                           upos,       \
                                           utens,      \
                                           utensstage, \
                                           isize,      \
                                           jsize,      \
                                           ksize,      \
                                           istride,    \
                                           jstride,    \
                                           kstride)
            __attribute__((always_inline)) inline void forward_sweep_kmax(const int i,
                const int j,
                const int ishift,
                const int jshift,
                value_type *__restrict__ ccol,
                value_type *__restrict__ ccol_cache,
                value_type *__restrict__ dcol,
                value_type *__restrict__ dcol_cache,
                const value_type *__restrict__ wcon,
                const value_type *__restrict__ ustage,
                const value_type *__restrict__ upos,
                const value_type *__restrict__ utens,
                const value_type *__restrict__ utensstage,
                const int isize,
                const int jsize,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride, 
				const int ib, 
				const int jb) {

                const int k = ksize - 1;
                const int index = i * istride + j * jstride + k * kstride;
                const int cache_index = (i - ib) + (j - jb) * m_iblocksize;
                //const int cache_index = i * istride + j * jstride;
                value_type gav = value_type(-0.25) * (wcon[index + ishift * istride + jshift * jstride] + wcon[index]);

                value_type as = gav * bet_m;

                value_type acol = gav * bet_p;
                value_type bcol = dtr_stage - acol;

                value_type correction_term = -as * (ustage[index - kstride] - ustage[index]);
                value_type dcoln = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                value_type divided = value_type(1.0) / (bcol - ccol_cache[cache_index] * acol);
                dcoln = (dcoln - dcol_cache[cache_index] * acol) * divided;

                dcol_cache[cache_index] = dcoln;
                dcol[index] = dcoln;
            }
#pragma omp declare simd linear(i) uniform(j,          \
                                           k,          \
                                           ishift,     \
                                           jshift,     \
                                           ccol,       \
                                           ccol_cache, \
                                           dcol,       \
                                           dcol_cache, \
                                           wcon,       \
                                           ustage,     \
                                           upos,       \
                                           utens,      \
                                           utensstage, \
                                           isize,      \
                                           jsize,      \
                                           ksize,      \
                                           istride,    \
                                           jstride,    \
                                           kstride)
            void forward_sweep_k(const int i,
                const int j,
                const int k,
                const int ishift,
                const int jshift,
                value_type *__restrict__ ccol,
                value_type *__restrict__ ccol_cache,
                value_type *__restrict__ dcol,
                value_type *__restrict__ dcol_cache,
                const value_type *__restrict__ wcon,
                const value_type *__restrict__ ustage,
                const value_type *__restrict__ upos,
                const value_type *__restrict__ utens,
                const value_type *__restrict__ utensstage,
                const int isize,
                const int jsize,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride, 
				const int ib, 
				const int jb) {

                if (k == 0) {
                    forward_sweep_kmin(i,
                        j,
                        ishift,
                        jshift,
                        ccol,
                        ccol_cache,
                        dcol,
                        dcol_cache,
                        wcon,
                        ustage,
                        upos,
                        utens,
                        utensstage,
                        isize,
                        jsize,
                        ksize,
                        istride,
                        jstride,
                        kstride, 
						ib, 
						jb);
                } else if (k > 0 && k < ksize - 1){
                    forward_sweep_kbody(i,
                        j,
                        k,
                        ishift,
                        jshift,
                        ccol,
                        ccol_cache,
                        dcol,
                        dcol_cache,
                        wcon,
                        ustage,
                        upos,
                        utens,
                        utensstage,
                        isize,
                        jsize,
                        ksize,
                        istride,
                        jstride,
                        kstride, 
						ib, 
						jb);
                } else {
                    forward_sweep_kmax(i,
                        j,
                        ishift,
                        jshift,
                        ccol,
                        ccol_cache,
                        dcol,
                        dcol_cache,
                        wcon,
                        ustage,
                        upos,
                        utens,
                        utensstage,
                        isize,
                        jsize,
                        ksize,
                        istride,
                        jstride,
                        kstride, 
						ib, 
						jb);
				}
            }

            int m_iblocksize, m_jblocksize;
        };
        extern template class vadv_variant_ik_blocked_j<float>;
        extern template class vadv_variant_ik_blocked_j<double>;

    } // knl

} // namespace platform
