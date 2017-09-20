#pragma once

#include "knl/knl_variant_vadv.h"

namespace platform {

    namespace knl {

        template <class Platform, class ValueType>
        class variant_vadv_ij_blocked_k_ring final : public knl_vadv_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;
            using platform = Platform;

            variant_vadv_ij_blocked_k_ring(const arguments_map &args)
                : knl_vadv_stencil_variant<Platform, ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
            }
            ~variant_vadv_ij_blocked_k_ring() {}

            void vadv() override {
                const value_type *__restrict__ ustage = this->ustage();
                const value_type *__restrict__ upos = this->upos();
                const value_type *__restrict__ utens = this->utens();
                value_type *__restrict__ utensstage = this->utensstage();
                const value_type *__restrict__ vstage = this->vstage();
                const value_type *__restrict__ vpos = this->vpos();
                const value_type *__restrict__ vtens = this->vtens();
                value_type *__restrict__ vtensstage = this->vtensstage();
                const value_type *__restrict__ wstage = this->wstage();
                const value_type *__restrict__ wpos = this->wpos();
                const value_type *__restrict__ wtens = this->wtens();
                value_type *__restrict__ wtensstage = this->wtensstage();
                value_type *__restrict__ ccol = this->ccol();
                value_type *__restrict__ dcol = this->dcol();
                const value_type *__restrict__ wcon = this->wcon();
                value_type *__restrict__ datacol = this->datacol();
                const int isize = this->isize();
                const int jsize = this->jsize();
                const int ksize = this->ksize();
                constexpr int istride = 1;
                const int jstride = this->jstride();
                const int kstride = this->kstride();
                if (this->istride() != 1)
                    throw ERROR("this variant is only compatible with unit i-stride layout");

                value_type *__restrict__ ccol_cache = this->datacol();
                value_type *__restrict__ dcol_cache = this->datacol() + kstride;

#pragma omp parallel
                {
#pragma omp for collapse(2) schedule(static, 1) nowait
                    for (int jb = 0; jb < jsize; jb += m_jblocksize) {
                        for (int ib = 0; ib < isize; ib += m_iblocksize) {
                            const int imax = ib + m_iblocksize <= isize ? ib + m_iblocksize : isize;
                            const int jmax = jb + m_jblocksize <= jsize ? jb + m_jblocksize : jsize;

                            for (int k = 0; k < ksize; ++k) {
                                for (int j = jb; j < jmax; ++j) {
#pragma omp simd
#pragma vector nontemporal(ccol, dcol)
                                    for (int i = ib; i < imax; ++i) {
                                        forward_sweep_k(i,
                                            j,
                                            k,
                                            1,
                                            0,
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
                                            kstride);
                                    }
                                }
                            }
                            for (int k = ksize - 1; k >= 0; --k) {
                                for (int j = jb; j < jmax; ++j) {
#pragma omp simd
#pragma vector nontemporal(wtensstage)
                                    for (int i = ib; i < imax; ++i) {
                                        backward_sweep_k(i,
                                            j,
                                            k,
                                            ccol,
                                            dcol,
                                            datacol,
                                            upos,
                                            utensstage,
                                            isize,
                                            jsize,
                                            ksize,
                                            istride,
                                            jstride,
                                            kstride);
                                    }
                                }
                            }

                            for (int k = 0; k < ksize; ++k) {
                                for (int j = jb; j < jmax; ++j) {
#pragma omp simd
#pragma vector nontemporal(ccol, dcol)
                                    for (int i = ib; i < imax; ++i) {
                                        forward_sweep_k(i,
                                            j,
                                            k,
                                            1,
                                            0,
                                            ccol,
                                            ccol_cache,
                                            dcol,
                                            dcol_cache,
                                            wcon,
                                            vstage,
                                            vpos,
                                            vtens,
                                            vtensstage,
                                            isize,
                                            jsize,
                                            ksize,
                                            istride,
                                            jstride,
                                            kstride);
                                    }
                                }
                            }
                            for (int k = ksize - 1; k >= 0; --k) {
                                for (int j = jb; j < jmax; ++j) {
#pragma omp simd
#pragma vector nontemporal(wtensstage)
                                    for (int i = ib; i < imax; ++i) {
                                        backward_sweep_k(i,
                                            j,
                                            k,
                                            ccol,
                                            dcol,
                                            datacol,
                                            vpos,
                                            vtensstage,
                                            isize,
                                            jsize,
                                            ksize,
                                            istride,
                                            jstride,
                                            kstride);
                                    }
                                }
                            }

                            for (int k = 0; k < ksize; ++k) {
                                for (int j = jb; j < jmax; ++j) {
#pragma omp simd
#pragma vector nontemporal(ccol, dcol)
                                    for (int i = ib; i < imax; ++i) {
                                        forward_sweep_k(i,
                                            j,
                                            k,
                                            1,
                                            0,
                                            ccol,
                                            ccol_cache,
                                            dcol,
                                            dcol_cache,
                                            wcon,
                                            wstage,
                                            wpos,
                                            wtens,
                                            wtensstage,
                                            isize,
                                            jsize,
                                            ksize,
                                            istride,
                                            jstride,
                                            kstride);
                                    }
                                }
                            }
                            for (int k = ksize - 1; k >= 0; --k) {
                                for (int j = jb; j < jmax; ++j) {
#pragma omp simd
#pragma vector nontemporal(wtensstage)
                                    for (int i = ib; i < imax; ++i) {
                                        backward_sweep_k(i,
                                            j,
                                            k,
                                            ccol,
                                            dcol,
                                            datacol,
                                            wpos,
                                            wtensstage,
                                            isize,
                                            jsize,
                                            ksize,
                                            istride,
                                            jstride,
                                            kstride);
                                    }
                                }
                            }
                        }
                    }
                }
            }

          private:
            static constexpr value_type dtr_stage = 3.0 / 20.0;
            static constexpr value_type beta_v = 0;
            static constexpr value_type bet_m = 0.5 * (1.0 - beta_v);
            static constexpr value_type bet_p = 0.5 * (1.0 + beta_v);
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
                const int kstride) {

                const int k = 0;
                const int index = i * istride + j * jstride + k * kstride;
                const int cache_index = i * istride + j * jstride;
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
                const int kstride) {

                const int index = i * istride + j * jstride + k * kstride;
                const int cache_index = i * istride + j * jstride;
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
                const int kstride) {

                const int k = ksize - 1;
                const int index = i * istride + j * jstride + k * kstride;
                const int cache_index = i * istride + j * jstride;
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
                const int kstride) {

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
                        kstride);
                } else if (k == ksize - 1) {
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
                        kstride);
                } else {
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
                        kstride);
                }
            }

            int m_iblocksize, m_jblocksize;
        };

    } // knl

} // namespace platform
