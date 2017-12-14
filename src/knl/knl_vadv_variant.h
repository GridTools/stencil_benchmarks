#pragma once

#include "vadv_stencil_variant.h"

namespace platform {

    namespace knl {

        template <class ValueType>
        class knl_vadv_stencil_variant : public vadv_stencil_variant<knl, ValueType> {
          public:
            using value_type = ValueType;
            using platform = knl;

            knl_vadv_stencil_variant(const arguments_map &args) : vadv_stencil_variant<knl, ValueType>(args) {
                knl::check_cache_conflicts("i-stride offsets", this->istride() * this->bytes_per_element());
                knl::check_cache_conflicts("j-stride offsets", this->jstride() * this->bytes_per_element());
                knl::check_cache_conflicts("k-stride offsets", this->kstride() * this->bytes_per_element());
                knl::check_cache_conflicts("2 * i-stride offsets", 2 * this->istride() * this->bytes_per_element());
                knl::check_cache_conflicts("2 * j-stride offsets", 2 * this->jstride() * this->bytes_per_element());
                knl::check_cache_conflicts("2 * k-stride offsets", 2 * this->kstride() * this->bytes_per_element());
            }
            virtual ~knl_vadv_stencil_variant() {}

            void prerun() override {
                vadv_stencil_variant<knl, ValueType>::prerun();
                knl::flush_cache();
            }

          protected:
            static constexpr value_type dtr_stage = 3.0 / 20.0;
            static constexpr value_type beta_v = 0;
            static constexpr value_type bet_m = 0.5 * (1.0 - beta_v);
            static constexpr value_type bet_p = 0.5 * (1.0 + beta_v);

#pragma omp declare simd linear(i) uniform( \
    j, ccol, dcol, datacol, upos, utensstage, isize, jsize, ksize, istride, jstride, kstride)
            __attribute__((always_inline)) void backward_sweep_kmax(const int i,
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
            __attribute__((always_inline)) void backward_sweep_kbody(const int i,
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
    j, ccol, dcol, upos, utensstage, isize, jsize, ksize, istride, jstride, kstride)
            __attribute__((always_inline)) inline void backward_sweep(const int i,
                const int j,
                const value_type *__restrict__ ccol,
                const value_type *__restrict__ dcol,
                const value_type *__restrict__ upos,
                value_type *__restrict__ utensstage,
                const int isize,
                const int jsize,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride) {
                constexpr value_type dtr_stage = 3.0 / 20.0;

                value_type datacol;

                int index = i * istride + j * jstride + (ksize - 1) * kstride;
                // k
                {
                    datacol = dcol[index];
                    utensstage[index] = dtr_stage * (datacol - upos[index]);

                    index -= kstride;
                }

                // k body
                for (int k = ksize - 2; k >= 0; --k) {
                    /*constexpr int prefdist = 6;
                    if (k >= prefdist) {
                        const int prefindex = index - prefdist * kstride;
                        _mm_prefetch(reinterpret_cast<const char *>(&dcol[prefindex]), _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<const char *>(&ccol[prefindex]), _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<const char *>(&upos[prefindex]), _MM_HINT_NTA);
                        _mm_prefetch(reinterpret_cast<const char *>(&utensstage[prefindex]), _MM_HINT_NTA);
                    }*/
                    datacol = dcol[index] - ccol[index] * datacol;
                    utensstage[index] = dtr_stage * (datacol - upos[index]);

                    index -= kstride;
                }
            }

#pragma omp declare simd linear(i) uniform( \
    j, k, ccol, dcol, datacol, upos, utensstage, isize, jsize, ksize, istride, jstride, kstride)
            __attribute__((always_inline)) void backward_sweep_k(const int i,
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

#pragma omp declare simd linear(i) uniform( \
    j, ishift, jshift, ccol, wcon, ustage, upos, utens, utensstage, isize, jsize, ksize, istride, jstride, kstride)
            __attribute__((always_inline)) void forward_sweep_kmin(const int i,
                const int j,
                const int ishift,
                const int jshift,
                value_type *__restrict__ ccol,
                value_type *__restrict__ dcol,
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
                value_type gcv = value_type(0.25) *
                                 (wcon[index + ishift * istride + jshift * jstride + kstride] + wcon[index + kstride]);
                value_type cs = gcv * bet_m;

                ccol[index] = gcv * bet_p;
                value_type bcol = dtr_stage - ccol[index];

                value_type correction_term = -cs * (ustage[index + kstride] - ustage[index]);
                dcol[index] = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                value_type divided = value_type(1.0) / bcol;
                ccol[index] = ccol[index] * divided;
                dcol[index] = dcol[index] * divided;
            }

#pragma omp declare simd linear(i) uniform( \
    j, k, ishift, jshift, ccol, wcon, ustage, upos, utens, utensstage, isize, jsize, ksize, istride, jstride, kstride)
            __attribute__((always_inline)) void forward_sweep_kbody(const int i,
                const int j,
                const int k,
                const int ishift,
                const int jshift,
                value_type *__restrict__ ccol,
                value_type *__restrict__ dcol,
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
                value_type gav = value_type(-0.25) * (wcon[index + ishift * istride + jshift * jstride] + wcon[index]);
                value_type gcv = value_type(0.25) *
                                 (wcon[index + ishift * istride + jshift * jstride + kstride] + wcon[index + kstride]);

                value_type as = gav * bet_m;
                value_type cs = gcv * bet_m;

                value_type acol = gav * bet_p;
                ccol[index] = gcv * bet_p;
                value_type bcol = dtr_stage - acol - ccol[index];

                value_type correction_term =
                    -as * (ustage[index - kstride] - ustage[index]) - cs * (ustage[index + kstride] - ustage[index]);
                dcol[index] = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                value_type divided = value_type(1.0) / (bcol - ccol[index - kstride] * acol);
                ccol[index] = ccol[index] * divided;
                dcol[index] = (dcol[index] - dcol[index - kstride] * acol) * divided;
            }

#pragma omp declare simd linear(i) uniform( \
    j, ishift, jshift, ccol, wcon, ustage, upos, utens, utensstage, isize, jsize, ksize, istride, jstride, kstride)
            __attribute__((always_inline)) void forward_sweep_kmax(const int i,
                const int j,
                const int ishift,
                const int jshift,
                value_type *__restrict__ ccol,
                value_type *__restrict__ dcol,
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
                value_type gav = value_type(-0.25) * (wcon[index + ishift * istride + jshift * jstride] + wcon[index]);

                value_type as = gav * bet_m;

                value_type acol = gav * bet_p;
                value_type bcol = dtr_stage - acol;

                value_type correction_term = -as * (ustage[index - kstride] - ustage[index]);
                dcol[index] = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                value_type divided = value_type(1.0) / (bcol - ccol[index - kstride] * acol);
                dcol[index] = (dcol[index] - dcol[index - kstride] * acol) * divided;
            }
#pragma omp declare simd linear(i) uniform( \
    j, k, ishift, jshift, ccol, wcon, ustage, upos, utens, utensstage, isize, jsize, ksize, istride, jstride, kstride)
            __attribute__((always_inline)) void forward_sweep_k(const int i,
                const int j,
                const int k,
                const int ishift,
                const int jshift,
                value_type *__restrict__ ccol,
                value_type *__restrict__ dcol,
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
                        dcol,
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
                        dcol,
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
                        dcol,
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

#pragma omp declare simd linear(i) uniform( \
    j, ishift, jshift, ccol, wcon, ustage, upos, utens, utensstage, isize, jsize, ksize, istride, jstride, kstride)
            __attribute__((always_inline)) void forward_sweep(const int i,
                const int j,
                const int ishift,
                const int jshift,
                value_type *__restrict__ ccol,
                value_type *__restrict__ dcol,
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

                value_type ccol0, ccol1;
                value_type dcol0, dcol1;
                ValueType ustage0, ustage1, ustage2;
                ValueType wcon0, wcon1;
                ValueType wcon_shift0, wcon_shift1;

                int index = i * istride + j * jstride;
                // k minimum
                {
                    wcon_shift0 = wcon[index + ishift * istride + jshift * jstride + kstride];
                    wcon0 = wcon[index + kstride];
                    ValueType gcv = ValueType(0.25) * (wcon_shift0 + wcon0);
                    ValueType cs = gcv * bet_m;

                    ccol0 = gcv * bet_p;
                    ValueType bcol = dtr_stage - ccol0;

                    ustage0 = ustage[index + kstride];
                    ustage1 = ustage[index];
                    ValueType correction_term = -cs * (ustage0 - ustage1);
                    dcol0 = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    ValueType divided = ValueType(1.0) / bcol;
                    ccol0 = ccol0 * divided;
                    dcol0 = dcol0 * divided;

                    ccol[index] = ccol0;
                    dcol[index] = dcol0;

                    index += kstride;
                }

                // k body
                for (int k = 1; k < ksize - 1; ++k) {
                    constexpr int prefdist = 3;
                    if (k < ksize - prefdist) {
                        const int prefindex = index + prefdist * kstride;
                        _mm_prefetch(reinterpret_cast<const char *>(&upos[prefindex]), _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<const char *>(&ustage[prefindex + kstride]), _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<const char *>(&utens[prefindex]), _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<const char *>(&utensstage[prefindex]), _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<const char *>(&wcon[prefindex + kstride]), _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<const char *>(
                                         &wcon[prefindex + ishift * istride + jshift * jstride + kstride]),
                            _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<const char *>(&ccol[prefindex]), _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<const char *>(&dcol[prefindex]), _MM_HINT_T1);
                    }

                    ccol1 = ccol0;
                    dcol1 = dcol0;
                    ustage2 = ustage1;
                    ustage1 = ustage0;
                    wcon1 = wcon0;
                    wcon_shift1 = wcon_shift0;

                    ValueType gav = ValueType(-0.25) * (wcon_shift1 + wcon1);
                    wcon_shift0 = wcon[index + ishift * istride + jshift * jstride + kstride];
                    wcon0 = wcon[index + kstride];
                    ValueType gcv = ValueType(0.25) * (wcon_shift0 + wcon0);

                    ValueType as = gav * bet_m;
                    ValueType cs = gcv * bet_m;

                    ValueType acol = gav * bet_p;
                    ccol0 = gcv * bet_p;
                    ValueType bcol = dtr_stage - acol - ccol0;

                    ustage0 = ustage[index + kstride];
                    ValueType correction_term = -as * (ustage2 - ustage1) - cs * (ustage0 - ustage1);
                    dcol0 = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    ValueType divided = ValueType(1.0) / (bcol - ccol1 * acol);
                    ccol0 = ccol0 * divided;
                    dcol0 = (dcol0 - dcol1 * acol) * divided;

                    ccol[index] = ccol0;
                    dcol[index] = dcol0;

                    index += kstride;
                }

                // k maximum
                {
                    ccol1 = ccol0;
                    dcol1 = dcol0;
                    ustage2 = ustage1;
                    ustage1 = ustage0;
                    wcon1 = wcon0;
                    wcon_shift1 = wcon_shift0;

                    ValueType gav = ValueType(-0.25) * (wcon_shift1 + wcon1);

                    ValueType as = gav * bet_m;

                    ValueType acol = gav * bet_p;
                    ValueType bcol = dtr_stage - acol;

                    ValueType correction_term = -as * (ustage2 - ustage1);
                    dcol0 = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    ValueType divided = ValueType(1.0) / (bcol - ccol1 * acol);
                    dcol0 = (dcol0 - dcol1 * acol) * divided;

                    ccol[index] = ccol0;
                    dcol[index] = dcol0;
                }
            }
        };

    } // namespace knl

} // namespace platform
