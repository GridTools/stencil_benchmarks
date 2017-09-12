#pragma once

#include "vadv_stencil_variant.h"

namespace platform {

    namespace knl {

        template <class Platform, class ValueType>
        class knl_vadv_stencil_variant : public vadv_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;
            using platform = Platform;

            knl_vadv_stencil_variant(const arguments_map &args) : vadv_stencil_variant<Platform, ValueType>(args) {
                Platform::check_cache_conflicts("i-stride offsets", this->istride() * this->bytes_per_element());
                Platform::check_cache_conflicts("j-stride offsets", this->jstride() * this->bytes_per_element());
                Platform::check_cache_conflicts("k-stride offsets", this->kstride() * this->bytes_per_element());
                Platform::check_cache_conflicts(
                    "2 * i-stride offsets", 2 * this->istride() * this->bytes_per_element());
                Platform::check_cache_conflicts(
                    "2 * j-stride offsets", 2 * this->jstride() * this->bytes_per_element());
                Platform::check_cache_conflicts(
                    "2 * k-stride offsets", 2 * this->kstride() * this->bytes_per_element());
            }
            virtual ~knl_vadv_stencil_variant() {}

            void prerun() override {
                vadv_stencil_variant<Platform, ValueType>::prerun();
                Platform::flush_cache();
            }

          protected:
#pragma omp declare simd linear(i) uniform( \
    j, ccol, dcol, datacol, upos, utensstage, isize, jsize, ksize, istride, jstride, kstride)
            /*__attribute__((always_inline))*/ void backward_sweep(const int i,
                const int j,
                value_type *__restrict__ ccol,
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

                // k maximum
                {
                    const int k = ksize - 1;
                    const int index = i * istride + j * jstride + k * kstride;
                    datacol[index] = dcol[index];
                    ccol[index] = datacol[index];
                    utensstage[index] = dtr_stage * (datacol[index] - upos[index]);
                }

                // k body
                for (int k = ksize - 2; k >= 0; --k) {
                    int index = i * istride + j * jstride + k * kstride;
                    datacol[index] = dcol[index] - ccol[index] * datacol[index + kstride];
                    ccol[index] = datacol[index];
                    utensstage[index] = dtr_stage * (datacol[index] - upos[index]);
                }
            }

#pragma omp declare simd linear(i) uniform( \
    j, ishift, jshift, ccol, wcon, ustage, upos, utens, utensstage, isize, jsize, ksize, istride, jstride, kstride)
            /*__attribute__((always_inline))*/ void forward_sweep(const int i,
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
                constexpr value_type dtr_stage = 3.0 / 20.0;
                constexpr value_type beta_v = 0;
                constexpr value_type bet_m = 0.5 * (1.0 - beta_v);
                constexpr value_type bet_p = 0.5 * (1.0 + beta_v);

                // k minimum
                {
                    const int k = 0;
                    const int index = i * istride + j * jstride + k * kstride;
                    value_type gcv = value_type(0.25) * (wcon[index + ishift * istride + jshift * jstride + kstride] +
                                                            wcon[index + kstride]);
                    value_type cs = gcv * bet_m;

                    ccol[index] = gcv * bet_p;
                    value_type bcol = dtr_stage - ccol[index];

                    value_type correction_term = -cs * (ustage[index + kstride] - ustage[index]);
                    dcol[index] = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    value_type divided = value_type(1.0) / bcol;
                    ccol[index] = ccol[index] * divided;
                    dcol[index] = dcol[index] * divided;
                }

                // k body
                for (int k = 1; k < ksize - 1; ++k) {
                    const int index = i * istride + j * jstride + k * kstride;
                    value_type gav =
                        value_type(-0.25) * (wcon[index + ishift * istride + jshift * jstride] + wcon[index]);
                    value_type gcv = value_type(0.25) * (wcon[index + ishift * istride + jshift * jstride + kstride] +
                                                            wcon[index + kstride]);

                    value_type as = gav * bet_m;
                    value_type cs = gcv * bet_m;

                    value_type acol = gav * bet_p;
                    ccol[index] = gcv * bet_p;
                    value_type bcol = dtr_stage - acol - ccol[index];

                    value_type correction_term = -as * (ustage[index - kstride] - ustage[index]) -
                                                 cs * (ustage[index + kstride] - ustage[index]);
                    dcol[index] = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    value_type divided = value_type(1.0) / (bcol - ccol[index - kstride] * acol);
                    ccol[index] = ccol[index] * divided;
                    dcol[index] = (dcol[index] - dcol[index - kstride] * acol) * divided;
                }

                // k maximum
                {
                    const int k = ksize - 1;
                    const int index = i * istride + j * jstride + k * kstride;
                    value_type gav =
                        value_type(-0.25) * (wcon[index + ishift * istride + jshift * jstride] + wcon[index]);

                    value_type as = gav * bet_m;

                    value_type acol = gav * bet_p;
                    value_type bcol = dtr_stage - acol;

                    value_type correction_term = -as * (ustage[index - kstride] - ustage[index]);
                    dcol[index] = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    value_type divided = value_type(1.0) / (bcol - ccol[index - kstride] * acol);
                    dcol[index] = (dcol[index] - dcol[index - kstride] * acol) * divided;
                }
            }
        };

    } // namespace knl

} // namespace platform
