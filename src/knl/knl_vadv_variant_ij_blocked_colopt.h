#pragma once

#include <omp.h>

#include "knl/knl_platform.h"
#include "knl/knl_vadv_variant.h"

namespace platform {

    namespace knl {

        template <class Platform, class ValueType>
        class vadv_variant_ij_blocked_colopt final : public knl_vadv_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;
            using platform = Platform;

            vadv_variant_ij_blocked_colopt(const arguments_map &args)
                : knl_vadv_stencil_variant<Platform, ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
                std::cout << m_iblocksize << " " << m_jblocksize << std::endl;
            }
            ~vadv_variant_ij_blocked_colopt() {}

            void vadv() override;

          private:
#pragma omp declare simd linear(i) uniform( \
    j, ccol, dcol, datacol, upos, utensstage, isize, jsize, ksize, istride, jstride, kstride)
            /*__attribute__((always_inline))*/ void backward_sweep(const int i,
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
                const int kstride,
                const int thread_index) {
                constexpr value_type dtr_stage = 3.0 / 20.0;
                const int kcolstride = kstride;

                // k maximum
                {
                    const int k = ksize - 1;
                    const int index = i * istride + j * jstride + k * kstride;
                    const int colindex = i * istride + thread_index * jstride + k * kstride;
                    datacol[colindex] = dcol[colindex];
                    utensstage[index] = dtr_stage * (datacol[colindex] - upos[index]);
                }

                // k body
                for (int k = ksize - 2; k >= 0; --k) {
                    const int index = i * istride + j * jstride + k * kstride;
                    const int colindex = i * istride + thread_index * jstride + k * kstride;
                    datacol[colindex] = dcol[colindex] - ccol[colindex] * datacol[colindex + kcolstride];
                    utensstage[index] = dtr_stage * (datacol[colindex] - upos[index]);
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
                const int kstride,
                const int thread_index) {
                constexpr value_type dtr_stage = 3.0 / 20.0;
                constexpr value_type beta_v = 0;
                constexpr value_type bet_m = 0.5 * (1.0 - beta_v);
                constexpr value_type bet_p = 0.5 * (1.0 + beta_v);
                const int kcolstride = kstride;

                // k minimum
                {
                    const int k = 0;
                    const int index = i * istride + j * jstride + k * kstride;
                    const int colindex = i * istride + thread_index * jstride + k * kstride;
                    value_type gcv = value_type(0.25) * (wcon[index + ishift * istride + jshift * jstride + kstride] +
                                                            wcon[index + kstride]);
                    value_type cs = gcv * bet_m;

                    ccol[colindex] = gcv * bet_p;
                    value_type bcol = dtr_stage - ccol[colindex];

                    value_type correction_term = -cs * (ustage[index + kstride] - ustage[index]);
                    dcol[colindex] = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    value_type divided = value_type(1.0) / bcol;
                    ccol[colindex] = ccol[colindex] * divided;
                    dcol[colindex] = dcol[colindex] * divided;
                }

                // k body
                for (int k = 1; k < ksize - 1; ++k) {
                    const int index = i * istride + j * jstride + k * kstride;
                    const int colindex = i * istride + thread_index * jstride + k * kstride;
                    value_type gav =
                        value_type(-0.25) * (wcon[index + ishift * istride + jshift * jstride] + wcon[index]);
                    value_type gcv = value_type(0.25) * (wcon[index + ishift * istride + jshift * jstride + kstride] +
                                                            wcon[index + kstride]);

                    value_type as = gav * bet_m;
                    value_type cs = gcv * bet_m;

                    value_type acol = gav * bet_p;
                    ccol[colindex] = gcv * bet_p;
                    value_type bcol = dtr_stage - acol - ccol[colindex];

                    value_type correction_term = -as * (ustage[index - kstride] - ustage[index]) -
                                                 cs * (ustage[index + kstride] - ustage[index]);
                    dcol[colindex] = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    value_type divided = value_type(1.0) / (bcol - ccol[colindex - kcolstride] * acol);
                    ccol[colindex] = ccol[colindex] * divided;
                    dcol[colindex] = (dcol[colindex] - dcol[colindex - kcolstride] * acol) * divided;
                }

                // k maximum
                {
                    const int k = ksize - 1;
                    const int index = i * istride + j * jstride + k * kstride;
                    const int colindex = i * istride + thread_index * jstride + k * kstride;
                    value_type gav =
                        value_type(-0.25) * (wcon[index + ishift * istride + jshift * jstride] + wcon[index]);

                    value_type as = gav * bet_m;

                    value_type acol = gav * bet_p;
                    value_type bcol = dtr_stage - acol;

                    value_type correction_term = -as * (ustage[index - kstride] - ustage[index]);
                    dcol[colindex] = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    value_type divided = value_type(1.0) / (bcol - ccol[colindex - kcolstride] * acol);
                    dcol[colindex] = (dcol[colindex] - dcol[colindex - kcolstride] * acol) * divided;
                }
            }

            int m_iblocksize, m_jblocksize;
        };

        extern template class vadv_variant_ij_blocked_colopt<knl, float>;
        extern template class vadv_variant_ij_blocked_colopt<knl, double>;

    } // knl

} // namespace platform
