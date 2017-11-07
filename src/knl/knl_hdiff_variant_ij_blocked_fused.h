#pragma once

#include "knl/knl_hdiff_stencil_variant.h"

namespace platform {

    namespace knl {

        template <class Platform, class ValueType>
        class knl_hdiff_variant_ij_blocked_fused final : public knl_hdiff_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;

            knl_hdiff_variant_ij_blocked_fused(const arguments_map &args)
                : knl_hdiff_stencil_variant<Platform, ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
            }

            void hdiff() override {

                const value_type *__restrict__ in = this->in();
                const value_type *__restrict__ coeff = this->coeff();
                value_type *__restrict__ lap = this->lap();
                value_type *__restrict__ flx = this->flx();
                value_type *__restrict__ fly = this->fly();
                value_type *__restrict__ out = this->out();

                constexpr int istride = 1;
                const int jstride = this->jstride();
                const int kstride = this->kstride();
                const int h = this->halo();
                const int isize = this->isize();
                const int jsize = this->jsize();
                const int ksize = this->ksize();

                if (this->istride() != 1)
                    throw ERROR("this variant is only compatible with unit i-stride layout");
                if (this->halo() < 2)
                    throw ERROR("Minimum required halo is 2");

#pragma omp parallel for collapse(3)
                for (int k = 0; k < ksize; ++k) {
                    for (int jb = 0; jb < jsize; jb += m_jblocksize) {
                        for (int ib = 0; ib < isize; ib += m_iblocksize) {
                            const int imax = ib + m_iblocksize <= isize ? ib + m_iblocksize : isize;
                            const int jmax = jb + m_jblocksize <= jsize ? jb + m_jblocksize : jsize;

                            int index = ib * istride + jb * jstride + k * kstride;
                            for (int j = jb; j < jmax; ++j) {
#pragma omp simd
#pragma vector nontemporal
                                for (int i = ib; i < imax; ++i) {
                                    value_type lap_ij = 4 * in[index] - in[index - istride] - in[index + istride] -
                                                        in[index - jstride] - in[index + jstride];
                                    value_type lap_imj = 4 * in[index - istride] - in[index - 2 * istride] - in[index] -
                                                         in[index - istride - jstride] - in[index - istride + jstride];
                                    value_type lap_ipj = 4 * in[index + istride] - in[index] - in[index + 2 * istride] -
                                                         in[index + istride - jstride] - in[index + istride + jstride];
                                    value_type lap_ijm = 4 * in[index - jstride] - in[index - istride - jstride] -
                                                         in[index + istride - jstride] - in[index - 2 * jstride] -
                                                         in[index];
                                    value_type lap_ijp = 4 * in[index + jstride] - in[index - istride + jstride] -
                                                         in[index + istride + jstride] - in[index] -
                                                         in[index + 2 * jstride];

                                    value_type flx_ij = lap_ipj - lap_ij;
                                    flx_ij = flx_ij * (in[index + istride] - in[index]) > 0 ? 0 : flx_ij;

                                    value_type flx_imj = lap_ij - lap_imj;
                                    flx_imj = flx_imj * (in[index] - in[index - istride]) > 0 ? 0 : flx_imj;

                                    value_type fly_ij = lap_ijp - lap_ij;
                                    fly_ij = fly_ij * (in[index + jstride] - in[index]) > 0 ? 0 : fly_ij;

                                    value_type fly_ijm = lap_ij - lap_ijm;
                                    fly_ijm = fly_ijm * (in[index] - in[index - jstride]) > 0 ? 0 : fly_ijm;

                                    out[index] = in[index] - coeff[index] * (flx_ij - flx_imj + fly_ij - fly_ijm);
                                    index += istride;
                                }
                                index += jstride - (imax - ib) * istride;
                            }
                        }
                    }
                }
            }

          private:
            int m_iblocksize, m_jblocksize;
        };

    } // namespace knl

} // namespace platform
