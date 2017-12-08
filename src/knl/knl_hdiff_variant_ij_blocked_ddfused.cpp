#include "knl/knl_hdiff_variant_ij_blocked_ddfused.h"

namespace platform {

    namespace knl {

        template <class ValueType>
        void hdiff_variant_ij_blocked_ddfused<ValueType>::hdiff() {
            const value_type *__restrict__ in = this->in();
            const value_type *__restrict__ coeff = this->coeff();
            value_type *__restrict__ lap = this->lap();
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

#pragma omp parallel for collapse(3) schedule(static, 1)
            for (int kb = 0; kb < m_kblocks; ++kb) {
                for (int jb = 0; jb < m_jblocks; ++jb) {
                    for (int ib = 0; ib < m_iblocks; ++ib) {
                        const int imin = ib * isize / m_iblocks;
                        const int imax = (ib + 1) * isize / m_iblocks;
                        const int jmin = jb * jsize / m_jblocks;
                        const int jmax = (jb + 1) * jsize / m_jblocks;
                        const int kmin = kb * ksize / m_kblocks;
                        const int kmax = (kb + 1) * ksize / m_kblocks;

                        int index = imin * istride + jmin * jstride + kmin * kstride;
                        for (int k = kmin; k < kmax; ++k) {
                            for (int j = jmin; j < jmax; ++j) {
#pragma omp simd
#pragma vector nontemporal
                                for (int i = imin; i < imax; ++i) {
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
                                index += jstride - (imax - imin) * istride;
                            }
                            index += kstride - (jmax - jmin) * jstride;
                        }
                    }
                }
            }
        }

        template class hdiff_variant_ij_blocked_ddfused<float>;
        template class hdiff_variant_ij_blocked_ddfused<double>;

    } // namespace knl

} // namespace platform
