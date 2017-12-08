#include "knl/knl_hdiff_variant_ij_blocked_private_halo.h"

namespace platform {

    namespace knl {

        template <class Platform, class ValueType>
        void hdiff_variant_ij_blocked_private_halo<Platform, ValueType>::hdiff() {
            const value_type *__restrict__ in = this->in();
            const value_type *__restrict__ coeff = this->coeff();
            value_type *__restrict__ lap = this->lap_tmp();
            value_type *__restrict__ flx = this->flx_tmp();
            value_type *__restrict__ fly = this->fly_tmp();
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
#pragma omp parallel
            {
#pragma omp for collapse(3) schedule(static, 1) nowait
                for (int k = 0; k < ksize; ++k) {
                    for (int jb = 0; jb < m_nbj; ++jb) {
                        for (int ib = 0; ib < m_nbi; ++ib) {
                            const int imax =
                                (ib + 1) * m_iblocksize <= isize ? m_iblocksize : (isize - ib * m_iblocksize);
                            const int jmax =
                                (jb + 1) * m_jblocksize <= jsize ? m_jblocksize : (jsize - jb * m_jblocksize);

                            int index_lap = ib * m_iblocksize * istride + jb * m_jblocksize * jstride + k * kstride -
                                            istride - jstride;
                            int index_flx =
                                ib * m_iblocksize * istride + jb * m_jblocksize * jstride + k * kstride - istride;
                            int index_fly =
                                ib * m_iblocksize * istride + jb * m_jblocksize * jstride + k * kstride - jstride;
                            int index_out = ib * m_iblocksize * istride + jb * m_jblocksize * jstride + k * kstride;

                            int index_lap_tmp = ib * (m_iblocksize + 2 * h) * m_istride_tmp +
                                                jb * (m_jblocksize + 2 * h) * m_jstride_tmp + k * m_kstride_tmp -
                                                m_istride_tmp - m_jstride_tmp;
                            int index_flx_tmp = ib * (m_iblocksize + 2 * h) * m_istride_tmp +
                                                jb * (m_jblocksize + 2 * h) * m_jstride_tmp + k * m_kstride_tmp -
                                                m_istride_tmp;
                            int index_fly_tmp = ib * (m_iblocksize + 2 * h) * m_istride_tmp +
                                                jb * (m_jblocksize + 2 * h) * m_jstride_tmp + k * m_kstride_tmp -
                                                m_jstride_tmp;
                            int index_out_tmp = ib * (m_iblocksize + 2 * h) * m_istride_tmp +
                                                jb * (m_jblocksize + 2 * h) * m_jstride_tmp + k * m_kstride_tmp;

                            for (int j = 0; j < jmax + 2; ++j) {
#pragma omp simd
                                for (int i = 0; i < imax + 2; ++i) {
                                    lap[index_lap_tmp] =
                                        4 * in[index_lap] - (in[index_lap - istride] + in[index_lap + istride] +
                                                                in[index_lap - jstride] + in[index_lap + jstride]);
                                    index_lap += istride;
                                    index_lap_tmp += m_istride_tmp;
                                }
                                index_lap += jstride - (imax + 2) * istride;
                                index_lap_tmp += m_jstride_tmp - (imax + 2) * m_istride_tmp;
                            }

                            for (int j = 0; j < jmax; ++j) {
#pragma omp simd
                                for (int i = 0; i < imax + 1; ++i) {
                                    flx[index_flx_tmp] = lap[index_flx_tmp + m_istride_tmp] - lap[index_flx_tmp];
                                    if (flx[index_flx_tmp] * (in[index_flx + istride] - in[index_flx]) > 0)
                                        flx[index_flx_tmp] = 0.;
                                    index_flx += istride;
                                    index_flx_tmp += m_istride_tmp;
                                }
                                index_flx += jstride - (imax + 1) * istride;
                                index_flx_tmp += m_jstride_tmp - (imax + 1) * m_istride_tmp;
                            }

                            for (int j = 0; j < jmax + 1; ++j) {
#pragma omp simd
                                for (int i = 0; i < imax; ++i) {
                                    fly[index_fly_tmp] = lap[index_fly_tmp + m_jstride_tmp] - lap[index_fly_tmp];
                                    if (fly[index_fly_tmp] * (in[index_fly + jstride] - in[index_fly]) > 0)
                                        fly[index_fly_tmp] = 0.;
                                    index_fly += istride;
                                    index_fly_tmp += m_istride_tmp;
                                }
                                index_fly += jstride - (imax)*istride;
                                index_fly_tmp += m_jstride_tmp - (imax)*m_istride_tmp;
                            }

                            for (int j = 0; j < jmax; ++j) {
#pragma omp simd
                                for (int i = 0; i < imax; ++i) {
                                    out[index_out] =
                                        in[index_out] -
                                        coeff[index_out] * (flx[index_out_tmp] - flx[index_out_tmp - m_istride_tmp] +
                                                               fly[index_out_tmp] - fly[index_out_tmp - m_jstride_tmp]);
                                    index_out += istride;
                                    index_out_tmp += m_istride_tmp;
                                }
                                index_out += jstride - (imax)*istride;
                                index_out_tmp += m_jstride_tmp - (imax)*m_istride_tmp;
                            }
                        }
                    }
                }
            }
        }

        template class hdiff_variant_ij_blocked_private_halo<knl, float>;
        template class hdiff_variant_ij_blocked_private_halo<knl, double>;

    } // namespace knl

} // namespace platform
