#include "knl/knl_hdiff_variant_ij_blocked_dycore.h"

namespace platform {

    namespace knl {

        template <class ValueType>
        void hdiff_variant_ij_blocked_dycore<ValueType>::hdiff() {
            const value_type *__restrict__ in = this->in();
            const value_type *__restrict__ coeff = this->coeff();
            value_type *__restrict__ out = this->out();
            const value_type *__restrict__ crlato = m_crlato.data();
            const value_type *__restrict__ crlatu = m_crlatu.data();

            constexpr int istride = 1;
            const int jstride = this->jstride();
            const int kstride = this->kstride();
            const int h = this->halo();
            const int isize = this->isize();
            const int jsize = this->jsize();
            const int ksize = this->ksize();
            constexpr int icachestride = 1;
            const int jcachestride = m_jcachestride;

            if (this->istride() != 1)
                throw ERROR("this variant is only compatible with unit i-stride layout");
            if (this->halo() < 2)
                throw ERROR("Minimum required halo is 2");

#pragma omp parallel
            {
                value_type *__restrict__ lap_cache = m_caches[omp_get_thread_num()].data();
#pragma omp for collapse(2)
                for (int jb = 0; jb < jsize; jb += m_jblocksize) {
                    for (int ib = 0; ib < isize; ib += m_iblocksize) {
                        for (int k = 0; k < ksize; ++k) {
                            const int imax = ib + m_iblocksize <= isize ? ib + m_iblocksize : isize;
                            const int jmax = jb + m_jblocksize <= jsize ? jb + m_jblocksize : jsize;

                            int index = (ib - 1) * istride + (jb - 1) * jstride + k * kstride;
                            int cache_index = 0;
                            for (int j = jb - 1; j < jmax + 1; ++j) {
#pragma omp simd
                                for (int i = ib - 1; i < imax + 1; ++i) {
                                    lap_cache[cache_index] =
                                        in[index + istride] + in[index - istride] - value_type(2) * in[index] + crlato[j] * (in[index + jstride] - in[index]) + crlatu[j] * (in[index - jstride] - in[index]);
                                    index += istride;
                                    cache_index += icachestride;
                                }
                                index += jstride - (imax - ib + 2) * istride;
                                cache_index += jcachestride - (imax - ib + 2) * icachestride;
                            }

                            index = ib * istride + jb * jstride + k * kstride;
                            cache_index = icachestride + jcachestride;
                            for (int j = jb; j < jmax; ++j) {
#pragma omp simd
                                for (int i = ib; i < imax; ++i) {
                                    const value_type hi_flx =
                                        lap_cache[cache_index + icachestride] - lap_cache[cache_index];
                                    const value_type hi_flx_lim =
                                        hi_flx * (in[index + istride] - in[index]) > value_type(0) ? value_type(0)
                                                                                                   : hi_flx;
                                    const value_type lo_flx =
                                        lap_cache[cache_index] - lap_cache[cache_index - icachestride];
                                    const value_type lo_flx_lim =
                                        lo_flx * (in[index] - in[index - istride]) > value_type(0) ? value_type(0)
                                                                                                   : lo_flx;
                                    const value_type hi_fly =
                                        lap_cache[cache_index + jcachestride] - lap_cache[cache_index];
                                    const value_type hi_fly_lim =
                                        hi_fly * (in[index + jstride] - in[index]) > value_type(0) ? value_type(0)
                                                                                                   : hi_fly;
                                    const value_type lo_fly =
                                        lap_cache[cache_index] - lap_cache[cache_index - jcachestride];
                                    const value_type lo_fly_lim =
                                        lo_fly * (in[index] - in[index - jstride]) > value_type(0) ? value_type(0)
                                                                                                   : lo_fly;

                                    out[index] =
                                        in[index] - coeff[index] * (hi_flx_lim - lo_flx_lim + hi_fly_lim - lo_fly_lim);

                                    index += istride;
                                    cache_index += icachestride;
                                }
                                index += jstride - (imax - ib) * istride;
                                cache_index += jcachestride - (imax - ib) * icachestride;
                            }
                        }
                    }
                }
            }
        }

        template class hdiff_variant_ij_blocked_dycore<float>;
        template class hdiff_variant_ij_blocked_dycore<double>;

    } // namespace knl

} // namespace platform
