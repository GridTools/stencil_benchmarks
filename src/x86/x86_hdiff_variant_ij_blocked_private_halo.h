#pragma once

#include "x86/x86_hdiff_stencil_variant.h"

namespace platform {

    namespace x86 {

        template <class Platform, class ValueType>
        class x86_hdiff_variant_ij_blocked_private_halo final : public x86_hdiff_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;
            using allocator = typename x86_hdiff_stencil_variant<Platform, ValueType>::allocator;

            x86_hdiff_variant_ij_blocked_private_halo(const arguments_map &args)
                : x86_hdiff_stencil_variant<Platform, ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
                // get number of blocks in I and J
                m_nbi = std::ceil((double)this->isize() / (double)m_iblocksize);
                m_nbj = std::ceil((double)this->jsize() / (double)m_jblocksize);
                // compute size of complete domain including halo
                m_isize_tmp = m_nbi * (this->halo() * 2 + m_iblocksize);
                m_jsize_tmp = m_nbj * (this->halo() * 2 + m_jblocksize);
                // compute padding in order to make the temporary fields aligned
                m_padding_tmp =
                    std::ceil((double)m_isize_tmp / (double)this->alignment()) * this->alignment() - m_isize_tmp;
                // strides
                m_jstride_tmp = m_isize_tmp + m_padding_tmp;
                m_kstride_tmp = m_jstride_tmp * m_jsize_tmp;
                // init tmps
                m_lap_tmp.resize(
                    this->data_offset() + m_jstride_tmp * m_jsize_tmp * (this->ksize() + 2 * this->halo()));
                m_flx_tmp.resize(
                    this->data_offset() + m_jstride_tmp * m_jsize_tmp * (this->ksize() + 2 * this->halo()));
                m_fly_tmp.resize(
                    this->data_offset() + m_jstride_tmp * m_jsize_tmp * (this->ksize() + 2 * this->halo()));
            }

            value_type *lap_tmp() {
                constexpr int m_istride_tmp = 1;
                return m_lap_tmp.data() + this->data_offset() + this->halo() * m_istride_tmp +
                       this->halo() * m_kstride_tmp + this->halo() * m_jstride_tmp;
            }
            value_type *flx_tmp() {
                constexpr int m_istride_tmp = 1;
                return m_flx_tmp.data() + this->data_offset() + this->halo() * m_istride_tmp +
                       this->halo() * m_kstride_tmp + this->halo() * m_jstride_tmp;
            }
            value_type *fly_tmp() {
                constexpr int m_istride_tmp = 1;
                return m_fly_tmp.data() + this->data_offset() + this->halo() * m_istride_tmp +
                       this->halo() * m_kstride_tmp + this->halo() * m_jstride_tmp;
            }

            void hdiff() override {

                const value_type *__restrict__ in = this->in();
                const value_type *__restrict__ coeff = this->coeff();
                value_type *__restrict__ lap = this->lap_tmp();
                value_type *__restrict__ flx = this->flx_tmp();
                value_type *__restrict__ fly = this->fly_tmp();
                value_type *__restrict__ out = this->out();

                constexpr int istride = 1;
                constexpr int m_istride_tmp = 1;
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

          private:
            int m_nbi, m_nbj, m_iblocksize, m_jblocksize;
            int m_jsize_tmp, m_isize_tmp;
            int m_jstride_tmp, m_kstride_tmp;
            int m_padding_tmp;
            std::vector<value_type, allocator> m_lap_tmp, m_flx_tmp, m_fly_tmp;
        };

    } // namespace x86

} // namespace platform
