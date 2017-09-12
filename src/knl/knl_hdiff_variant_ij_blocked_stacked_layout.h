#pragma once

#include "knl/knl_hdiff_stencil_variant.h"

namespace platform {

    namespace knl {

        template <class Platform, class ValueType>
        class knl_hdiff_variant_ij_blocked_stacked_layout final : public knl_hdiff_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;
            using allocator = typename knl_hdiff_stencil_variant<Platform, ValueType>::allocator;

            knl_hdiff_variant_ij_blocked_stacked_layout(const arguments_map &args)
                : knl_hdiff_stencil_variant<Platform, ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
                // get number of blocks in I and J
                m_nbi = std::ceil((double)this->isize()/(double)m_iblocksize);
                m_nbj = std::ceil((double)this->jsize()/(double)m_jblocksize);
                // compute size of complete domain including halo
                m_isize_tmp = (this->halo() * 2 + m_iblocksize);
                m_jsize_tmp = (this->halo() * 2 + m_jblocksize);
                m_ksize_tmp = (this->ksize() + 2 * this->halo()) * m_nbi * m_nbj;
                // compute padding in order to make the temporary fields aligned
                m_padding_tmp = std::ceil((double)m_isize_tmp/(double)this->alignment()) * this->alignment() - m_isize_tmp;
                // strides
                m_jstride_tmp = m_isize_tmp + m_padding_tmp;
                m_kstride_tmp = m_jstride_tmp * m_jsize_tmp;
                // init tmps
                m_lap_tmp.resize(this->data_offset() + m_jstride_tmp * m_jsize_tmp * m_ksize_tmp);
                m_flx_tmp.resize(this->data_offset() + m_jstride_tmp * m_jsize_tmp * m_ksize_tmp);
                m_fly_tmp.resize(this->data_offset() + m_jstride_tmp * m_jsize_tmp * m_ksize_tmp);
                m_in_tmp.resize(this->data_offset() + m_jstride_tmp * m_jsize_tmp * m_ksize_tmp);
                m_coeff_tmp.resize(this->data_offset() + m_jstride_tmp * m_jsize_tmp * m_ksize_tmp);
            }

            value_type *lap_tmp() { return m_lap_tmp.data() + this->data_offset() + this->halo()*m_istride_tmp + this->halo()*m_kstride_tmp + this->halo()*m_jstride_tmp; }
            value_type *flx_tmp() { return m_flx_tmp.data() + this->data_offset() + this->halo()*m_istride_tmp + this->halo()*m_kstride_tmp + this->halo()*m_jstride_tmp; }
            value_type *fly_tmp() { return m_fly_tmp.data() + this->data_offset() + this->halo()*m_istride_tmp + this->halo()*m_kstride_tmp + this->halo()*m_jstride_tmp; }
            value_type *in_tmp() { return m_in_tmp.data() + this->data_offset() + this->halo()*m_istride_tmp + this->halo()*m_kstride_tmp + this->halo()*m_jstride_tmp; }
            value_type *coeff_tmp() { return m_coeff_tmp.data() + this->data_offset() + this->halo()*m_istride_tmp + this->halo()*m_kstride_tmp + this->halo()*m_jstride_tmp; }

            void prerun() override {
                knl_hdiff_stencil_variant<Platform, ValueType>::prerun();
                // copy data from in and coeff into block storage
                const int istride = 1;
                const int jstride = this->jstride();
                const int kstride = this->kstride();
                const int h = this->halo();
                const int isize = this->isize();
                const int jsize = this->jsize();
                const int ksize = this->ksize();

                #pragma omp parallel for collapse(2) schedule(static, 1)
                for (int jb = 0; jb < m_nbj; ++jb) {
                    for (int ib = 0; ib < m_nbi; ++ib) {
                        const int bn = (jb*m_nbi + ib);
                        const int imax = (ib+1)*m_iblocksize <= isize ? m_iblocksize : (isize - ib*m_iblocksize);
                        const int jmax = (jb+1)*m_jblocksize <= jsize ? m_jblocksize : (jsize - jb*m_jblocksize);
                        // iterate over k level for given block
                        for (int k = -h; k < this->ksize()+h; ++k) {
                            // iterate over i and j for given block
                            for (int j = -h; j < jmax+h; ++j) {
                                for (int i = -h; i < imax+h; ++i) {
                                    const int index_tmp = (bn*this->ksize() + 2*bn*h)*m_kstride_tmp + k*m_kstride_tmp + i*m_istride_tmp + j*m_jstride_tmp;
                                    const int index_real = ib*m_iblocksize*istride + jb*m_jblocksize*jstride + k*kstride + i*istride + j*jstride; 
                                    in_tmp()[index_tmp] = this->in()[index_real];
                                    coeff_tmp()[index_tmp] = this->coeff()[index_real];                            
                                }
                            }
                        }
                    }
                }
            }

            void hdiff() override {

                const value_type *__restrict__ in = this->in_tmp();
                const value_type *__restrict__ coeff = this->coeff_tmp();
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
                for (int k = 0; k < ksize; ++k) {
                #pragma omp for collapse(2) schedule(static,1) nowait
                    for (int jb = 0; jb < m_nbj; ++jb) {
                        for (int ib = 0; ib < m_nbi; ++ib) {
                            const int imax = (ib+1)*m_iblocksize <= isize ? m_iblocksize : (isize - ib*m_iblocksize);
                            const int jmax = (jb+1)*m_jblocksize <= jsize ? m_jblocksize : (jsize - jb*m_jblocksize);

                            int index_out = ib*m_iblocksize*istride + jb*m_jblocksize*jstride + k*kstride;
                            const int bn = (jb*m_nbi + ib);
                            
                            int index_lap_tmp = (bn*this->ksize() + 2*bn*h)*m_kstride_tmp + k*m_kstride_tmp - m_istride_tmp - m_jstride_tmp;
                            int index_flx_tmp = (bn*this->ksize() + 2*bn*h)*m_kstride_tmp + k*m_kstride_tmp - m_istride_tmp;
                            int index_fly_tmp = (bn*this->ksize() + 2*bn*h)*m_kstride_tmp + k*m_kstride_tmp - m_jstride_tmp;
                            int index_out_tmp = (bn*this->ksize() + 2*bn*h)*m_kstride_tmp + k*m_kstride_tmp;

                            for (int j = 0; j < jmax+2; ++j) {
                                #pragma omp simd
                                for (int i = 0; i < imax+2; ++i) {
                                    lap[index_lap_tmp] = 4 * in[index_lap_tmp] -
                                        (in[index_lap_tmp - m_istride_tmp] + in[index_lap_tmp + m_istride_tmp] + 
                                         in[index_lap_tmp - m_jstride_tmp] + in[index_lap_tmp + m_jstride_tmp]);
                                    index_lap_tmp += m_istride_tmp;
                                }
                                index_lap_tmp += m_jstride_tmp - (imax+2) * m_istride_tmp;
                            }
                            
                            for (int j = 0; j < jmax; ++j) {
                                #pragma omp simd
                                for (int i = 0; i < imax+1; ++i) {
                                    flx[index_flx_tmp] = lap[index_flx_tmp + m_istride_tmp] - lap[index_flx_tmp];
                                    if (flx[index_flx_tmp] * (in[index_flx_tmp + m_istride_tmp] - in[index_flx_tmp]) > 0)
                                        flx[index_flx_tmp] = 0.;
                                    index_flx_tmp += m_istride_tmp;
                                }
                                index_flx_tmp += m_jstride_tmp - (imax+1) * m_istride_tmp;
                            }

                            for (int j = 0; j < jmax+1; ++j) {
                                #pragma omp simd
                                for (int i = 0; i < imax; ++i) {
                                    fly[index_fly_tmp] = lap[index_fly_tmp + m_jstride_tmp] - lap[index_fly_tmp];
                                    if (fly[index_fly_tmp] * (in[index_fly_tmp + m_jstride_tmp] - in[index_fly_tmp]) > 0)
                                        fly[index_fly_tmp] = 0.;
                                    index_fly_tmp += m_istride_tmp;
                                }
                                index_fly_tmp += m_jstride_tmp - (imax) * m_istride_tmp;
                            }
            
                            for (int j = 0; j < jmax; ++j) {
                                #pragma omp simd
                                #pragma vector nontemporal
                                for (int i = 0; i < imax; ++i) {
                                    out[index_out] =
                                        in[index_out_tmp] - coeff[index_out_tmp] *
                                            (flx[index_out_tmp] - flx[index_out_tmp - m_istride_tmp] + 
                                             fly[index_out_tmp] - fly[index_out_tmp - m_jstride_tmp]);
                                    index_out += istride;
                                    index_out_tmp += m_istride_tmp;
                                }
                                index_out += jstride - (imax) * istride;
                                index_out_tmp += m_jstride_tmp - (imax) * m_istride_tmp;                                
                            }
                        }
                    }
                }
}
            }

          private:
            int m_nbi, m_nbj, m_iblocksize, m_jblocksize;
            int m_jsize_tmp, m_isize_tmp, m_ksize_tmp;
            constexpr static int m_istride_tmp = 1;
            int m_jstride_tmp, m_kstride_tmp;
            int m_padding_tmp;
            std::vector<value_type, allocator> m_lap_tmp, m_flx_tmp, m_fly_tmp, m_in_tmp, m_coeff_tmp;        
        };

    } // namespace knl

} // namespace platform
