#pragma once

#include "knl/knl_hdiff_stencil_variant.h"
#include "knl/knl_platform.h"

namespace platform {

    namespace knl {

        template <class Platform, class ValueType>
        class hdiff_variant_ij_blocked_private_halo final : public knl_hdiff_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;
            using allocator = typename knl_hdiff_stencil_variant<Platform, ValueType>::allocator;

            hdiff_variant_ij_blocked_private_halo(const arguments_map &args)
                : knl_hdiff_stencil_variant<Platform, ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
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
                return m_lap_tmp.data() + this->data_offset() + this->halo() * m_istride_tmp +
                       this->halo() * m_kstride_tmp + this->halo() * m_jstride_tmp;
            }
            value_type *flx_tmp() {
                return m_flx_tmp.data() + this->data_offset() + this->halo() * m_istride_tmp +
                       this->halo() * m_kstride_tmp + this->halo() * m_jstride_tmp;
            }
            value_type *fly_tmp() {
                return m_fly_tmp.data() + this->data_offset() + this->halo() * m_istride_tmp +
                       this->halo() * m_kstride_tmp + this->halo() * m_jstride_tmp;
            }

            void hdiff() override;

          private:
            int m_nbi, m_nbj, m_iblocksize, m_jblocksize;
            int m_jsize_tmp, m_isize_tmp;
            constexpr static int m_istride_tmp = 1;
            int m_jstride_tmp, m_kstride_tmp;
            int m_padding_tmp;
            std::vector<value_type, allocator> m_lap_tmp, m_flx_tmp, m_fly_tmp;
        };

        extern template class hdiff_variant_ij_blocked_private_halo<knl, float>;
        extern template class hdiff_variant_ij_blocked_private_halo<knl, double>;

    } // namespace knl

} // namespace platform
