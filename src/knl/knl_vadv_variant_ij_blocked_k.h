#pragma once

#include "knl/knl_variant_vadv.h"

namespace platform {

    namespace knl {

        template <class Platform, class ValueType>
        class variant_vadv_ij_blocked_k final : public knl_vadv_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;
            using platform = Platform;

            variant_vadv_ij_blocked_k(const arguments_map &args)
                : knl_vadv_stencil_variant<Platform, ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
            }
            ~variant_vadv_ij_blocked_k() {}

            void vadv() override {
                const value_type *__restrict__ ustage = this->ustage();
                const value_type *__restrict__ upos = this->upos();
                const value_type *__restrict__ utens = this->utens();
                value_type *__restrict__ utensstage = this->utensstage();
                const value_type *__restrict__ vstage = this->vstage();
                const value_type *__restrict__ vpos = this->vpos();
                const value_type *__restrict__ vtens = this->vtens();
                value_type *__restrict__ vtensstage = this->vtensstage();
                const value_type *__restrict__ wstage = this->wstage();
                const value_type *__restrict__ wpos = this->wpos();
                const value_type *__restrict__ wtens = this->wtens();
                value_type *__restrict__ wtensstage = this->wtensstage();
                value_type *__restrict__ ccol = this->ccol();
                value_type *__restrict__ dcol = this->dcol();
                const value_type *__restrict__ wcon = this->wcon();
                value_type *__restrict__ datacol = this->datacol();
                const int isize = this->isize();
                const int jsize = this->jsize();
                const int ksize = this->ksize();
                constexpr int istride = 1;
                const int jstride = this->jstride();
                const int kstride = this->kstride();
                if (this->istride() != 1)
                    throw ERROR("this variant is only compatible with unit i-stride layout");

#pragma omp parallel
                {
#pragma omp for collapse(2) nowait
                    for (int jb = 0; jb < jsize; jb += m_jblocksize) {
                        for (int ib = 0; ib < isize; ib += m_iblocksize) {
                            const int imax = ib + m_iblocksize <= isize ? ib + m_iblocksize : isize;
                            const int jmax = jb + m_jblocksize <= jsize ? jb + m_jblocksize : jsize;

                            for (int k = 0; k < ksize; ++k) {
                                for (int j = jb; j < jmax; ++j) {
#pragma omp simd
                                    for (int i = ib; i < imax; ++i) {
                                        forward_sweep_k(i,
                                            j,
                                            k,
                                            1,
                                            0,
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
                            }
                            for (int k = ksize - 1; k >= 0; --k) {
                                for (int j = jb; j < jmax; ++j) {
#pragma omp simd
                                    for (int i = ib; i < imax; ++i) {
                                        backward_sweep_k(i,
                                            j,
                                            k,
                                            ccol,
                                            dcol,
                                            datacol,
                                            upos,
                                            utensstage,
                                            isize,
                                            jsize,
                                            ksize,
                                            istride,
                                            jstride,
                                            kstride);
                                    }
                                }
                            }

                            for (int k = 0; k < ksize; ++k) {
                                for (int j = jb; j < jmax; ++j) {
#pragma omp simd
                                    for (int i = ib; i < imax; ++i) {
                                        forward_sweep_k(i,
                                            j,
                                            k,
                                            1,
                                            0,
                                            ccol,
                                            dcol,
                                            wcon,
                                            vstage,
                                            vpos,
                                            vtens,
                                            vtensstage,
                                            isize,
                                            jsize,
                                            ksize,
                                            istride,
                                            jstride,
                                            kstride);
                                    }
                                }
                            }
                            for (int k = ksize - 1; k >= 0; --k) {
                                for (int j = jb; j < jmax; ++j) {
#pragma omp simd
                                    for (int i = ib; i < imax; ++i) {
                                        backward_sweep_k(i,
                                            j,
                                            k,
                                            ccol,
                                            dcol,
                                            datacol,
                                            vpos,
                                            vtensstage,
                                            isize,
                                            jsize,
                                            ksize,
                                            istride,
                                            jstride,
                                            kstride);
                                    }
                                }
                            }

                            for (int k = 0; k < ksize; ++k) {
                                for (int j = jb; j < jmax; ++j) {
#pragma omp simd
                                    for (int i = ib; i < imax; ++i) {
                                        forward_sweep_k(i,
                                            j,
                                            k,
                                            1,
                                            0,
                                            ccol,
                                            dcol,
                                            wcon,
                                            wstage,
                                            wpos,
                                            wtens,
                                            wtensstage,
                                            isize,
                                            jsize,
                                            ksize,
                                            istride,
                                            jstride,
                                            kstride);
                                    }
                                }
                            }
                            for (int k = ksize - 1; k >= 0; --k) {
                                for (int j = jb; j < jmax; ++j) {
#pragma omp simd
                                    for (int i = ib; i < imax; ++i) {
                                        backward_sweep_k(i,
                                            j,
                                            k,
                                            ccol,
                                            dcol,
                                            datacol,
                                            wpos,
                                            wtensstage,
                                            isize,
                                            jsize,
                                            ksize,
                                            istride,
                                            jstride,
                                            kstride);
                                    }
                                }
                            }
                        }
                    }
                }
            }

          private:
            int m_iblocksize, m_jblocksize;
        };

    } // knl

} // namespace platform
