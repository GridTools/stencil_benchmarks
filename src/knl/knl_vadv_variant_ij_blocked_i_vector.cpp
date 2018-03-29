#include "knl/knl_vadv_variant_ij_blocked_i_vector.h"

namespace platform {

    namespace knl {

        template <class ValueType>
        void vadv_variant_ij_blocked_i_vector<ValueType>::vadv() {
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
                int thread_id = omp_get_thread_num();
                value_type *__restrict__ ccol_vector = ccol_v.data() + thread_id * vec_size;
                value_type *__restrict__ dcol_vector = dcol_v.data() + thread_id * vec_size;
                value_type *__restrict__ datacol_vector = datacol_v.data() + thread_id * vec_size;
#pragma omp for collapse(2)
                for (int jb = 0; jb < jsize; jb += m_jblocksize) {
                    for (int ib = 0; ib < isize; ib += m_iblocksize) {
                        const int imax = ib + m_iblocksize <= isize ? ib + m_iblocksize : isize;
                        const int jmax = jb + m_jblocksize <= jsize ? jb + m_jblocksize : jsize;
                        int i = ib;

                        for (int j = jb; j < jmax; ++j) {
                            for (int i = ib; i < imax; i += vec_size) {
                                int vector_size = i + vec_size <= imax ? vec_size : imax - i;
                                this->forward_sweep_vec(i,
                                    j,
                                    1,
                                    0,
                                    ccol,
                                    ccol_vector,
                                    dcol,
                                    dcol_vector,
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
                                    kstride,
                                    vector_size);
                                this->backward_sweep_vec(i,
                                    j,
                                    ccol,
                                    dcol,
                                    datacol_vector,
                                    upos,
                                    utensstage,
                                    isize,
                                    jsize,
                                    ksize,
                                    istride,
                                    jstride,
                                    kstride,
                                    vector_size);
                            }
                            for (int i = ib; i < imax; i += vec_size) {
                                int vector_size = i + vec_size <= imax ? vec_size : imax - i;
                                this->forward_sweep_vec(i,
                                    j,
                                    0,
                                    1,
                                    ccol,
                                    ccol_vector,
                                    dcol,
                                    dcol_vector,
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
                                    kstride,
                                    vector_size);
                                this->backward_sweep_vec(i,
                                    j,
                                    ccol,
                                    dcol,
                                    datacol_vector,
                                    vpos,
                                    vtensstage,
                                    isize,
                                    jsize,
                                    ksize,
                                    istride,
                                    jstride,
                                    kstride,
                                    vector_size);
                            }
                            for (int i = ib; i < imax; i += vec_size) {
                                int vector_size = i + vec_size <= imax ? vec_size : imax - i;
                                this->forward_sweep_vec(i,
                                    j,
                                    0,
                                    0,
                                    ccol,
                                    ccol_vector,
                                    dcol,
                                    dcol_vector,
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
                                    kstride,
                                    vector_size);
                                this->backward_sweep_vec(i,
                                    j,
                                    ccol,
                                    dcol,
                                    datacol_vector,
                                    wpos,
                                    wtensstage,
                                    isize,
                                    jsize,
                                    ksize,
                                    istride,
                                    jstride,
                                    kstride,
                                    vector_size);
                            }
                        }
                    }
                }
            }
        }
        template class vadv_variant_ij_blocked_i_vector<float>;
        template class vadv_variant_ij_blocked_i_vector<double>;

    } // namespace knl

} // namespace platform
