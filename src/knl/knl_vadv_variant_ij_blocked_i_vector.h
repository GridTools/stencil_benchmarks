#pragma once

#include "knl/knl_platform.h"
#include "knl/knl_vadv_variant.h"
#include "omp.h"

#define eproma 64 

namespace platform {

    namespace knl {

        template <class ValueType>
        class vadv_variant_ij_blocked_i_vector final : public knl_vadv_stencil_variant<ValueType> {
          public:
            using value_type = ValueType;
            using platform = knl;

            vadv_variant_ij_blocked_i_vector(const arguments_map &args)
                : knl_vadv_stencil_variant<ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
				int num_threads;
                #pragma omp parallel 
                {
                    num_threads = omp_get_num_threads();
                }
				ccol_v.resize(num_threads * eproma);	
				dcol_v.resize(num_threads * eproma);	
				datacol_v.resize(num_threads * eproma);	
            }
            ~vadv_variant_ij_blocked_i_vector() {}
            std::vector<value_type> ccol_v, dcol_v, datacol_v;

            void vadv() override;

          private:
            static constexpr value_type dtr_stage = 3.0 / 20.0;
            static constexpr value_type beta_v = 0;
            static constexpr value_type bet_m = 0.5 * (1.0 - beta_v);
            static constexpr value_type bet_p = 0.5 * (1.0 + beta_v);
            int m_iblocksize, m_jblocksize;

            __attribute__((always_inline)) void forward_sweep_vec(const int i,
                const int j,
                const int ishift,
                const int jshift,
                value_type *__restrict__ ccol,
                value_type *__restrict__ ccol_vector,
                value_type *__restrict__ dcol,
                value_type *__restrict__ dcol_vector,
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
				const int vec_size) { 

                int index = i * istride + j * jstride;

                // k minimum
#pragma omp simd
                for (int iv = 0; iv < vec_size; ++iv) {
                    ValueType gcv   = ValueType(0.25) * (wcon[index + ishift * istride + jshift * jstride + kstride + iv] + wcon[index + kstride + iv]);
                    ValueType cs    = gcv * bet_m;

                    ValueType ccol0 = gcv * bet_p;
                    ValueType bcol = dtr_stage - ccol0;

                    ValueType correction_term = -cs * (ustage[index + kstride + iv] - ustage[index + iv]);
                    ValueType dcol0 = dtr_stage * upos[index + iv] + utens[index + iv] + utensstage[index + iv] + correction_term;

                    ValueType divided = ValueType(1.0) / bcol;
                    ccol0 = ccol0 * divided;
                    dcol0 = dcol0 * divided;

                    ccol[index + iv] = ccol0;
                    dcol[index + iv] = dcol0;
                    ccol_vector[iv] = ccol0;
                    dcol_vector[iv] = dcol0;

                }
                index += kstride;

                // k body
                for (int k = 1; k < ksize - 1; ++k) {
#pragma omp simd
                	for (int iv = 0; iv < vec_size; ++iv) {
                    	ValueType gav = ValueType(-0.25) * (wcon[index + ishift * istride + jshift * jstride + iv] + wcon[index + iv]);
                    	ValueType gcv = ValueType(0.25) * (wcon[index + ishift * istride + jshift * jstride + kstride +iv] + wcon[index + kstride + iv]);

                    	ValueType as = gav * bet_m;
                   	 	ValueType cs = gcv * bet_m;

                    	ValueType acol = gav * bet_p;
        	            ValueType ccol0 = gcv * bet_p;
    	                ValueType bcol = dtr_stage - acol - ccol0;
	
        	            ValueType correction_term = -as * (ustage[index - kstride + iv] - ustage[index + iv]) - cs * (ustage[index + kstride + iv] - ustage[index + iv]);
    	                ValueType dcol0 = dtr_stage * upos[index + iv] + utens[index + iv] + utensstage[index + iv] + correction_term;
	
    	                ValueType divided = ValueType(1.0) / (bcol - ccol_vector[iv] * acol);
	                    ccol0 = ccol0 * divided;
                    	dcol0 = (dcol0 - dcol_vector[iv] * acol) * divided;

                    	ccol[index + iv] = ccol0;
                    	dcol[index + iv] = dcol0;
                    	ccol_vector[iv] = ccol0;
                    	dcol_vector[iv] = dcol0;

					}
                    index += kstride;
                }

                // k maximum
#pragma omp simd
                for (int iv = 0; iv < vec_size; ++iv) {
                    ValueType gav = ValueType(-0.25) * (wcon[index + ishift * istride + jshift * jstride + iv] + wcon[index + iv]);

                    ValueType as = gav * bet_m;

                    ValueType acol = gav * bet_p;
                    ValueType bcol = dtr_stage - acol;

                    ValueType correction_term = -as * (ustage[index - kstride + iv] - ustage[index + iv]);
                    ValueType dcol0 = dtr_stage * upos[index + iv] + utens[index + iv] + utensstage[index + iv] + correction_term;

                    ValueType divided = ValueType(1.0) / (bcol - ccol_vector[iv] * acol);
                    dcol0 = (dcol0 - dcol_vector[iv] * acol) * divided;

                    ccol[index + iv] = ccol_vector[iv];
                    dcol[index + iv] = dcol0;
                }
			}


            __attribute__((always_inline)) inline void backward_sweep_vec(const int i,
                const int j,
                const value_type *__restrict__ ccol,
                const value_type *__restrict__ dcol,
                value_type *__restrict__ datacol_vector,
                const value_type *__restrict__ upos,
                value_type *__restrict__ utensstage,
                const int isize,
                const int jsize,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride, 
				const int vec_size) { 
                constexpr value_type dtr_stage = 3.0 / 20.0;

                int index = i * istride + j * jstride + (ksize - 1) * kstride;
                // k
#pragma omp simd
                for (int iv = 0; iv < vec_size; ++iv){
                    datacol_vector[iv] = dcol[index + iv];
                    utensstage[index + iv] = dtr_stage * (datacol_vector[iv] - upos[index + iv]);

                }
                index -= kstride;

                // k body
#pragma omp simd
                for (int k = ksize - 2; k >= 0; --k) {
                	for (int iv = 0; iv < vec_size; ++iv){
                    	datacol_vector[iv] = dcol[index + iv] - ccol[index + iv] * datacol_vector[iv];
                    	utensstage[index + iv] = dtr_stage * (datacol_vector[iv] - upos[index + iv]);
					}

                    index -= kstride;
				}
            }
        };

        extern template class vadv_variant_ij_blocked_i_vector<float>;
        extern template class vadv_variant_ij_blocked_i_vector<double>;

    } // knl

} // namespace platform
