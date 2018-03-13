#pragma once

#include "knl/knl_platform.h"
#include "knl/knl_vadv_variant.h"
#include "omp.h"

#define eproma 128 

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
				m_ccol0.resize(num_threads * eproma); 
				m_ccol1.resize(num_threads * eproma); 
				m_dcol0.resize(num_threads * eproma); 
				m_dcol1.resize(num_threads * eproma); 
				m_ustage0.resize(num_threads * eproma); 
				m_ustage1.resize(num_threads * eproma); 
				m_ustage2.resize(num_threads * eproma); 
				m_wcon0.resize(num_threads * eproma); 
				m_wcon1.resize(num_threads * eproma); 
				m_wcon_shift0.resize(num_threads * eproma); 
				m_wcon_shift1.resize(num_threads * eproma); 
				m_datacol.resize(num_threads * eproma); 
            }
            ~vadv_variant_ij_blocked_i_vector() {}

            void vadv() override;

          private:
            static constexpr value_type dtr_stage = 3.0 / 20.0;
            static constexpr value_type beta_v = 0;
            static constexpr value_type bet_m = 0.5 * (1.0 - beta_v);
            static constexpr value_type bet_p = 0.5 * (1.0 + beta_v);
            int m_iblocksize, m_jblocksize;
            std::vector<value_type> m_ccol0, m_ccol1;
            std::vector<value_type> m_dcol0, m_dcol1;
            std::vector<value_type> m_ustage0, m_ustage1, m_ustage2;
            std::vector<value_type> m_wcon0, m_wcon1;
            std::vector<value_type> m_wcon_shift0, m_wcon_shift1;
            std::vector<value_type> m_datacol;


            __attribute__((always_inline)) void forward_sweep_vec(const int i,
                const int j,
                const int ishift,
                const int jshift,
                value_type *__restrict__ ccol,
                value_type *__restrict__ dcol,
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
				const int vec_size, 
				const int thread_id) {

				value_type *__restrict__ ccol0 = m_ccol0.data() + thread_id * eproma;
				value_type *__restrict__ ccol1 = m_ccol1.data() + thread_id * eproma;
				value_type *__restrict__ dcol0 = m_dcol0.data() + thread_id * eproma;
				value_type *__restrict__ dcol1 = m_dcol1.data() + thread_id * eproma;
				value_type *__restrict__ ustage0 = m_ustage0.data() + thread_id * eproma;
				value_type *__restrict__ ustage1 = m_ustage1.data() + thread_id * eproma;
				value_type *__restrict__ ustage2 = m_ustage2.data() + thread_id * eproma;
				value_type *__restrict__ wcon0 = m_wcon0.data() + thread_id * eproma;
				value_type *__restrict__ wcon1 = m_wcon1.data() + thread_id * eproma;
				value_type *__restrict__ wcon_shift0 = m_wcon_shift0.data() + thread_id * eproma;
				value_type *__restrict__ wcon_shift1 = m_wcon_shift1.data() + thread_id * eproma;
                int index = i * istride + j * jstride;

                // k minimum
#pragma omp simd
                for (int iv = 0; iv < vec_size; ++iv) {
                    wcon_shift0[iv] = wcon[index + ishift * istride + jshift * jstride + kstride + iv];
                    wcon0[iv] 		= wcon[index + kstride + iv];
                    ValueType gcv   = ValueType(0.25) * (wcon_shift0[iv] + wcon0[iv]);
                    ValueType cs    = gcv * bet_m;

                    ccol0[iv] = gcv * bet_p;
                    ValueType bcol = dtr_stage - ccol0[iv];

                    ustage0[iv] = ustage[index + kstride + iv];
                    ustage1[iv] = ustage[index + iv];
                    ValueType correction_term = -cs * (ustage0[iv] - ustage1[iv]);
                    dcol0[iv] = dtr_stage * upos[index + iv] + utens[index + iv] + utensstage[index + iv] + correction_term;

                    ValueType divided = ValueType(1.0) / bcol;
                    ccol0[iv] = ccol0[iv] * divided;
                    dcol0[iv] = dcol0[iv] * divided;

                    ccol[index + iv] = ccol0[iv];
                    dcol[index + iv] = dcol0[iv];

                }
                index += kstride;

                // k body
                for (int k = 1; k < ksize - 1; ++k) {
#pragma omp simd
					for (int iv = 0; iv < vec_size; ++iv){
                    	ccol1[iv] = ccol0[iv];
                    	dcol1[iv] = dcol0[iv];
                    	ustage2[iv] = ustage1[iv];
                    	ustage1[iv] = ustage0[iv];
                    	wcon1[iv] = wcon0[iv];
                    	wcon_shift1[iv] = wcon_shift0[iv];
					}
#pragma omp simd
                	for (int iv = 0; iv < vec_size; ++iv) {
                    	ValueType gav = ValueType(-0.25) * (wcon_shift1[iv] + wcon1[iv]);
                    	wcon_shift0[iv] = wcon[index + ishift * istride + jshift * jstride + kstride + iv];
                    	wcon0[iv] 		= wcon[index + kstride + iv];
                    	ValueType gcv = ValueType(0.25) * (wcon_shift0[iv] + wcon0[iv]);

                    	ValueType as = gav * bet_m;
                   	 	ValueType cs = gcv * bet_m;

                    	ValueType acol = gav * bet_p;
        	            ccol0[iv] = gcv * bet_p;
    	                ValueType bcol = dtr_stage - acol - ccol0[iv];
	
            	        ustage0[iv] = ustage[index + kstride + iv];
        	            ValueType correction_term = -as * (ustage2[iv] - ustage1[iv]) - cs * (ustage0[iv] - ustage1[iv]);
    	                dcol0[iv] = dtr_stage * upos[index + iv] + utens[index + iv] + utensstage[index + iv] + correction_term;
	
    	                ValueType divided = ValueType(1.0) / (bcol - ccol1[iv] * acol);
	                    ccol0[iv] = ccol0[iv] * divided;
                    	dcol0[iv] = (dcol0[iv] - dcol1[iv] * acol) * divided;

                    	ccol[index + iv] = ccol0[iv];
                    	dcol[index + iv] = dcol0[iv];

					}
                    index += kstride;
                }

                // k maximum
#pragma omp simd
				for (int iv = 0; iv < vec_size; ++iv){
                 	ccol1[iv] = ccol0[iv];
                   	dcol1[iv] = dcol0[iv];
                   	ustage2[iv] = ustage1[iv];
                   	ustage1[iv] = ustage0[iv];
                   	wcon1[iv] = wcon0[iv];
                   	wcon_shift1[iv] = wcon_shift0[iv];
				}

#pragma omp simd
                for (int iv = 0; iv < vec_size; ++iv) {
                    ValueType gav = ValueType(-0.25) * (wcon_shift1[iv] + wcon1[iv]);

                    ValueType as = gav * bet_m;

                    ValueType acol = gav * bet_p;
                    ValueType bcol = dtr_stage - acol;

                    ValueType correction_term = -as * (ustage2[iv] - ustage1[iv]);
                    dcol0[iv] = dtr_stage * upos[index + iv] + utens[index + iv] + utensstage[index + iv] + correction_term;

                    ValueType divided = ValueType(1.0) / (bcol - ccol1[iv] * acol);
                    dcol0[iv] = (dcol0[iv] - dcol1[iv] * acol) * divided;

                    ccol[index + iv] = ccol0[iv];
                    dcol[index + iv] = dcol0[iv];
                }
			}


            __attribute__((always_inline)) inline void backward_sweep_vec(const int i,
                const int j,
                const value_type *__restrict__ ccol,
                const value_type *__restrict__ dcol,
                const value_type *__restrict__ upos,
                value_type *__restrict__ utensstage,
                const int isize,
                const int jsize,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride, 
				const int vec_size, 
				const int thread_id) {
                constexpr value_type dtr_stage = 3.0 / 20.0;

				value_type *__restrict__ datacol = m_datacol.data() + thread_id * eproma;

                int index = i * istride + j * jstride + (ksize - 1) * kstride;
                // k
#pragma omp simd
                for (int iv = 0; iv < vec_size; ++iv){
                    datacol[iv] = dcol[index + iv];
                    utensstage[index + iv] = dtr_stage * (datacol[iv] - upos[index + iv]);

                }
                index -= kstride;

                // k body
#pragma omp simd
                for (int k = ksize - 2; k >= 0; --k) {
                	for (int iv = 0; iv < vec_size; ++iv){
                    	datacol[iv] = dcol[index + iv] - ccol[index + iv] * datacol[iv];
                    	utensstage[index + iv] = dtr_stage * (datacol[iv] - upos[index + iv]);
					}

                    index -= kstride;
				}
            }
        };

        extern template class vadv_variant_ij_blocked_i_vector<float>;
        extern template class vadv_variant_ij_blocked_i_vector<double>;

    } // knl

} // namespace platform
