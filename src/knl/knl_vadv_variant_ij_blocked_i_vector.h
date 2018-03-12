#pragma once

#include "knl/knl_platform.h"
#include "knl/knl_vadv_variant.h"

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
            }
            ~vadv_variant_ij_blocked_i_vector() {}

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
				const int vec_size) {

                std::vector<value_type> ccol0(eproma), ccol1(eproma);
                std::vector<value_type> dcol0(eproma), dcol1(eproma);
                std::vector<value_type> ustage0(eproma), ustage1(eproma), ustage2(eproma);
                std::vector<value_type> wcon0(eproma), wcon1(eproma);
                std::vector<value_type> wcon_shift0(eproma), wcon_shift1(eproma);

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
                    ccol1 = ccol0;
                    dcol1 = dcol0;
                    ustage2 = ustage1;
                    ustage1 = ustage0;
                    wcon1 = wcon0;
                    wcon_shift1 = wcon_shift0;
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
                ccol1 = ccol0;
                dcol1 = dcol0;
                ustage2 = ustage1;
                ustage1 = ustage0;
                wcon1 = wcon0;
                wcon_shift1 = wcon_shift0;

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
				const int vec_size) {
                constexpr value_type dtr_stage = 3.0 / 20.0;

                std::vector<value_type> datacol(eproma);

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
