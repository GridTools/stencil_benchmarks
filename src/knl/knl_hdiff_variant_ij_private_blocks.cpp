#include "knl/knl_hdiff_variant_ij_private_blocks.h"
#include "omp.h"

namespace platform {

	namespace knl {

        template <class ValueType>                                                                                            
    	void hdiff_variant_ij_private_blocks<ValueType>::prerun_init() {	
			variant_base::prerun();                                                                                                           
			value_type *__restrict__ in = this->in();
			value_type *__restrict__ coeff = this->coeff();
			const int isize = this->isize(); 
			const int jsize = this->jsize(); 
			const int ksize = this->ksize(); 
        	double dx = 1. / (double)(isize);                                                                                              
        	double dy = 1. / (double)(jsize);                                                                                              
        	double dz = 1. / (double)(ksize);      
#pragma omp parallel for collapse(3)
        	for (int k = 0; k < ksize; ++k) {
            	for (int jb = 0; jb < jsize; jb += m_jblocksize) {
                	for (int ib = 0; ib < isize; ib += m_iblocksize) {
                   		const int imax = ib + m_iblocksize <= isize ? ib + m_iblocksize : isize;
                    	const int jmax = jb + m_jblocksize <= jsize ? jb + m_jblocksize : jsize;

                    	int index = ib + jb * isize + k * isize * jsize;
                    	for (int j = jb; j < jmax; ++j) {
                        	for (int i = ib; i < imax; ++i) {
                            	double x = dx * (double)(ib);
                            	double y = dy * (double)(jb);
                            	double z = dz * (double)(k);
                            	in[index] = 3.0 +
                                        	1.25 * (2.5 + cos(M_PI * (18.4 * x + 20.3 * y)) +
                                    	            0.78 * sin(2 * M_PI * (18.4 * x + 20.3 * y) * z)) /
                                	                4.;
                            	coeff[index] = 1.4 +
                                        	    0.87 * (0.3 + cos(M_PI * (1.4 * x + 2.3 * y)) +
                                    	                1.11 * sin(2 * M_PI * (1.4 * x + 2.3 * y) * z)) /
                                	                    4.;
                           		index++;
                        	}
                    	    index += isize - (imax - ib);
                	    }
            	    }
        	    }
        	}
		}

		template <class ValueType>
		void hdiff_variant_ij_private_blocks<ValueType>::hdiff() {
			const value_type *__restrict__ in = this->in();
			const value_type *__restrict__ coeff = this->coeff();
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
				/*std::vector<value_type> lap_data ((m_iblocksize + 2) * (m_jblocksize + 2)); 
				std::vector<value_type> flx_data ((m_iblocksize + 1) * (m_jblocksize)); 
				std::vector<value_type> fly_data ((m_iblocksize) * (m_jblocksize + 1)); */
				int thread_id = omp_get_thread_num();
				value_type *__restrict__ lap = lap_data.data() + thread_id * (m_iblocksize + 2) * (m_jblocksize + 2);
				value_type *__restrict__ flx = flx_data.data() + thread_id * (m_iblocksize + 1) * (m_jblocksize);
				value_type *__restrict__ fly = fly_data.data() + thread_id * (m_iblocksize) * (m_jblocksize + 1);
#pragma omp for collapse(3)
				for (int k = 0; k < ksize; ++k) {
					for (int jb = 0; jb < jsize; jb += m_jblocksize) {
						for (int ib = 0; ib < isize; ib += m_iblocksize) {

							const int i_blocksize_lap = ib + m_iblocksize <= isize + 2 ? m_iblocksize + 2 : isize + 2 - ib; //possibly optimize
							const int j_blocksize_lap = jb + m_jblocksize <= jsize + 2 ? m_jblocksize + 2 : jsize + 2 - jb;
							const int i_blocksize_flx = ib + m_iblocksize <= isize + 1 ? m_iblocksize + 1 : isize + 1 - ib;
							const int j_blocksize_flx = jb + m_jblocksize <= jsize ? m_jblocksize : jsize - jb;
							const int i_blocksize_fly = ib + m_iblocksize <= isize ? m_iblocksize : isize - ib;
							const int j_blocksize_fly = jb + m_jblocksize <= jsize + 1 ? m_jblocksize + 1 : jsize + 1 - jb;
							const int i_blocksize_out = ib + m_iblocksize <= isize ? m_iblocksize : isize - ib;
							const int j_blocksize_out = jb + m_jblocksize <= jsize ? m_jblocksize : jsize - jb;

							int index = (ib - 1) * istride + (jb - 1) * jstride + k * kstride;
							for (int j = 0; j < j_blocksize_lap; ++j) {
								int j_base = j * i_blocksize_lap; 
								index = index - j_base;
#pragma omp simd
								for (int i = j_base; i < j_base + i_blocksize_lap; ++i) {
									lap[i] = 4 * in[index + i] - (in[index + i - istride] + in[index + i + istride] +
											in[index + i - jstride] + in[index + i + jstride]);
								}
								index += (jstride + j_base);
							}

							//TODO: either optimize for this loop or merge with next loop 
							index = (ib - 1) * istride + jb * jstride + k * kstride;
							for (int j = 0; j < j_blocksize_flx; ++j) {
								int j_base = j * i_blocksize_flx; 
								index = index - j_base;
#pragma omp simd
								for (int i = j_base; i < j_base + i_blocksize_flx; ++i) {
									flx[i] = lap[(i + 1) + (j + 1) * i_blocksize_lap - j_base] - lap[i + (j + 1) * i_blocksize_lap - j_base];
									if (flx[i] * (in[index + istride + i] - in[index + i]) > 0)
										flx[i] = 0.;
								}
								index += (jstride + j_base);
							}

							index = ib * istride + (jb - 1) * jstride + k * kstride;
							for (int j = 0; j < j_blocksize_fly; ++j) {
								int j_base = j * i_blocksize_fly; 
								index = index - j_base;
#pragma omp simd
								for (int i = j_base; i < j_base + i_blocksize_fly; ++i) {
									fly[i] = lap[(i + 1) + (j + 1) * i_blocksize_lap - j_base] - lap[(i + 1) + j * i_blocksize_lap - j_base];
									if (fly[i] * (in[index + jstride + i] - in[index + i]) > 0)
										fly[i] = 0.;
								}
								index += (jstride + j_base);
							}

							index = ib * istride + jb * jstride + k * kstride;
							for (int j = 0; j < j_blocksize_out; ++j) {
								int j_base = j * i_blocksize_out; 
								index = index - j_base;
#pragma omp simd
								for (int i = j_base; i < j_base + i_blocksize_out; ++i) {
									out[index + i] = in[index + i] -
										coeff[index + i] * (flx[(i + 1) + j * i_blocksize_flx - j_base] - flx[i + j * i_blocksize_flx - j_base] +
												fly[i + (j + 1) * i_blocksize_fly - j_base] - fly[i + j * i_blocksize_fly - j_base]);
								}
								index += (jstride + j_base);
							}
						}
					}
				}
			}
		}

		template class hdiff_variant_ij_private_blocks<float>;
		template class hdiff_variant_ij_private_blocks<double>;

	} // namespace knl

} // namespace platform
