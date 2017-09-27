#pragma once

#include "x86/x86_fast_waves_uv_stencil_variant.h"

namespace platform {

    namespace x86 {

        template <class Platform, class ValueType>
        class x86_fast_waves_uv_variant_ij_blocked final : public x86_fast_waves_uv_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;

            x86_fast_waves_uv_variant_ij_blocked(const arguments_map &args)
                : x86_fast_waves_uv_stencil_variant<Platform, ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
            }

            void fast_waves_uv() override {
    
                const ValueType dt_small = 10;
                const ValueType edadlat = 1;
                
                int total_size = this->storage_size();
        
                ValueType* frefUField = this->u_out();
                ValueType* frefVField = this->v_out();
        
                ValueType* fupos = this->u_pos();
                ValueType* fvpos = this->v_pos();
        
                ValueType* futensstage = this->u_tens();
                ValueType* fvtensstage = this->v_tens();
        
                ValueType* frho = this->rho();
                ValueType* fppuv = this->ppuv();
                ValueType* ffx = this->fx();
        
                ValueType* frho0 = this->rho0();
                ValueType* fcwp = this->cwp();
                ValueType* fp0 = this->p0();
                ValueType* fwbbctens_stage = this->wbbctens_stage();
                ValueType* fwgtfac = this->wgtfac();
                ValueType* fhhl = this->hhl();
                ValueType* fxlhsx = this->xlhsx();
                ValueType* fxlhsy = this->xlhsy();
                ValueType* fxdzdx = this->xdzdx();
                ValueType* fxdzdy = this->xdzdy();
        
                ValueType* fxrhsy = this->xrhsy_ref();
                ValueType* fxrhsx = this->xrhsx_ref();
                ValueType* fxrhsz = this->xrhsz_ref();
                ValueType* fppgradcor = this->ppgradcor();
                ValueType* fppgradu = this->ppgradu();
                ValueType* fppgradv = this->ppgradv();
        
                const int cFlatLimit=10;
                const int istride = this->istride();
                const int jstride = this->jstride();
                const int kstride = this->kstride();
                const int h = this->halo();
                const int isize = this->isize();
                const int jsize = this->jsize();
                const int ksize = this->ksize();
            
                auto computePPGradCor = [&](int i, int j, int k) {
                    fppgradcor[this->index(i,j,k)] = fwgtfac[this->index(i,j,k)] * fppuv[this->index(i,j,k)] + 
                        ((ValueType)1.0 - fwgtfac[this->index(i,j,k)]) * fppuv[this->index(i,j,k-1)];
                };
    
                //PPGradCorStage
                #pragma omp parallel for collapse(2)
                for (int jb = 0; jb < jsize+1; jb += m_jblocksize) {
                    for (int ib = 0; ib < isize+1; ib += m_iblocksize) {
                        const int imax = ib + m_iblocksize <= isize+1 ? ib + m_iblocksize : isize+1;
                        const int jmax = jb + m_jblocksize <= jsize+1 ? jb + m_jblocksize : jsize+1;
                        int k=cFlatLimit;                        
                        
                        //PPGradCorStage
                        for (int i = ib; i < imax; ++i) {
                            for (int j = jb; j < jmax; ++j) {
                                computePPGradCor(i,j,k);
                            }
                        }
 
                        for(k=cFlatLimit+1; k < ksize; ++k) {
                            for (int i = ib; i < imax; ++i) {
                                for (int j = jb; j < jmax; ++j) {
                                    computePPGradCor(i,j,k);
                                    fppgradcor[this->index(i,j,k-1)] = (fppgradcor[this->index(i,j,k)] - fppgradcor[this->index(i,j,k-1)]);
                                }
                            }
                        }
                    }
                }

                int k=ksize-1;                
                #pragma omp parallel for collapse(2)
                for (int jb = 0; jb < jsize; jb += m_jblocksize) {
                    for (int ib = 0; ib < isize; ib += m_iblocksize) {
                        const int imax = ib + m_iblocksize <= isize ? ib + m_iblocksize : isize;
                        const int jmax = jb + m_jblocksize <= jsize ? jb + m_jblocksize : jsize;

                        // XRHSXStage
                        // FullDomain
                        for (int i = ib-1; i < imax; ++i) {
                            for (int j = jb; j < jmax+1; ++j) {
                                fxrhsx[this->index(i,j,k)] = -ffx[this->index(i,j,k)] / ((ValueType)0.5*(frho[this->index(i,j,k)] +frho[this->index(i+1,j,k)])) * (fppuv[this->index(i+1,j,k)] - fppuv[this->index(i,j,k)]) +
                                    futensstage[this->index(i,j,k)];
                                fxrhsy[this->index(i,j,k)] = -edadlat / ((ValueType)0.5*(frho[this->index(i,j+1,k)] + frho[this->index(i,j,k)])) * (fppuv[this->index(i,j+1,k)]-fppuv[this->index(i,j,k)])
                                    +fvtensstage[this->index(i,j,k)];
                            }
                        }
                        for (int i = ib; i < imax+1; ++i) {
                            for (int j = jb-1; j < jmax; ++j) {
                                fxrhsy[this->index(i,j,k)] = -edadlat / ((ValueType)0.5*(frho[this->index(i,j+1,k)] + frho[this->index(i,j,k)])) * (fppuv[this->index(i,j+1,k)]-fppuv[this->index(i,j,k)])
                                    +fvtensstage[this->index(i,j,k)];
                            }
                        }
                        for(int i = ib; i < imax+1; ++i) {
                            for (int j = jb; j < jmax+1; ++j) {
                                fxrhsz[this->index(i,j,k)] = frho0[this->index(i,j,k)] / frho[this->index(i,j,k)] * 9.8 *
                                    ((ValueType)1.0 - fcwp[this->index(i,j,k)] * (fp0[this->index(i,j,k)] + fppuv[this->index(i,j,k)])) +
                                    fwbbctens_stage[this->index(i,j,k+1)];
                            }
                        }
                    }
                }

                //PPGradStage
                for(k=0; k < ksize-1; ++k) {                                    
                    #pragma omp parallel for collapse(2)
                    for (int jb = 0; jb < jsize; jb += m_jblocksize) {
                        for (int ib = 0; ib < isize; ib += m_iblocksize) {
                            const int imax = ib + m_iblocksize <= isize ? ib + m_iblocksize : isize;
                            const int jmax = jb + m_jblocksize <= jsize ? jb + m_jblocksize : jsize;
                
                            for(int i = ib; i < imax; ++i) {
                                for (int j = jb; j < jmax; ++j) {
                                    if(k < cFlatLimit) {
                                        fppgradu[this->index(i,j,k)] = (fppuv[this->index(i+1,j,k)]-fppuv[this->index(i,j,k)]);
                                        fppgradv[this->index(i,j,k)] = (fppuv[this->index(i,j+1,k)]-fppuv[this->index(i,j,k)]);
                                    } else {
                                        fppgradu[this->index(i,j,k)] = (fppuv[this->index(i+1,j,k)]-fppuv[this->index(i,j,k)]) + (fppgradcor[this->index(i+1,j,k)] + fppgradcor[this->index(i,j,k)])*
                                            (ValueType)0.5 * ( (fhhl[this->index(i,j,k+1)] + fhhl[this->index(i,j,k)]) - (fhhl[this->index(i+1,j,k+1)]+fhhl[this->index(i+1,j,k)])) /
                                            ( (fhhl[this->index(i,j,k+1)] - fhhl[this->index(i,j,k)]) + (fhhl[this->index(i+1,j,k+1)] - fhhl[this->index(i+1,j,k)]));
                                        fppgradv[this->index(i,j,k)] = (fppuv[this->index(i,j+1,k)]-fppuv[this->index(i,j,k)]) + (fppgradcor[this->index(i,j+1,k)] + fppgradcor[this->index(i,j,k)])*
                                            (ValueType)0.5 * ( (fhhl[this->index(i,j,k+1)] + fhhl[this->index(i,j,k)]) - (fhhl[this->index(i,j+1,k+1)]+fhhl[this->index(i,j+1,k)])) /
                                            ( (fhhl[this->index(i,j,k+1)] - fhhl[this->index(i,j,k)]) + (fhhl[this->index(i,j+1,k+1)] - fhhl[this->index(i,j+1,k)]));
                                    }
                                }
                            }

                            for(int i = ib; i < imax; ++i) {
                                for (int j = jb; j < jmax; ++j) {
                                    ValueType rhou = ffx[this->index(i,j,k)] / ((ValueType)0.5*(frho[this->index(i+1,j,k)] + frho[this->index(i,j,k)]));
                                    ValueType rhov = edadlat / ((ValueType)0.5*(frho[this->index(i,j+1,k)] + frho[this->index(i,j,k)]));

                                    frefUField[this->index(i,j,k)] = fupos[this->index(i,j,k)] + (futensstage[this->index(i,j,k)] - fppgradu[this->index(i,j,k)]*rhou) * dt_small;
                                    frefVField[this->index(i,j,k)] = fvpos[this->index(i,j,k)] + (fvtensstage[this->index(i,j,k)] - fppgradv[this->index(i,j,k)]*rhov) * dt_small;
                                }
                            }
                        }
                    }
                }

                k = ksize-1;

                #pragma omp parallel for collapse(2)
                for (int jb = 0; jb < jsize; jb += m_jblocksize) {
                    for (int ib = 0; ib < isize; ib += m_iblocksize) {
                        const int imax = ib + m_iblocksize <= isize ? ib + m_iblocksize : isize;
                        const int jmax = jb + m_jblocksize <= jsize ? jb + m_jblocksize : jsize;
                        
                        for(int i = ib; i < imax; ++i) {
                            for (int j = jb; j < jmax; ++j) {
                                ValueType bottU = fxlhsx[this->index(i,j,k)] * fxdzdx[this->index(i,j,k)] * (
                                        (ValueType)0.5*(fxrhsz[this->index(i+1,j,k)]+fxrhsz[this->index(i,j,k)]) -
                                        fxdzdx[this->index(i,j,k)] * fxrhsx[this->index(i,j,k)] -
                                        (ValueType)0.5*( (ValueType)0.5*(fxdzdy[this->index(i+1,j-1,k)]+fxdzdy[this->index(i+1,j,k)]) + (ValueType)0.5*(fxdzdy[this->index(i,j-1,k)]+fxdzdy[this->index(i,j,k)])) *
                                        (ValueType)0.5*( (ValueType)0.5*(fxrhsy[this->index(i+1,j-1,k)]+fxrhsy[this->index(i+1,j,k)]) + (ValueType)0.5*(fxrhsy[this->index(i,j-1,k)]+fxrhsy[this->index(i,j,k)]))
                                    ) + fxrhsx[this->index(i,j,k)];
                                frefUField[this->index(i,j,k)] = fupos[this->index(i,j,k)] + bottU * dt_small;
                                ValueType bottV = fxlhsy[this->index(i,j,k)] * fxdzdy[this->index(i,j,k)] * (
                                        (ValueType)0.5*(fxrhsz[this->index(i,j+1,k)]+fxrhsz[this->index(i,j,k)]) -
                                        fxdzdy[this->index(i,j,k)] * fxrhsy[this->index(i,j,k)] -
                                        (ValueType)0.5*( (ValueType)0.5*(fxdzdx[this->index(i-1,j+1,k)]+fxdzdx[this->index(i,j+1,k)]) + (ValueType)0.5*(fxdzdx[this->index(i-1,j,k)]+fxdzdx[this->index(i,j,k)])) *
                                        (ValueType)0.5*( (ValueType)0.5*(fxrhsx[this->index(i-1,j+1,k)]+fxrhsx[this->index(i,j+1,k)]) + (ValueType)0.5*(fxrhsx[this->index(i-1,j,k)]+fxrhsx[this->index(i,j,k)]))
                                ) + fxrhsy[this->index(i,j,k)];
                                frefVField[this->index(i,j,k)] = fvpos[this->index(i,j,k)]+bottV*dt_small;
                            }
                        }
                    }
                }

            }

          private:
            int m_iblocksize, m_jblocksize;
        };

    } // namespace x86

} // namespace platform
