#include <omp.h>

#include <storage/storage-facility.hpp>

#include "tools.h"
#include "defs.h"

constexpr int h = 2;
typedef gridtools::storage_traits<gridtools::enumtype::Host> storage_tr;
typedef storage_tr::custom_layout_storage_info_t<0, gridtools::layout_map< LAYOUT >, gridtools::halo< h,h,h >, gridtools::alignment< ALIGN > > storage_info_t;

template<typename T>
void copy( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int ksize, const unsigned int isize, const unsigned int jsize) {
    const int NBI = isize/BLOCKSIZEX;
    const int NBJ = jsize/BLOCKSIZEY;

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX;
            int start_index_j = bj*BLOCKSIZEY;
            const int idx = si.index(start_index_i+h,start_index_j+h,h)-si.get_initial_offset();
            for(int i=0; i<BLOCKSIZEX; ++i) {
                for(int j=0; j<BLOCKSIZEY; ++j) {
                    for(int k=0; k < ksize; ++k) {
                        b[idx+si.index(i,j,k)] = a[idx+si.index(i,j,k)];
                    }
                }
            }
        }
    }
}

template<typename T>
void copyi1( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int ksize, const unsigned int isize, const unsigned int jsize) {
    const int NBI = isize/BLOCKSIZEX;
    const int NBJ = jsize/BLOCKSIZEY;

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX;
            int start_index_j = bj*BLOCKSIZEY;
            const int idx = si.index(start_index_i+h,start_index_j+h,h)-si.get_initial_offset();
            for(int i=0; i<BLOCKSIZEX; ++i) {
                for(int j=0; j<BLOCKSIZEY; ++j) {
                    for(int k=0; k < ksize; ++k) {
                        b[idx+si.index(i,j,k)] = a[idx+si.index(i+1,j,k)];
                    }
                }
            }
        }
    }
}

template<typename T>
void sumi1( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int ksize, const unsigned int isize, const unsigned int jsize) {
    const int NBI = isize/BLOCKSIZEX;
    const int NBJ = jsize/BLOCKSIZEY;

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX;
            int start_index_j = bj*BLOCKSIZEY;
            const int idx = si.index(start_index_i+h,start_index_j+h,h)-si.get_initial_offset();
            for(int i=0; i<BLOCKSIZEX; ++i) {
                for(int j=0; j<BLOCKSIZEY; ++j) {
                    for(int k=0; k < ksize; ++k) {
                        b[idx+si.index(i,j,k)] = a[idx+si.index(i,j,k)] + a[idx+si.index(i+1,j,k)];
                    }
                }
            }
        }
    }
}

template<typename T>
void avgi( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int ksize, const unsigned int isize, const unsigned int jsize) {
    const int NBI = isize/BLOCKSIZEX;
    const int NBJ = jsize/BLOCKSIZEY;

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX;
            int start_index_j = bj*BLOCKSIZEY;
            const int idx = si.index(start_index_i+h,start_index_j+h,h)-si.get_initial_offset();
            for(int i=0; i<BLOCKSIZEX; ++i) {
                for(int j=0; j<BLOCKSIZEY; ++j) {
                    for(int k=0; k < ksize; ++k) {
                        b[idx+si.index(i,j,k)] = a[idx+si.index(i-1,j,k)] + a[idx+si.index(i+1,j,k)];
                    }
                }
            }
        }
    }
}

template<typename T>
void sumj1( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int ksize, const unsigned int isize, const unsigned int jsize) {
    const int NBI = isize/BLOCKSIZEX;
    const int NBJ = jsize/BLOCKSIZEY;

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX;
            int start_index_j = bj*BLOCKSIZEY;
            const int idx = si.index(start_index_i+h,start_index_j+h,h)-si.get_initial_offset();
            for(int i=0; i<BLOCKSIZEX; ++i) {
                for(int j=0; j<BLOCKSIZEY; ++j) {
                    for(int k=0; k < ksize; ++k) {
                        b[idx+si.index(i,j,k)] = a[idx+si.index(i,j,k)] + a[idx+si.index(i,j+1,k)];
                    }
                }
            }
        }
    }
}

template<typename T>
void avgj( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int ksize, const unsigned int isize, const unsigned int jsize) {
    const int NBI = isize/BLOCKSIZEX;
    const int NBJ = jsize/BLOCKSIZEY;

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX;
            int start_index_j = bj*BLOCKSIZEY;
            const int idx = si.index(start_index_i+h,start_index_j+h,h)-si.get_initial_offset();
            for(int i=0; i<BLOCKSIZEX; ++i) {
                for(int j=0; j<BLOCKSIZEY; ++j) {
                    for(int k=0; k < ksize; ++k) {
                        b[idx+si.index(i,j,k)] = a[idx+si.index(i,j-1,k)] + a[idx+si.index(i,j+1,k)];
                    }
                }
            }
        }
    }
}

template<typename T>
void sumk1( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int ksize, const unsigned int isize, const unsigned int jsize) {
    const int NBI = isize/BLOCKSIZEX;
    const int NBJ = jsize/BLOCKSIZEY;

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX;
            int start_index_j = bj*BLOCKSIZEY;
            const int idx = si.index(start_index_i+h,start_index_j+h,h)-si.get_initial_offset();
            for(int i=0; i<BLOCKSIZEX; ++i) {
                for(int j=0; j<BLOCKSIZEY; ++j) {
                    for(int k=0; k < ksize; ++k) {
                        b[idx+si.index(i,j,k)] = a[idx+si.index(i,j,k)] + a[idx+si.index(i,j,k+1)];
                    }
                }
            }
        }
    }    
}

template<typename T>
void avgk( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int ksize, const unsigned int isize, const unsigned int jsize) {
    const int NBI = isize/BLOCKSIZEX;
    const int NBJ = jsize/BLOCKSIZEY;

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX;
            int start_index_j = bj*BLOCKSIZEY;
            const int idx = si.index(start_index_i+h,start_index_j+h,h)-si.get_initial_offset();
            for(int i=0; i<BLOCKSIZEX; ++i) {
                for(int j=0; j<BLOCKSIZEY; ++j) {
                    for(int k=0; k < ksize; ++k) {
                        b[idx+si.index(i,j,k)] = a[idx+si.index(i,j,k-1)] + a[idx+si.index(i,j,k+1)];
                    }
                }
            }
        }
    }
}


template<typename T>
void lap( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int ksize, const unsigned int isize, const unsigned int jsize) {
    const int NBI = isize/BLOCKSIZEX;
    const int NBJ = jsize/BLOCKSIZEY;

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX;
            int start_index_j = bj*BLOCKSIZEY;
            const int idx = si.index(start_index_i+h,start_index_j+h,h)-si.get_initial_offset();
            for(int i=0; i<BLOCKSIZEX; ++i) {
                for(int j=0; j<BLOCKSIZEY; ++j) {
                    for(int k=0; k < ksize; ++k) {
                        b[idx+si.index(i,j,k)] = a[idx+si.index(i,j,k)] + a[idx+si.index(i-1,j,k)] + a[idx+si.index(i+1,j,k)]
                            + a[idx+si.index(i,j+1,k)] + a[idx+si.index(i,j-1,k)];
                    }
                }
            }
        }
    }
}


template<typename T>
void launch( std::vector<double>& timings, const unsigned int isize, const unsigned int jsize, const unsigned ksize, const unsigned tsteps, const unsigned warmup_step ) {

    const storage_info_t si(isize, jsize, ksize);

    std::cout << "Size: " << isize << " " << jsize << " " << ksize << std::endl;
    std::cout << "Halo: " << h << std::endl;
    std::cout << "Alignment: " << storage_info_t::alignment_t::value << std::endl;
    std::cout << "Initial offset: " << si.get_initial_offset() << std::endl;
    std::cout << "Total size: " << si.size() << std::endl;
    std::cout << "Storage Size: " << si.template dim<0>() << " " << si.template dim<1>() << " " << si.template dim<2>() << std::endl;    
    std::cout << "Zero pos: " << si.index(0,0,0) << std::endl;
    std::cout << "Zero pos+halo: " << si.index(h,h,h) << std::endl;

    T* a = (T*)aligned_alloc(ALIGN, si.size()*sizeof(T));
    T* b = (T*)aligned_alloc(ALIGN, si.size()*sizeof(T));

    for(unsigned int i=0; i < isize+2*h; ++i) {
        for(unsigned int j=0; j < jsize+2*h; ++j) {
            for(unsigned int k=0; k < ksize+2*h; ++k) {
                a[si.index(i,j,k)] = si.index(i,j,k);
            }
        }
    }

    std::chrono::high_resolution_clock::time_point t1,t2;

    for(unsigned int t=0; t < tsteps; t++) {

        //----------------------------------------//
        //----------------  COPY  ----------------//
        //----------------------------------------//
        t1 = std::chrono::high_resolution_clock::now();
        copy(a, b, si, ksize, isize, jsize);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            timings[copy_st] += std::chrono::duration<double>(t2-t1).count();
        if(!t) {
            for(unsigned int i=h; i < isize; ++i) {
                for(unsigned int j=h; j < jsize; ++j) {
                    for(unsigned int k=h; k < ksize; ++k) {
                        if( b[si.index(i,j,k)] != a[si.index(i,j,k)] ) {
                            printf("Error in (%d,%d,%d) : %f %f\n", (int)i,(int)j,(int)k,b[si.index(i,j,k)], a[si.index(i,j,k)]);
                        }
                    }
                }
            }
        }
    
        //----------------------------------------//
        //---------------- COPYi1 ----------------//
        //----------------------------------------//
        t1 = std::chrono::high_resolution_clock::now();
        copyi1(a, b, si, ksize, isize, jsize);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            timings[copyi1_st] += std::chrono::duration<double>(t2-t1).count();
        if(!t) {
            for(int i=h; i < isize; ++i) {
                for(int j=h; j < jsize; ++j) {
                    for(int k=h; k < ksize; ++k) {
                        if( b[si.index(i,j,k)] != a[si.index(i+1,j,k)] ) {
                            printf("Error1 in (%d,%d,%d) : %f %f\n", (int)i,(int)j,(int)k,b[si.index(i,j,k)], a[si.index(i+1,j,k)]);
                        }
                    }
                }
            }
        }

        //----------------------------------------//
        //----------------  SUMi1 ----------------//
        //----------------------------------------//
        t1 = std::chrono::high_resolution_clock::now();
        sumi1(a, b, si, ksize, isize, jsize);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            timings[sumi1_st] += std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        if(!t) {
            for(int i=h; i < isize; ++i) {
                for(int j=h; j < jsize; ++j) {
                    for(int k=h; k < ksize; ++k) {
                        if( b[si.index(i,j,k)] != a[si.index(i,j,k)] + a[si.index(i+1,j,k)] ) {
                            printf("Error in (%d,%d,%d) : %f %f %f\n", (int)i,(int)j,(int)k,b[si.index(i,j,k)], a[si.index(i,j,k)], a[si.index(i+1,j,k)]);
                        }
                    }
                }
            }
        }

        //----------------------------------------//
        //----------------  SUMj1 ----------------//
        //----------------------------------------//
        t1 = std::chrono::high_resolution_clock::now();
        sumj1(a, b, si, ksize, isize, jsize);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            timings[sumj1_st] += std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        if(!t) {
            for(int i=h; i < isize; ++i) {
                for(int j=h; j < jsize; ++j) {
                    for(int k=h; k < ksize; ++k) {            
                        if( b[si.index(i,j,k)] != a[si.index(i,j,k)] + a[si.index(i,j+1,k)] ) {
                            printf("Error in (%d,%d,%d) : %f %f\n", (int)i,(int)j,(int)k,b[si.index(i,j,k)], a[si.index(i,j+1,k)]);
                        }
                    }
                }
            }
        }

        //----------------------------------------//
        //----------------  SUMk1 ----------------//
        //----------------------------------------//
        t1 = std::chrono::high_resolution_clock::now();
        sumk1(a, b, si, ksize, isize, jsize);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            timings[sumk1_st] += std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        if(!t) {
            for(int i=h; i < isize; ++i) {
                for(int j=h; j < jsize; ++j) {
                    for(int k=h; k < ksize; ++k) {            
                        if( b[si.index(i,j,k)] != a[si.index(i,j,k)] + a[si.index(i,j,k+1)] ) {                        
                            printf("Error in (%d,%d,%d) : %f %f %f\n", (int)i,(int)j,(int)k,b[si.index(i,j,k)], a[si.index(i,j,k)], a[si.index(i,j,k+1)]);
                        }
                    }
                }
            }
        }
       

        //----------------------------------------//
        //----------------  AVGi  ----------------//
        //----------------------------------------//
        t1 = std::chrono::high_resolution_clock::now();
        avgi(a, b, si, ksize, isize, jsize);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            timings[avgi_st] += std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        if(!t) {
            for(int i=h; i < isize; ++i) {
                for(int j=h; j < jsize; ++j) {
                    for(int k=h; k < ksize; ++k) {
                        if( b[si.index(i,j,k)] != a[si.index(i-1,j,k)] + a[si.index(i+1,j,k)] ) {
                            printf("Error in (%d,%d,%d) : %f %f %f\n", (int)i,(int)j,(int)k,b[si.index(i,j,k)], a[si.index(i-1,j,k)], a[si.index(i+1,j,k)]);
                        }
                    }
                }
            }
        }

        //----------------------------------------//
        //----------------  AVGj  ----------------//
        //----------------------------------------//
        t1 = std::chrono::high_resolution_clock::now();
        avgj(a, b, si, ksize, isize, jsize);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            timings[avgj_st] += std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        if(!t) {
            for(int i=h; i < isize; ++i) {
                for(int j=h; j < jsize; ++j) {
                    for(int k=h; k < ksize; ++k) {
                        if( b[si.index(i,j,k)] != a[si.index(i,j-1,k)] + a[si.index(i,j+1,k)] ) {
                            printf("Error in (%d,%d,%d) : %f %f %f\n", (int)i,(int)j,(int)k,b[si.index(i,j,k)], 
                                    a[si.index(i,j-1,k)], a[si.index(i,j+1,k)]);
                        }
                    }
                }
            }
        }

        //----------------------------------------//
        //----------------  AVGk  ----------------//
        //----------------------------------------//
        t1 = std::chrono::high_resolution_clock::now();
        avgk(a, b, si, ksize, isize, jsize);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            timings[avgk_st] += std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        if(!t) {
            for(int i=h; i < isize; ++i) {
                for(int j=h; j < jsize; ++j) {
                    for(int k=h; k < ksize; ++k) {
                        if( b[si.index(i,j,k)] != a[si.index(i,j,k-1)] + a[si.index(i,j,k+1)]) {
                            printf("Error in (%d,%d,%d) : %f %f %f\n", (int)i,(int)j,(int)k,b[si.index(i,j,k)],  
                                    a[si.index(i,j,k-1)], a[si.index(i,j,k+1)]);
                        }
                    }
                }
            }
        }
 
 
        //----------------------------------------//
        //----------------  LAP   ----------------//
        //----------------------------------------//
        t1 = std::chrono::high_resolution_clock::now();
        lap(a, b, si, ksize, isize, jsize);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            timings[lap_st] += std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        if(!t) {
            for(int i=h; i < isize; ++i) {
                for(int j=h; j < jsize; ++j) {
                    for(int k=h; k < ksize; ++k) {
                        if( b[si.index(i,j,k)] != a[si.index(i,j,k)] + a[si.index(i+1,j,k)] +
                               a[si.index(i-1,j,k)] + a[si.index(i,j+1,k)] + a[si.index(i,j-1,k)] ) {
                            auto res = a[si.index(i,j,k)] + a[si.index(i+1,j,k)] +
                                a[si.index(i-1,j,k)] + a[si.index(i,j+1,k)] + a[si.index(i,j-1,k)];
                            printf("Error in (%d,%d,%d) : %f %f\n", (int)i,(int)j,(int)k,b[si.index(i,j,k)], res);
                        }
                    }
                }
            }
        }
    }

}

