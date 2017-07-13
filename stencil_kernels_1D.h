#include <omp.h>
#include <boost/align/is_aligned.hpp>

#ifdef FLAT_MODE
#include <hbwmalloc.h>
#endif

#include <storage/storage-facility.hpp>

#include "tools.h"

constexpr int h = 2;
typedef gridtools::storage_traits<gridtools::enumtype::Host> storage_tr;
typedef storage_tr::custom_layout_storage_info_t<0, gridtools::layout_map< LAYOUT >, gridtools::halo< h,h,h >, gridtools::alignment< ALIGN > > storage_info_t;

template<typename T>
void copy( T const* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int begin, const unsigned int end) {
    #pragma omp parallel for
    for (int i = begin; i < end; ++i) {
        b[i] = a[i];
    }
}

template<typename T>
void copyi1( T const* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int begin, const unsigned int end) {
    const int istride = si.template stride<0>();

    #pragma omp parallel for
    for (int i = begin; i < end; ++i) {
        b[i] = a[i+istride];
    }
}

template<typename T>
void sumi1( T const* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int begin, const unsigned int end) {
    const int istride = si.template stride<0>();

    #pragma omp parallel for
    for (int i = begin; i < end; ++i) {
        b[i] = a[i] + a[i+istride];
    }
}

template<typename T>
void avgi( T const* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int begin, const unsigned int end) {
    const int istride = si.template stride<0>();

    #pragma omp parallel for
    for (int i = begin; i < end; ++i) {
        b[i] = a[i-istride] + a[i+istride];
    }
}

template<typename T>
void sumj1( T const* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int begin, const unsigned int end) {
    const int jstride = si.template stride<1>();

    #pragma omp parallel for
    for (int i = begin; i < end; ++i) {
        b[i] = a[i] + a[i+jstride];
    }
}

template<typename T>
void avgj( T const* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int begin, const unsigned int end) {
    const int jstride = si.template stride<1>();

    #pragma omp parallel for
    for (int i = begin; i < end; ++i) {
        b[i] = a[i-jstride] + a[i+jstride];
    }
}

template<typename T>
void sumk1( T const* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int begin, const unsigned int end) {
    const int kstride = si.template stride<2>();

    #pragma omp parallel for
    for (int i = begin; i < end; ++i) {
        b[i] = a[i] + a[i+kstride];
    } 
}

template<typename T>
void avgk( T const* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int begin, const unsigned int end) {
    const int kstride = si.template stride<2>();

    #pragma omp parallel for
    for (int i = begin; i < end; ++i) {
        b[i] = a[i-kstride] + a[i+kstride];
    } 
}


template<typename T>
void lap( T const* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int begin, const unsigned int end) {
    const int istride = si.template stride<0>();
    const int jstride = si.template stride<1>();

    #pragma omp parallel for
    for (int i = begin; i < end; ++i) {
        b[i] = a[i] + a[i-istride] + a[i+istride] + a[i-jstride] + a[i+jstride];
    }
}


template<typename T>
void launch( timing& times, const unsigned int is, const unsigned int js, const unsigned ks, const unsigned tsteps, const unsigned warmup_step ) {
    int isize = is-2*h;
    int jsize = js-2*h;
    int ksize = ks-2*h;
    
    const storage_info_t si(isize, jsize, ksize);

    std::cout << "Size: " << isize << " " << jsize << " " << ksize << std::endl;
    std::cout << "Storage size (including halo+padding): " << si.template dim<0>() << " " << si.template dim<1>() << " " << si.template dim<2>() << std::endl;
    std::cout << "Halo: " << h << std::endl;
    std::cout << "Alignment: " << storage_info_t::alignment_t::value << std::endl;
    std::cout << "Initial offset: " << si.get_initial_offset() << std::endl;
    std::cout << "Total size: " << si.size() << std::endl;
    std::cout << "Storage Size: " << si.template dim<0>() << " " << si.template dim<1>() << " " << si.template dim<2>() << std::endl;    
    std::cout << "Zero pos: " << si.index(0,0,0) << std::endl;
    std::cout << "Zero pos+halo: " << si.index(h,h,h) << std::endl;

#ifdef CACHE_MODE
    T* a = (T*)aligned_alloc(ALIGN*sizeof(T), si.size()*sizeof(T));
    T* b = (T*)aligned_alloc(ALIGN*sizeof(T), si.size()*sizeof(T));
#elif FLAT_MODE
    T* a = (T*) hbw_malloc(si.size()*sizeof(T));
    T* b = (T*) hbw_malloc(si.size()*sizeof(T));
#else
    static_assert(false, "Please define either FLAT_MODE or CACHE_MODE.");
#endif

    //check if aligned to ALIGN*sizeof(T) bytes
    bool is_align = boost::alignment::is_aligned(a+si.index(h,h,h), ALIGN*sizeof(T));
    is_align &= boost::alignment::is_aligned(b+si.index(h,h,h), ALIGN*sizeof(T));
    std::cout << "Alignment checks:\n";
    std::cout << "\t" << a << " --> " << boost::alignment::is_aligned(a, ALIGN*sizeof(T)) << std::endl;
    std::cout << "\t" << a+si.index(h,h,h) << " --> " << boost::alignment::is_aligned(a+si.index(h,h,h), ALIGN*sizeof(T)) << std::endl;
    std::cout << "\t" << b << " --> " << boost::alignment::is_aligned(b, ALIGN*sizeof(T)) << std::endl;
    std::cout << "\t" << b+si.index(h,h,h) << " --> " << boost::alignment::is_aligned(b+si.index(h,h,h), ALIGN*sizeof(T)) << std::endl;
    if (!is_align) {
        std::cout << "first data point is not aligned...\n";
        abort();
    }

    for(unsigned int i=0; i < isize+2*h; ++i) {
        for(unsigned int j=0; j < jsize+2*h; ++j) {
            for(unsigned int k=0; k < ksize+2*h; ++k) {
                a[si.index(i,j,k)] = si.index(i,j,k);
            }
        }
    }

    int begin = si.index(h,h,h);
    int end = si.index(isize+h,jsize+h,ksize+h);

    std::chrono::high_resolution_clock::time_point t1,t2;
    cache_flusher c(64, isize*1.5);

    for(unsigned int t=0; t < tsteps; t++) {

        //----------------------------------------//
        //----------------  COPY  ----------------//
        //----------------------------------------//
        c.flush();
        t1 = std::chrono::high_resolution_clock::now();
        copy(a, b, si, begin, end);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            times.insert("copy", std::chrono::duration<double>(t2-t1).count());
        if(!t) {
            for(unsigned int i=h; i < isize+h; ++i) {
                for(unsigned int j=h; j < jsize+h; ++j) {
                    for(unsigned int k=h; k < ksize+h; ++k) {
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
        c.flush();
        t1 = std::chrono::high_resolution_clock::now();
        copyi1(a, b, si, begin, end);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            times.insert("copyi1", std::chrono::duration<double>(t2-t1).count());
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
        c.flush();
        t1 = std::chrono::high_resolution_clock::now();
        sumi1(a, b, si, begin, end);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            times.insert("sumi1", std::chrono::duration<double>(t2-t1).count());

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
        c.flush();
        t1 = std::chrono::high_resolution_clock::now();
        sumj1(a, b, si, begin, end);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            times.insert("sumj1", std::chrono::duration<double>(t2-t1).count());

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
        c.flush();
        t1 = std::chrono::high_resolution_clock::now();
        sumk1(a, b, si, begin, end);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            times.insert("sumk1", std::chrono::duration<double>(t2-t1).count());

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
        c.flush();
        t1 = std::chrono::high_resolution_clock::now();
        avgi(a, b, si, begin, end);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            times.insert("avgi", std::chrono::duration<double>(t2-t1).count());

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
        c.flush();
        t1 = std::chrono::high_resolution_clock::now();
        avgj(a, b, si, begin, end);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            times.insert("avgj", std::chrono::duration<double>(t2-t1).count());

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
        c.flush();
        t1 = std::chrono::high_resolution_clock::now();
        avgk(a, b, si, begin, end);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            times.insert("avgk", std::chrono::duration<double>(t2-t1).count());

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
        c.flush();
        t1 = std::chrono::high_resolution_clock::now();
        lap(a, b, si, begin, end);
        t2 = std::chrono::high_resolution_clock::now();
        if(t > warmup_step)
            times.insert("lap", std::chrono::duration<double>(t2-t1).count());

        if(!t) {
            for(int i=h; i < isize; ++i) {
                for(int j=h; j < jsize; ++j) {
                    for(int k=h; k < ksize; ++k) {
                        if( b[si.index(i,j,k)] != a[si.index(i,j,k)] + a[si.index(i-1,j,k)] +
                               a[si.index(i+1,j,k)] + a[si.index(i,j-1,k)] + a[si.index(i,j+1,k)] ) {
                            auto res =a[si.index(i,j,k)] + a[si.index(i-1,j,k)] +
                               a[si.index(i+1,j,k)] + a[si.index(i,j-1,k)] + a[si.index(i,j+1,k)];
                            printf("Error in (%d,%d,%d) : %f %f\n", (int)i,(int)j,(int)k,b[si.index(i,j,k)], res);
                        }
                    }
                }
            }
        }
    }

#ifdef CACHE_MODE
    free(a);
    free(b);
#elif FLAT_MODE
    hbw_free(a);
    hbw_free(b);
#endif

}

