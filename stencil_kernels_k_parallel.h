
template<typename T>
void copy( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    const int istride = si.template stride<0>();

    #pragma omp parallel for
    for(int k=h; k<ksize+h; ++k) {
        int index = si.index(h,h,k);
        for(int j=h; j<jsize+h; ++j) {
            for(int i=h; i<isize+h; ++i) {
                b[index] = a[index];
                ++index;
            }
            index+=(jstride-isize);
        }
    }
/*
    #pragma omp parallel for
    for(int k=h; k<ksize+h; ++k) {
        for(int j=h; j<jsize+h; ++j) {
            for(int i=h; i<isize+h; ++i) {
                b[si.index(i,j,k)] = a[si.index(i,j,k)];
            }
        }
    }*/
}

template<typename T>
void copyi1( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    const int kstride = si.template stride<2>();
    const int jstride = si.template stride<1>();
    const int istride = si.template stride<0>();
    
    #pragma omp parallel for
    for(int k=h; k<ksize+h; ++k) {
        int index = si.index(h,h,k);
        for(int j=h; j<jsize+h; ++j) {
            for(int i=h; i<isize+h; ++i) {
                b[index] = a[index+istride];
                index += istride;
            }
            index += jstride - isize * istride;
        }
    }
}

template<typename T>
void sumi1( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    const int kstride = si.template stride<2>();
    const int jstride = si.template stride<1>();
    const int istride = si.template stride<0>();
    
    #pragma omp parallel for
    for(int k=h; k<ksize+h; ++k) {
        int index = si.index(h,h,k);
        for(int j=h; j<jsize+h; ++j) {
            for(int i=h; i<isize+h; ++i) {
                b[index] = a[index] + a[index+istride];
                index += istride;
            }
            index += jstride - isize * istride;
        }
    }
}

template<typename T>
void avgi( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    const int kstride = si.template stride<2>();
    const int jstride = si.template stride<1>();
    const int istride = si.template stride<0>();
    
    #pragma omp parallel for
    for(int k=h; k<ksize+h; ++k) {
        int index = si.index(h,h,k);
        for(int j=h; j<jsize+h; ++j) {
            for(int i=h; i<isize+h; ++i) {
                b[index] = a[index-istride] + a[index+istride];
                index += istride;
            }
            index += jstride - isize * istride;
        }
    }
}

template<typename T>
void sumj1( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    const int kstride = si.template stride<2>();
    const int jstride = si.template stride<1>();
    const int istride = si.template stride<0>();
    
    #pragma omp parallel for
    for(int k=h; k<ksize+h; ++k) {
        int index = si.index(h,h,k);
        for(int j=h; j<jsize+h; ++j) {
            for(int i=h; i<isize+h; ++i) {
                b[index] = a[index] + a[index+jstride];
                index += istride;
            }
            index += jstride - isize * istride;
        }
    }
}

template<typename T>
void avgj( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    const int kstride = si.template stride<2>();
    const int jstride = si.template stride<1>();
    const int istride = si.template stride<0>();
    
    #pragma omp parallel for
    for(int k=h; k<ksize+h; ++k) {
        int index = si.index(h,h,k);
        for(int j=h; j<jsize+h; ++j) {
            for(int i=h; i<isize+h; ++i) {
                b[index] = a[index-jstride] + a[index+jstride];
                index += istride;
            }
            index += jstride - isize * istride;
        }
    }
}

template<typename T>
void sumk1( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    const int kstride = si.template stride<2>();
    const int jstride = si.template stride<1>();
    const int istride = si.template stride<0>();

    
    #pragma omp parallel for
    for(int k=h; k<ksize+h; ++k) {
        int index = si.index(h,h,k);
        for(int j=h; j<jsize+h; ++j) {
            for(int i=h; i<isize+h; ++i) {
                b[index] = a[index] + a[index+kstride];
                index += istride;
            }
            index += jstride - isize * istride;
        }
    }
}

template<typename T>
void avgk( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    const int kstride = si.template stride<2>();
    const int jstride = si.template stride<1>();
    const int istride = si.template stride<0>();

    
    #pragma omp parallel for
    for(int k=h; k<ksize+h; ++k) {
        int index = si.index(h,h,k);
        for(int j=h; j<jsize+h; ++j) {
            for(int i=h; i<isize+h; ++i) {
                b[index] = a[index-kstride] + a[index+kstride];
                index += istride;
            }
            index += jstride - isize * istride;
        }
    }
}


template<typename T>
void lap( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    const int kstride = si.template stride<2>();
    const int jstride = si.template stride<1>();
    const int istride = si.template stride<0>();

    #pragma omp parallel for
    for(int k=h; k<ksize+h; ++k) {
        int index = si.index(h,h,k);
        for(int j=h; j<jsize+h; ++j) {
            for(int i=h; i<isize+h; ++i) {
                b[index] = a[index] + a[index+1] + a[index-1] + a[index+jstride] + a[index-jstride];
                ++index;
            }
            index+=(jstride-isize);
        }
    }
/*
    #pragma omp parallel for
    for(int k=h; k<ksize+h; ++k) {
        for(int j=h; j<jsize+h; ++j) {
            for(int i=h; i<isize+h; ++i) {
                b[si.index(i,j,k)] = a[si.index(i,j,k)] + a[si.index(i-1,j,k)] + a[si.index(i+1,j,k)]
                    + a[si.index(i,j+1,k)] + a[si.index(i,j-1,k)];
            }
        }
    }*/
}
