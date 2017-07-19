template<typename T>
void copy( T const* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned isize, const unsigned jsize, const unsigned ksize) {
    const int begin = si.index(h, h, h);
    const int end = si.index(h + isize, h + jsize, h + ksize);

    #pragma omp parallel for
    for (int i = begin; i < end; ++i) {
        b[i] = a[i];
    }
}

template<typename T>
void copyi1( T const* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned isize, const unsigned jsize, const unsigned ksize) {
    const int begin = si.index(h, h, h);
    const int end = si.index(h + isize, h + jsize, h + ksize);
    const int istride = si.template stride<0>();

    #pragma omp parallel for
    for (int i = begin; i < end; ++i) {
        b[i] = a[i+istride];
    }
}

template<typename T>
void sumi1( T const* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned isize, const unsigned jsize, const unsigned ksize) {
    const int begin = si.index(h, h, h);
    const int end = si.index(h + isize, h + jsize, h + ksize);
    const int istride = si.template stride<0>();

    #pragma omp parallel for
    for (int i = begin; i < end; ++i) {
        b[i] = a[i] + a[i+istride];
    }
}

template<typename T>
void avgi( T const* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned isize, const unsigned jsize, const unsigned ksize) {
    const int begin = si.index(h, h, h);
    const int end = si.index(h + isize, h + jsize, h + ksize);
    const int istride = si.template stride<0>();

    #pragma omp parallel for
    for (int i = begin; i < end; ++i) {
        b[i] = a[i-istride] + a[i+istride];
    }
}

template<typename T>
void sumj1( T const* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned isize, const unsigned jsize, const unsigned ksize) {
    const int begin = si.index(h, h, h);
    const int end = si.index(h + isize, h + jsize, h + ksize);
    const int jstride = si.template stride<1>();

    #pragma omp parallel for
    for (int i = begin; i < end; ++i) {
        b[i] = a[i] + a[i+jstride];
    }
}

template<typename T>
void avgj( T const* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned isize, const unsigned jsize, const unsigned ksize) {
    const int begin = si.index(h, h, h);
    const int end = si.index(h + isize, h + jsize, h + ksize);
    const int jstride = si.template stride<1>();

    #pragma omp parallel for
    for (int i = begin; i < end; ++i) {
        b[i] = a[i-jstride] + a[i+jstride];
    }
}

template<typename T>
void sumk1( T const* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned isize, const unsigned jsize, const unsigned ksize) {
    const int begin = si.index(h, h, h);
    const int end = si.index(h + isize, h + jsize, h + ksize);
    const int kstride = si.template stride<2>();

    #pragma omp parallel for
    for (int i = begin; i < end; ++i) {
        b[i] = a[i] + a[i+kstride];
    } 
}

template<typename T>
void avgk( T const* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned isize, const unsigned jsize, const unsigned ksize) {
    const int begin = si.index(h, h, h);
    const int end = si.index(h + isize, h + jsize, h + ksize);
    const int kstride = si.template stride<2>();

    #pragma omp parallel for
    for (int i = begin; i < end; ++i) {
        b[i] = a[i-kstride] + a[i+kstride];
    } 
}


template<typename T>
void lap( T const* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned isize, const unsigned jsize, const unsigned ksize) {
    const int begin = si.index(h, h, h);
    const int end = si.index(h + isize, h + jsize, h + ksize);
    const int istride = si.template stride<0>();
    const int jstride = si.template stride<1>();

    #pragma omp parallel for
    for (int i = begin; i < end; ++i) {
        b[i] = a[i] + a[i-istride] + a[i+istride] + a[i-jstride] + a[i+jstride];
    }
}
