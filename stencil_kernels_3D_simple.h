template <typename T>
void copy(const T* __restrict__ a, T* __restrict__ b, const storage_info_t si,
          const unsigned isize, const unsigned jsize, const unsigned ksize) {
    if (si.template stride<0>() != 1) {
        std::cerr << "Incompatible layout for this kernel, aborting..." << std::endl;
        std::abort();
    }

    constexpr int istride = 1;
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int k = h; k < ksize + h; ++k) {
        for (int j = h; j < jsize + h; ++j) {
            int index = si.index(h, j, k);
            #pragma vector nontemporal
            #pragma omp simd
            for (int i = h; i < isize + h; ++i) {
                b[index] = a[index];
                index += istride;
            }
        }
    }
}

template <typename T>
void copyi1(const T* __restrict__ a, T* __restrict__ b, const storage_info_t si,
          const unsigned isize, const unsigned jsize, const unsigned ksize) {
    if (si.template stride<0>() != 1) {
        std::cerr << "Incompatible layout for this kernel, aborting..." << std::endl;
        std::abort();
    }

    constexpr int istride = 1;
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int k = h; k < ksize + h; ++k) {
        for (int j = h; j < jsize + h; ++j) {
            int index = si.index(h, j, k);
            #pragma vector nontemporal
            #pragma omp simd
            for (int i = h; i < isize + h; ++i) {
                b[index] = a[index + istride];
                index += istride;
            }
        }
    }
}

template <typename T>
void sumi1(const T* __restrict__ a, T* __restrict__ b, const storage_info_t si,
          const unsigned isize, const unsigned jsize, const unsigned ksize) {
    if (si.template stride<0>() != 1) {
        std::cerr << "Incompatible layout for this kernel, aborting..." << std::endl;
        std::abort();
    }

    constexpr int istride = 1;
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int k = h; k < ksize + h; ++k) {
        for (int j = h; j < jsize + h; ++j) {
            int index = si.index(h, j, k);
            #pragma vector nontemporal
            #pragma omp simd
            for (int i = h; i < isize + h; ++i) {
                b[index] = a[index] + a[index + istride];
                index += istride;
            }
        }
    }
}

template <typename T>
void avgi(const T* __restrict__ a, T* __restrict__ b, const storage_info_t si,
          const unsigned isize, const unsigned jsize, const unsigned ksize) {
    if (si.template stride<0>() != 1) {
        std::cerr << "Incompatible layout for this kernel, aborting..." << std::endl;
        std::abort();
    }

    constexpr int istride = 1;
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int k = h; k < ksize + h; ++k) {
        for (int j = h; j < jsize + h; ++j) {
            int index = si.index(h, j, k);
            #pragma vector nontemporal
            #pragma omp simd
            for (int i = h; i < isize + h; ++i) {
                b[index] = a[index - 1] + a[index + 1];
                index += istride;
            }
        }
    }
}

template <typename T>
void copyj1(const T* __restrict__ a, T* __restrict__ b, const storage_info_t si,
          const unsigned isize, const unsigned jsize, const unsigned ksize) {
    if (si.template stride<0>() != 1) {
        std::cerr << "Incompatible layout for this kernel, aborting..." << std::endl;
        std::abort();
    }

    constexpr int istride = 1;
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int k = h; k < ksize + h; ++k) {
        for (int j = h; j < jsize + h; ++j) {
            int index = si.index(h, j, k);
            #pragma vector nontemporal
            #pragma omp simd
            for (int i = h; i < isize + h; ++i) {
                b[index] = a[index + jstride];
                index += istride;
            }
        }
    }
}

template <typename T>
void sumj1(const T* __restrict__ a, T* __restrict__ b, const storage_info_t si,
          const unsigned isize, const unsigned jsize, const unsigned ksize) {
    if (si.template stride<0>() != 1) {
        std::cerr << "Incompatible layout for this kernel, aborting..." << std::endl;
        std::abort();
    }

    constexpr int istride = 1;
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int k = h; k < ksize + h; ++k) {
        for (int j = h; j < jsize + h; ++j) {
            int index = si.index(h, j, k);
            #pragma vector nontemporal
            #pragma omp simd
            for (int i = h; i < isize + h; ++i) {
                b[index] = a[index] + a[index + jstride];
                index += istride;
            }
        }
    }
}

template <typename T>
void avgj(const T* __restrict__ a, T* __restrict__ b, const storage_info_t si,
          const unsigned isize, const unsigned jsize, const unsigned ksize) {
    if (si.template stride<0>() != 1) {
        std::cerr << "Incompatible layout for this kernel, aborting..." << std::endl;
        std::abort();
    }

    constexpr int istride = 1;
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int k = h; k < ksize + h; ++k) {
        for (int j = h; j < jsize + h; ++j) {
            int index = si.index(h, j, k);
            #pragma vector nontemporal
            #pragma omp simd
            for (int i = h; i < isize + h; ++i) {
                b[index] = a[index - jstride] + a[index + jstride];
                index += istride;
            }
        }
    }
}

template <typename T>
void copyk1(const T* __restrict__ a, T* __restrict__ b, const storage_info_t si,
          const unsigned isize, const unsigned jsize, const unsigned ksize) {
    if (si.template stride<0>() != 1) {
        std::cerr << "Incompatible layout for this kernel, aborting..." << std::endl;
        std::abort();
    }

    constexpr int istride = 1;
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int k = h; k < ksize + h; ++k) {
        for (int j = h; j < jsize + h; ++j) {
            int index = si.index(h, j, k);
            #pragma vector nontemporal
            #pragma omp simd
            for (int i = h; i < isize + h; ++i) {
                b[index] = a[index + kstride];
                index += istride;
            }
        }
    }
}

template <typename T>
void sumk1(const T* __restrict__ a, T* __restrict__ b, const storage_info_t si,
          const unsigned isize, const unsigned jsize, const unsigned ksize) {
    if (si.template stride<0>() != 1) {
        std::cerr << "Incompatible layout for this kernel, aborting..." << std::endl;
        std::abort();
    }

    constexpr int istride = 1;
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int k = h; k < ksize + h; ++k) {
        for (int j = h; j < jsize + h; ++j) {
            int index = si.index(h, j, k);
            #pragma vector nontemporal
            #pragma omp simd
            for (int i = h; i < isize + h; ++i) {
                b[index] = a[index] + a[index + kstride];
                index += istride;
            }
        }
    }
}

template <typename T>
void avgk(const T* __restrict__ a, T* __restrict__ b, const storage_info_t si,
          const unsigned isize, const unsigned jsize, const unsigned ksize) {
    if (si.template stride<0>() != 1) {
        std::cerr << "Incompatible layout for this kernel, aborting..." << std::endl;
        std::abort();
    }

    constexpr int istride = 1;
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int k = h; k < ksize + h; ++k) {
        for (int j = h; j < jsize + h; ++j) {
            int index = si.index(h, j, k);
            #pragma vector nontemporal
            #pragma omp simd
            for (int i = h; i < isize + h; ++i) {
                b[index] = a[index - kstride] + a[index + kstride];
                index += istride;
            }
        }
    }
}

template <typename T>
void lap(const T* __restrict__ a, T* __restrict__ b, const storage_info_t si,
         const unsigned isize, const unsigned jsize, const unsigned ksize) {
    if (si.template stride<0>() != 1) {
        std::cerr << "Incompatible layout for this kernel, aborting..." << std::endl;
        std::abort();
    }

    constexpr int istride = 1;
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int k = h; k < ksize + h; ++k) {
        for (int j = h; j < jsize + h; ++j) {
            int index = si.index(h, j, k);
            #pragma vector nontemporal
            #pragma omp simd
            for (int i = h; i < isize + h; ++i) {
                b[index] = a[index] + a[index - istride] + a[index + istride] + a[index - jstride] + a[index + jstride];
                index += istride;
            }
        }
    }
}
