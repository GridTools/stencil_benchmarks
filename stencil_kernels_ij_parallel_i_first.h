template<typename T>
void copy( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    if (si.template stride<0>() != 1) {
        std::cerr << "Incompatible layout for this kernel, aborting..." << std::endl;
        std::abort();
    }

    const int NBI = (isize + BLOCKSIZEX - 1) / BLOCKSIZEX;
    const int NBJ = (jsize + BLOCKSIZEY - 1) / BLOCKSIZEY;
    constexpr int istride = 1;
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX + h;
            int start_index_j = bj*BLOCKSIZEY + h;
            int index = si.index(start_index_i,start_index_j,h);
            const int iblocksize = start_index_i + BLOCKSIZEX <= isize + h ? BLOCKSIZEX : isize + h - start_index_i;
            const int jblocksize = start_index_j + BLOCKSIZEY <= jsize + h ? BLOCKSIZEY : jsize + h - start_index_j;

            for(int k=0; k < ksize; ++k) {
                for(int j=0; j<jblocksize; ++j) {
                    #pragma vector nontemporal
                    #pragma omp simd
                    for(int i=0; i<iblocksize; ++i) {
                        b[index] = a[index];
                        index += istride;
                    }
                    index+=(jstride-istride*iblocksize);
                }
                index+=(kstride-jstride*jblocksize);
            }
        }
    }
}

template<typename T>
void copyi1( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    if (si.template stride<0>() != 1) {
        std::cerr << "Incompatible layout for this kernel, aborting..." << std::endl;
        std::abort();
    }

    const int NBI = (isize + BLOCKSIZEX - 1) / BLOCKSIZEX;
    const int NBJ = (jsize + BLOCKSIZEY - 1) / BLOCKSIZEY;
    constexpr int istride = 1;
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX + h;
            int start_index_j = bj*BLOCKSIZEY + h;
            int index = si.index(start_index_i,start_index_j,h);
            const int iblocksize = start_index_i + BLOCKSIZEX <= isize + h ? BLOCKSIZEX : isize + h - start_index_i;
            const int jblocksize = start_index_j + BLOCKSIZEY <= jsize + h ? BLOCKSIZEY : jsize + h - start_index_j;

            for(int k=0; k < ksize; ++k) {
                for(int j=0; j<jblocksize; ++j) {
                    #pragma vector nontemporal
                    #pragma omp simd
                    for(int i=0; i<iblocksize; ++i) {
                        b[index] = a[index + istride];
                        index += istride;
                    }
                    index+=(jstride-istride*iblocksize);
                }
                index+=(kstride-jstride*jblocksize);
            }
        }
    }
}

template<typename T>
void sumi1( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    if (si.template stride<0>() != 1) {
        std::cerr << "Incompatible layout for this kernel, aborting..." << std::endl;
        std::abort();
    }

    const int NBI = (isize + BLOCKSIZEX - 1) / BLOCKSIZEX;
    const int NBJ = (jsize + BLOCKSIZEY - 1) / BLOCKSIZEY;
    constexpr int istride = 1;
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX + h;
            int start_index_j = bj*BLOCKSIZEY + h;
            int index = si.index(start_index_i,start_index_j,h);
            const int iblocksize = start_index_i + BLOCKSIZEX <= isize + h ? BLOCKSIZEX : isize + h - start_index_i;
            const int jblocksize = start_index_j + BLOCKSIZEY <= jsize + h ? BLOCKSIZEY : jsize + h - start_index_j;

            for(int k=0; k < ksize; ++k) {
                for(int j=0; j<jblocksize; ++j) {
                    #pragma vector nontemporal
                    #pragma omp simd
                    for(int i=0; i<iblocksize; ++i) {
                        b[index] = a[index] + a[index + istride];
                        index += istride;
                    }
                    index+=(jstride-istride*iblocksize);
                }
                index+=(kstride-jstride*jblocksize);
            }
        }
    }
}

template<typename T>
void avgi( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    if (si.template stride<0>() != 1) {
        std::cerr << "Incompatible layout for this kernel, aborting..." << std::endl;
        std::abort();
    }

    const int NBI = (isize + BLOCKSIZEX - 1) / BLOCKSIZEX;
    const int NBJ = (jsize + BLOCKSIZEY - 1) / BLOCKSIZEY;
    constexpr int istride = 1;
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX + h;
            int start_index_j = bj*BLOCKSIZEY + h;
            int index = si.index(start_index_i,start_index_j,h);
            const int iblocksize = start_index_i + BLOCKSIZEX <= isize + h ? BLOCKSIZEX : isize + h - start_index_i;
            const int jblocksize = start_index_j + BLOCKSIZEY <= jsize + h ? BLOCKSIZEY : jsize + h - start_index_j;

            for(int k=0; k < ksize; ++k) {
                for(int j=0; j<jblocksize; ++j) {
                    #pragma vector nontemporal
                    #pragma omp simd
                    for(int i=0; i<iblocksize; ++i) {
                        b[index] = a[index - istride] + a[index + istride];
                        index += istride;
                    }
                    index+=(jstride-istride*iblocksize);
                }
                index+=(kstride-jstride*jblocksize);
            }
        }
    }
}

template<typename T>
void sumj1( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    if (si.template stride<0>() != 1) {
        std::cerr << "Incompatible layout for this kernel, aborting..." << std::endl;
        std::abort();
    }

    const int NBI = (isize + BLOCKSIZEX - 1) / BLOCKSIZEX;
    const int NBJ = (jsize + BLOCKSIZEY - 1) / BLOCKSIZEY;
    constexpr int istride = 1;
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX + h;
            int start_index_j = bj*BLOCKSIZEY + h;
            int index = si.index(start_index_i,start_index_j,h);
            const int iblocksize = start_index_i + BLOCKSIZEX <= isize + h ? BLOCKSIZEX : isize + h - start_index_i;
            const int jblocksize = start_index_j + BLOCKSIZEY <= jsize + h ? BLOCKSIZEY : jsize + h - start_index_j;

            for(int k=0; k < ksize; ++k) {
                for(int j=0; j<jblocksize; ++j) {
                    #pragma vector nontemporal
                    #pragma omp simd
                    for(int i=0; i<iblocksize; ++i) {
                        b[index] = a[index] + a[index + jstride];
                        index += istride;
                    }
                    index+=(jstride-istride*iblocksize);
                }
                index+=(kstride-jstride*jblocksize);
            }
        }
    }
}

template<typename T>
void avgj( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    if (si.template stride<0>() != 1) {
        std::cerr << "Incompatible layout for this kernel, aborting..." << std::endl;
        std::abort();
    }

    const int NBI = (isize + BLOCKSIZEX - 1) / BLOCKSIZEX;
    const int NBJ = (jsize + BLOCKSIZEY - 1) / BLOCKSIZEY;
    constexpr int istride = 1;
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX + h;
            int start_index_j = bj*BLOCKSIZEY + h;
            int index = si.index(start_index_i,start_index_j,h);
            const int iblocksize = start_index_i + BLOCKSIZEX <= isize + h ? BLOCKSIZEX : isize + h - start_index_i;
            const int jblocksize = start_index_j + BLOCKSIZEY <= jsize + h ? BLOCKSIZEY : jsize + h - start_index_j;

            for(int k=0; k < ksize; ++k) {
                for(int j=0; j<jblocksize; ++j) {
                    #pragma vector nontemporal
                    #pragma omp simd
                    for(int i=0; i<iblocksize; ++i) {
                        b[index] = a[index - jstride] + a[index + jstride];
                        index += istride;
                    }
                    index+=(jstride-istride*iblocksize);
                }
                index+=(kstride-jstride*jblocksize);
            }
        }
    }
}

template<typename T>
void sumk1( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    if (si.template stride<0>() != 1) {
        std::cerr << "Incompatible layout for this kernel, aborting..." << std::endl;
        std::abort();
    }

    const int NBI = (isize + BLOCKSIZEX - 1) / BLOCKSIZEX;
    const int NBJ = (jsize + BLOCKSIZEY - 1) / BLOCKSIZEY;
    constexpr int istride = 1;
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX + h;
            int start_index_j = bj*BLOCKSIZEY + h;
            int index = si.index(start_index_i,start_index_j,h);
            const int iblocksize = start_index_i + BLOCKSIZEX <= isize + h ? BLOCKSIZEX : isize + h - start_index_i;
            const int jblocksize = start_index_j + BLOCKSIZEY <= jsize + h ? BLOCKSIZEY : jsize + h - start_index_j;

            for(int k=0; k < ksize; ++k) {
                for(int j=0; j<jblocksize; ++j) {
                    #pragma vector nontemporal
                    #pragma omp simd
                    for(int i=0; i<iblocksize; ++i) {
                        b[index] = a[index] + a[index + kstride];
                        index += istride;
                    }
                    index+=(jstride-istride*iblocksize);
                }
                index+=(kstride-jstride*jblocksize);
            }
        }
    }
}

template<typename T>
void avgk( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    if (si.template stride<0>() != 1) {
        std::cerr << "Incompatible layout for this kernel, aborting..." << std::endl;
        std::abort();
    }

    const int NBI = (isize + BLOCKSIZEX - 1) / BLOCKSIZEX;
    const int NBJ = (jsize + BLOCKSIZEY - 1) / BLOCKSIZEY;
    constexpr int istride = 1;
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX + h;
            int start_index_j = bj*BLOCKSIZEY + h;
            int index = si.index(start_index_i,start_index_j,h);
            const int iblocksize = start_index_i + BLOCKSIZEX <= isize + h ? BLOCKSIZEX : isize + h - start_index_i;
            const int jblocksize = start_index_j + BLOCKSIZEY <= jsize + h ? BLOCKSIZEY : jsize + h - start_index_j;

            for(int k=0; k < ksize; ++k) {
                for(int j=0; j<jblocksize; ++j) {
                    #pragma vector nontemporal
                    #pragma omp simd
                    for(int i=0; i<iblocksize; ++i) {
                        b[index] = a[index - kstride] + a[index + kstride];
                        index += istride;
                    }
                    index+=(jstride-istride*iblocksize);
                }
                index+=(kstride-jstride*jblocksize);
            }
        }
    }
}


template<typename T>
void lap( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    if (si.template stride<0>() != 1) {
        std::cerr << "Incompatible layout for this kernel, aborting..." << std::endl;
        std::abort();
    }

    const int NBI = (isize + BLOCKSIZEX - 1) / BLOCKSIZEX;
    const int NBJ = (jsize + BLOCKSIZEY - 1) / BLOCKSIZEY;
    constexpr int istride = 1;
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX + h;
            int start_index_j = bj*BLOCKSIZEY + h;
            int index = si.index(start_index_i,start_index_j,h);
            const int iblocksize = start_index_i + BLOCKSIZEX <= isize + h ? BLOCKSIZEX : isize + h - start_index_i;
            const int jblocksize = start_index_j + BLOCKSIZEY <= jsize + h ? BLOCKSIZEY : jsize + h - start_index_j;

            for(int k=0; k < ksize; ++k) {
                for(int j=0; j<jblocksize; ++j) {
                    #pragma vector nontemporal
                    #pragma omp simd
                    for(int i=0; i<iblocksize; ++i) {
                        b[index] = a[index] + a[index - istride] + a[index + istride] + a[index - jstride] + a[index + jstride];
                        index += istride;
                    }
                    index+=(jstride-istride*iblocksize);
                }
                index+=(kstride-jstride*jblocksize);
            }
        }
    }
}
