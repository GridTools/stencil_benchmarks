template<typename T>
void copy( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    const int NBI = isize/BLOCKSIZEX;
    const int NBJ = jsize/BLOCKSIZEY;
    const int istride = si.template stride<0>();
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX;
            int start_index_j = bj*BLOCKSIZEY;
            int index = si.index(start_index_i+h,start_index_j+h,h);
            for(int k=0; k < ksize; ++k) {
                for(int j=0; j<BLOCKSIZEY; ++j) {
                    for(int i=0; i<BLOCKSIZEX; ++i) {
                        b[index] = a[index];
                        ++index;
                    }
                    index+=(jstride-BLOCKSIZEX);
                }
                index+=(kstride-jstride*BLOCKSIZEY);
            }
        }
    }
}

template<typename T>
void copyi1( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    const int NBI = isize/BLOCKSIZEX;
    const int NBJ = jsize/BLOCKSIZEY;
    const int istride = si.template stride<0>();
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX;
            int start_index_j = bj*BLOCKSIZEY;
            int index = si.index(start_index_i+h,start_index_j+h,h);
            for(int k=0; k < ksize; ++k) {
                for(int j=0; j<BLOCKSIZEY; ++j) {
                    for(int i=0; i<BLOCKSIZEX; ++i) {
                        b[index] = a[index+1];
                        ++index;
                    }
                    index+=(jstride-BLOCKSIZEX);
                }
                index+=(kstride-jstride*BLOCKSIZEY);
            }
        }
    }
}

template<typename T>
void sumi1( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    const int NBI = isize/BLOCKSIZEX;
    const int NBJ = jsize/BLOCKSIZEY;
    const int istride = si.template stride<0>();
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX;
            int start_index_j = bj*BLOCKSIZEY;
            int index = si.index(start_index_i+h,start_index_j+h,h);
            for(int k=0; k < ksize; ++k) {
                for(int j=0; j<BLOCKSIZEY; ++j) {
                    for(int i=0; i<BLOCKSIZEX; ++i) {
                        b[index] = a[index] + a[index+1];
                        ++index;
                    }
                    index+=(jstride-BLOCKSIZEX);
                }
                index+=(kstride-jstride*BLOCKSIZEY);
            }
        }
    }
}

template<typename T>
void avgi( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    const int NBI = isize/BLOCKSIZEX;
    const int NBJ = jsize/BLOCKSIZEY;
    const int istride = si.template stride<0>();
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();


    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX;
            int start_index_j = bj*BLOCKSIZEY;
            int index = si.index(start_index_i+h,start_index_j+h,h);
            for(int k=0; k < ksize; ++k) {
                for(int j=0; j<BLOCKSIZEY; ++j) {
                    for(int i=0; i<BLOCKSIZEX; ++i) {
                        b[index] = a[index-1] + a[index+1];
                        ++index;
                    }
                    index+=(jstride-BLOCKSIZEX);
                }
                index+=(kstride-jstride*BLOCKSIZEY);
            }
        }
    }
}

template<typename T>
void sumj1( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    const int NBI = isize/BLOCKSIZEX;
    const int NBJ = jsize/BLOCKSIZEY;
    const int istride = si.template stride<0>();
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX;
            int start_index_j = bj*BLOCKSIZEY;
            int index = si.index(start_index_i+h,start_index_j+h,h);
            for(int k=0; k < ksize; ++k) {
                for(int j=0; j<BLOCKSIZEY; ++j) {
                    for(int i=0; i<BLOCKSIZEX; ++i) {
                        b[index] = a[index] + a[index+jstride];
                        ++index;
                    }
                    index+=(jstride-BLOCKSIZEX);
                }
                index+=(kstride-jstride*BLOCKSIZEY);
            }
        }
    }
}

template<typename T>
void avgj( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    const int NBI = isize/BLOCKSIZEX;
    const int NBJ = jsize/BLOCKSIZEY;
    const int istride = si.template stride<0>();
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX;
            int start_index_j = bj*BLOCKSIZEY;
            int index = si.index(start_index_i+h,start_index_j+h,h);
            for(int k=0; k < ksize; ++k) {
                for(int j=0; j<BLOCKSIZEY; ++j) {
                    for(int i=0; i<BLOCKSIZEX; ++i) {
                        b[index] = a[index-jstride] + a[index+jstride];
                        ++index;
                    }
                    index+=(jstride-BLOCKSIZEX);
                }
                index+=(kstride-jstride*BLOCKSIZEY);
            }
        }
    }
}

template<typename T>
void sumk1( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    const int NBI = isize/BLOCKSIZEX;
    const int NBJ = jsize/BLOCKSIZEY;
    const int istride = si.template stride<0>();
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX;
            int start_index_j = bj*BLOCKSIZEY;
            int index = si.index(start_index_i+h,start_index_j+h,h);
            for(int k=0; k < ksize; ++k) {
                for(int j=0; j<BLOCKSIZEY; ++j) {
                    for(int i=0; i<BLOCKSIZEX; ++i) {
                        b[index] = a[index] + a[index+kstride];
                        ++index;
                    }
                    index+=(jstride-BLOCKSIZEX);
                }
                index+=(kstride-jstride*BLOCKSIZEY);
            }
        }
    }
}

template<typename T>
void avgk( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    const int NBI = isize/BLOCKSIZEX;
    const int NBJ = jsize/BLOCKSIZEY;
    const int istride = si.template stride<0>();
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX;
            int start_index_j = bj*BLOCKSIZEY;
            int index = si.index(start_index_i+h,start_index_j+h,h);
            for(int k=0; k < ksize; ++k) {
                for(int j=0; j<BLOCKSIZEY; ++j) {
                    for(int i=0; i<BLOCKSIZEX; ++i) {
                        b[index] = a[index-kstride] + a[index+kstride];
                        ++index;
                    }
                    index+=(jstride-BLOCKSIZEX);
                }
                index+=(kstride-jstride*BLOCKSIZEY);
            }
        }
    }
}


template<typename T>
void lap( T* __restrict__ a,  T* __restrict__ b, const storage_info_t si, const unsigned int isize, const unsigned int jsize, const unsigned int ksize) {
    const int NBI = isize/BLOCKSIZEX;
    const int NBJ = jsize/BLOCKSIZEY;
    const int istride = si.template stride<0>();
    const int jstride = si.template stride<1>();
    const int kstride = si.template stride<2>();

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < NBI; ++bi) {
        for (int bj = 0; bj < NBJ; ++bj) {
            int start_index_i = bi*BLOCKSIZEX;
            int start_index_j = bj*BLOCKSIZEY;
            int index = si.index(start_index_i+h,start_index_j+h,h);
            for(int k=0; k < ksize; ++k) {
                for(int j=0; j<BLOCKSIZEY; ++j) {
                    for(int i=0; i<BLOCKSIZEX; ++i) {
                        b[index] = a[index] + a[index-1] + a[index+1] + a[index-jstride] + a[index+jstride];
                        ++index;
                    }
                    index+=(jstride-BLOCKSIZEX);
                }
                index+=(kstride-jstride*BLOCKSIZEY);
            }
        }
    }
}
