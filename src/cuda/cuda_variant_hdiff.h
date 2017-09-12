#pragma once

#include "vadv_stencil_variant.h"

namespace platform {

    namespace cuda {



#define BLOCK_X_SIZE 32
#define BLOCK_Y_SIZE 8

#define HALO_BLOCK_X_MINUS 1
#define HALO_BLOCK_X_PLUS 1

#define HALO_BLOCK_Y_MINUS 1
#define HALO_BLOCK_Y_PLUS 1

#define PADDED_BOUNDARY 1

inline __device__ unsigned int cache_index(const int ipos, const int jpos) {
    return (ipos + PADDED_BOUNDARY) +
           (jpos + HALO_BLOCK_Y_MINUS) * (BLOCK_X_SIZE + HALO_BLOCK_X_MINUS + HALO_BLOCK_X_PLUS);
}

template < typename ValueType >
__global__ void cukernel(
    ValueType const* in, ValueType *out, ValueType const* coeff, const IJKSize domain, const IJKSize halo, const IJKSize strides) {

    unsigned int ipos, jpos;
    int iblock_pos, jblock_pos;
    const unsigned int jboundary_limit = BLOCK_Y_SIZE + HALO_BLOCK_Y_MINUS + HALO_BLOCK_Y_PLUS;
    const unsigned int iminus_limit = jboundary_limit + HALO_BLOCK_X_MINUS;
    const unsigned int iplus_limit = iminus_limit + HALO_BLOCK_X_PLUS;

    const unsigned int block_size_i =
        (blockIdx.x + 1) * BLOCK_X_SIZE < domain.m_i ? BLOCK_X_SIZE : domain.m_i - blockIdx.x * BLOCK_X_SIZE;
    const unsigned int block_size_j =
        (blockIdx.y + 1) * BLOCK_Y_SIZE < domain.m_j ? BLOCK_Y_SIZE : domain.m_j - blockIdx.y * BLOCK_Y_SIZE;

    // set the thread position by default out of the block
    iblock_pos = -HALO_BLOCK_X_MINUS - 1;
    jblock_pos = -HALO_BLOCK_Y_MINUS - 1;
    if (threadIdx.y < jboundary_limit) {
        ipos = blockIdx.x * BLOCK_X_SIZE + threadIdx.x + halo.m_i;
        jpos = blockIdx.y * BLOCK_Y_SIZE + threadIdx.y - HALO_BLOCK_Y_MINUS + halo.m_j;
        iblock_pos = threadIdx.x;
        jblock_pos = threadIdx.y - HALO_BLOCK_Y_MINUS;
    } else if (threadIdx.y < iminus_limit && threadIdx.x < BLOCK_Y_SIZE * PADDED_BOUNDARY) {
        ipos = blockIdx.x * BLOCK_X_SIZE - PADDED_BOUNDARY + threadIdx.x % PADDED_BOUNDARY + halo.m_i;
        jpos = blockIdx.y * BLOCK_Y_SIZE + threadIdx.x / PADDED_BOUNDARY + halo.m_j;
        iblock_pos = -PADDED_BOUNDARY + (int)threadIdx.x % PADDED_BOUNDARY;
        jblock_pos = threadIdx.x / PADDED_BOUNDARY;
    } else if (threadIdx.y < iplus_limit && threadIdx.x < BLOCK_Y_SIZE * PADDED_BOUNDARY) {
        ipos = blockIdx.x * BLOCK_X_SIZE + threadIdx.x % PADDED_BOUNDARY + BLOCK_X_SIZE + halo.m_i;
        jpos = blockIdx.y * BLOCK_Y_SIZE + threadIdx.x / PADDED_BOUNDARY + halo.m_j;
        iblock_pos = threadIdx.x % PADDED_BOUNDARY + BLOCK_X_SIZE;
        jblock_pos = threadIdx.x / PADDED_BOUNDARY;
    }

    int index_ = index(ipos, jpos, 0, strides);

// flx and fly can be defined with smaller cache sizes, however in order to reuse the same cache_index function, I
// defined them here
// with same size. shared memory pressure should not be too high nevertheless
#define CACHE_SIZE (BLOCK_X_SIZE + HALO_BLOCK_X_MINUS + HALO_BLOCK_X_PLUS) * (BLOCK_Y_SIZE + 2)
    __shared__ ValueType lap[CACHE_SIZE];
    __shared__ ValueType flx[CACHE_SIZE];
    __shared__ ValueType fly[CACHE_SIZE];

    for (int kpos = 0; kpos < domain.m_k; ++kpos) {

        if (is_in_domain< -1, 1, -1, 1 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {

            lap[cache_index(iblock_pos, jblock_pos)] =
                (ValueType)4 * __ldg(& in[index_] ) -
                ( __ldg(& in[index_+index(1, 0,0, strides)] ) + __ldg(& in[index_ - index(1, 0,0, strides)] ) +
                    __ldg(&in[index_+index(0, 1, 0, strides)]) + __ldg(&in[index_ - index(0, 1, 0, strides)]));
        }

        __syncthreads();

        if (is_in_domain< -1, 0, 0, 0 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {
            flx[cache_index(iblock_pos, jblock_pos)] =
                lap[cache_index(iblock_pos + 1, jblock_pos)] - lap[cache_index(iblock_pos, jblock_pos)];
            if (flx[cache_index(iblock_pos, jblock_pos)] *
                    (__ldg(&in[index_+index(1, 0, 0, strides)]) - __ldg(&in[index_])) >
                0) {
                flx[cache_index(iblock_pos, jblock_pos)] = 0.;
            }
        }

        if (is_in_domain< 0, 0, -1, 0 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {
            fly[cache_index(iblock_pos, jblock_pos)] =
                lap[cache_index(iblock_pos, jblock_pos + 1)] - lap[cache_index(iblock_pos, jblock_pos)];
            if (fly[cache_index(iblock_pos, jblock_pos)] *
                    (__ldg(&in[index_+index(0, 1, 0, strides)]) - __ldg(&in[index_])) >
                0) {
                fly[cache_index(iblock_pos, jblock_pos)] = 0.;
            }
        }

        __syncthreads();

        if (is_in_domain< 0, 0, 0, 0 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {
            out[index_] =
                __ldg(&in[index_]) -
                coeff[index_] *
                    (flx[cache_index(iblock_pos, jblock_pos)] - flx[cache_index(iblock_pos - 1, jblock_pos)] +
                        fly[cache_index(iblock_pos, jblock_pos)] - fly[cache_index(iblock_pos, jblock_pos - 1)]);
        }

        index_ += index(0,0,1, strides);
    }
}


        template <class Platform, class ValueType>
        class variant_hdiff final : public hdiff_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;
            using platform = Platform;
            using allocator = typename platform::template allocator<value_type>;

            variant_hdiff(const arguments_map &args)
                : vadv_stencil_variant<Platform, ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
            }
            ~variant_hdiff() {}

            void hdiff() override {
                kernel_hdiff<<<blocks(), blocksize()>>>(this->lap(),
                    this->flx(),
                    this->fly(),
                    this->out(),
                    this->in(),
                    this->coeff(),
                    this->isize(),
                    this->jsize(),
                    this->ksize(),
                    this->istride(),
                    this->jstride(),
                    this->kstride(),
                    this->halo());
                if (cudaDeviceSynchronize() != cudaSuccess)
                    throw ERROR("error in cudaDeviceSynchronize");
            }

          private:
            dim3 blocks() const {
                return dim3((this->isize() + m_iblocksize - 1) / m_iblocksize,
                    (this->jsize() + m_jblocksize - 1) / m_jblocksize);
            }

            dim3 blocksize() const { return dim3(m_iblocksize, m_jblocksize); }

            int m_iblocksize, m_jblocksize;
        };

    } // cuda

} // namespace platform
