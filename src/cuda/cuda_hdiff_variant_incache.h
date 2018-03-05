#pragma once

#include "hdiff_stencil_variant.h"

namespace platform {

    namespace cuda {
        template <typename ValueType>
        __global__ void kernel_hdiff_incache(ValueType const *__restrict__ in,
            ValueType const *__restrict__ coeff,
            ValueType *__restrict__ out,
            const int isize,
            const int jsize,
            const int ksize,
            const int istride,
            const int jstride,
            const int kstride,
            const int iblocksize,
            const int jblocksize,
            const int kblocksize) {

            constexpr int block_halo = 2;
            constexpr int padded_boundary = block_halo;

            int ipos, jpos;
            int iblock_pos, jblock_pos;
            const int jboundary_limit = jblocksize + 2 * block_halo;
            const int iminus_limit = jboundary_limit + block_halo;
            const int iplus_limit = iminus_limit + block_halo;

            iblock_pos = -block_halo - 1;
            jblock_pos = -block_halo - 1;
            ipos = -block_halo - 1;
            jpos = -block_halo - 1;
            if (threadIdx.y < jboundary_limit) {
                ipos = blockIdx.x * iblocksize + threadIdx.x;
                jpos = (int)blockIdx.y * jblocksize + (int)threadIdx.y + -block_halo;
                iblock_pos = threadIdx.x;
                jblock_pos = (int)threadIdx.y + -block_halo;
            } else if (threadIdx.y < iminus_limit) {
                ipos = blockIdx.x * iblocksize - padded_boundary + threadIdx.x % padded_boundary;
                jpos = (int)blockIdx.y * jblocksize + (int)threadIdx.x / padded_boundary + -block_halo;
                iblock_pos = -padded_boundary + (int)threadIdx.x % padded_boundary;
                jblock_pos = (int)threadIdx.x / padded_boundary + -block_halo;
            } else if (threadIdx.y < iplus_limit) {
                ipos = blockIdx.x * iblocksize + threadIdx.x % padded_boundary + iblocksize;
                jpos = (int)blockIdx.y * jblocksize + (int)threadIdx.x / padded_boundary + -block_halo;
                iblock_pos = threadIdx.x % padded_boundary + iblocksize;
                jblock_pos = (int)threadIdx.x / padded_boundary + -block_halo;
            }

            extern __shared__ char smem[];

            ValueType *__restrict__ inc = reinterpret_cast<ValueType *>(&smem[0]);

            constexpr int cache_istride = 1;
            const int cache_jstride = iblocksize + 2 * block_halo;
            const int cache_kstride = cache_jstride * (jblocksize + 2 * block_halo);

            const int iblock_max = (blockIdx.x + 1) * iblocksize < isize ? iblocksize : isize - blockIdx.x * iblocksize;
            const int jblock_max = (blockIdx.y + 1) * jblocksize < jsize ? jblocksize : jsize - blockIdx.y * jblocksize;

            const int kblock_pos = threadIdx.z;
            const int kpos = blockIdx.z * blockDim.z + threadIdx.z;

            const int cache_index = (iblock_pos + padded_boundary) * cache_istride +
                                    (jblock_pos + block_halo) * cache_jstride + kblock_pos * cache_kstride;
            const int index = ipos * istride + jpos * jstride + kpos * kstride;

            if (iblock_pos >= -2 && iblock_pos < iblock_max + 2 && jblock_pos >= -2 && jblock_pos < jblock_max + 2)
                inc[cache_index] = in[index];

            __syncthreads();

            if (iblock_pos >= 0 && iblock_pos < iblock_max && jblock_pos >= 0 && jblock_pos < jblock_max) {
                ValueType lap_ij = 4 * inc[cache_index] - inc[cache_index - cache_istride] -
                                   inc[cache_index + cache_istride] - inc[cache_index - cache_jstride] -
                                   inc[cache_index + cache_jstride];
                ValueType lap_imj = 4 * inc[cache_index - cache_istride] - inc[cache_index - 2 * cache_istride] -
                                    inc[cache_index] - inc[cache_index - cache_istride - cache_jstride] -
                                    inc[cache_index - cache_istride + cache_jstride];
                ValueType lap_ipj =
                    4 * inc[cache_index + cache_istride] - inc[cache_index] - inc[cache_index + 2 * cache_istride] -
                    inc[cache_index + cache_istride - cache_jstride] - inc[cache_index + cache_istride + cache_jstride];
                ValueType lap_ijm = 4 * inc[cache_index - cache_jstride] -
                                    inc[cache_index - cache_istride - cache_jstride] -
                                    inc[cache_index + cache_istride - cache_jstride] -
                                    inc[cache_index - 2 * cache_jstride] - inc[cache_index];
                ValueType lap_ijp = 4 * inc[cache_index + cache_jstride] -
                                    inc[cache_index - cache_istride + cache_jstride] -
                                    inc[cache_index + cache_istride + cache_jstride] - inc[cache_index] -
                                    inc[cache_index + 2 * cache_jstride];

                ValueType flx_ij = lap_ipj - lap_ij;
                flx_ij = flx_ij * (inc[cache_index + cache_istride] - inc[cache_index]) > 0 ? 0 : flx_ij;

                ValueType flx_imj = lap_ij - lap_imj;
                flx_imj = flx_imj * (inc[cache_index] - inc[cache_index - cache_istride]) > 0 ? 0 : flx_imj;

                ValueType fly_ij = lap_ijp - lap_ij;
                fly_ij = fly_ij * (inc[cache_index + cache_jstride] - inc[cache_index]) > 0 ? 0 : fly_ij;

                ValueType fly_ijm = lap_ij - lap_ijm;
                fly_ijm = fly_ijm * (inc[cache_index] - inc[cache_index - cache_jstride]) > 0 ? 0 : fly_ijm;

                out[index] = inc[cache_index] - coeff[index] * (flx_ij - flx_imj + fly_ij - fly_ijm);
            }
        }

        template <class ValueType>
        class hdiff_variant_incache final : public hdiff_stencil_variant<cuda, ValueType> {
            static constexpr int block_halo = 2;
            static constexpr int padded_boundary = block_halo;

          public:
            using value_type = ValueType;
            using platform = cuda;
            using allocator = typename platform::template allocator<value_type>;

            hdiff_variant_incache(const arguments_map &args)
                : hdiff_stencil_variant<cuda, ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")), m_kblocksize(args.get<int>("k-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0 || m_kblocksize <= 0)
                    throw ERROR("invalid block size");
                m_jblocksize += 2 * block_halo + 2;
                platform::limit_blocksize(m_iblocksize, m_jblocksize, m_kblocksize);
                m_jblocksize -= 2 * block_halo + 2;
                if ((m_jblocksize + 2 * block_halo) * padded_boundary > 32 ||
                    m_iblocksize < m_jblocksize + 2 * block_halo) {
                    std::cerr << "WARNING: reset CUDA block size to default to conform to implementation limits "
                              << "(" << m_iblocksize << "x" << m_jblocksize << "x" << m_kblocksize << " to 32x8x1)"
                              << std::endl;
                    m_iblocksize = 32;
                    m_jblocksize = 8;
                    m_kblocksize = 1;
                }
            }
            ~hdiff_variant_incache() {}

            void setup() override {
                hdiff_stencil_variant<platform, value_type>::setup();

                auto prefetch = [&](const value_type *ptr) {
                    if (cudaMemPrefetchAsync(ptr - this->zero_offset(), this->storage_size() * sizeof(value_type), 0) !=
                        cudaSuccess)
                        throw ERROR("error in cudaMemPrefetchAsync");
                };
                prefetch(this->in());
                prefetch(this->coeff());
                prefetch(this->out());

                if (cudaDeviceSynchronize() != cudaSuccess)
                    throw ERROR("error in cudaDeviceSynchronize");

                if (sizeof(value_type) == 4) {
                    if (cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte) != cudaSuccess)
                        throw ERROR("could not set shared memory bank size");
                } else if (sizeof(value_type) == 8) {
                    if (cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) != cudaSuccess)
                        throw ERROR("could not set shared memory bank size");
                }
            }

            void hdiff(unsigned int i) override {
                dim3 bs = blocksize();
                int smem_size = sizeof(value_type) * (bs.x + 2 * block_halo) * (bs.y + 2 * block_halo) * bs.z;
                bs.y += 2 * block_halo + 2 * padded_boundary;
                kernel_hdiff_incache<<<blocks(), bs, smem_size>>>(this->in(),
                    this->coeff(),
                    this->out(),
                    this->isize(),
                    this->jsize(),
                    this->ksize(),
                    this->istride(),
                    this->jstride(),
                    this->kstride(),
                    m_iblocksize,
                    m_jblocksize,
                    m_kblocksize);
                if (cudaDeviceSynchronize() != cudaSuccess)
                    throw ERROR("error in cudaDeviceSynchronize");
            }

          private:
            dim3 blocks() const {
                return dim3((this->isize() + m_iblocksize - 1) / m_iblocksize,
                    (this->jsize() + m_jblocksize - 1) / m_jblocksize,
                    (this->ksize() + m_kblocksize - 1) / m_kblocksize);
            }

            dim3 blocksize() const { return dim3(m_iblocksize, m_jblocksize, m_kblocksize); }

            int m_iblocksize, m_jblocksize, m_kblocksize;
        };

    } // cuda

} // namespace platform
