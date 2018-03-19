#pragma once

#include "irrumesh_stencil_variant.h"
#include "unstructured/mesh.hpp"
#include "unstructured/neighbours_table.hpp"

namespace platform {

    namespace cuda {

#define LOAD(x) x

        template <class ValueType>
        __global__ void kernel_ij_copy_umesh(ValueType *__restrict__ dst,
            const ValueType *__restrict__ src,
            int isize,
            int jsize,
            int ksize,
            int istride,
            int jstride,
            int kstride,
            size_t mesh_size,
            sneighbours_table table) {
            unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < mesh_size) {
                for (int k = 0; k < ksize; ++k) {
                    dst[idx] = src[idx];
                    idx += kstride;
                }
            }
        }

        template <class ValueType>
        __global__ void kernel_ij_on_cells_umesh(ValueType *__restrict__ dst,
            const ValueType *__restrict__ src,
            int isize,
            int jsize,
            int ksize,
            int istride,
            int jstride,
            int kstride,
            size_t mesh_size,
            sneighbours_table table) {

            const unsigned int idx2 = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int idx = idx2;

            extern __shared__ size_t tab[];
            const size_t shared_stride = blockDim.x;
            const size_t stride = table.isize() * table.jsize() * num_colors(table.nloc());

            tab[threadIdx.x + shared_stride * 0] = table.raw_data(idx2 + 0 * stride);
            tab[threadIdx.x + shared_stride * 1] = table.raw_data(idx2 + 1 * stride);
            tab[threadIdx.x + shared_stride * 2] = table.raw_data(idx2 + 2 * stride);

            __syncthreads();

            if (idx < mesh_size) {
                for (int k = 0; k < ksize; ++k) {
                    dst[idx] = src[k * kstride + tab[threadIdx.x + 0 * shared_stride]] +
                               src[k * kstride + tab[threadIdx.x + 1 * shared_stride]] +
                               src[k * kstride + tab[threadIdx.x + 2 * shared_stride]];

                    idx += kstride;
                }
            }
        }

        template <class ValueType>
        class irregular_umesh final : public cuda_umesh_variant<ValueType, irrumesh_stencil_variant> {
          public:
            using platform = cuda;
            using value_type = ValueType;

            inline irregular_umesh(const arguments_map &args)
                : cuda_umesh_variant<ValueType, irrumesh_stencil_variant>(args),
                  m_iblocksize(args.get<int>("i-blocksize")), m_jblocksize(args.get<int>("j-blocksize")) {
                if (this->jsize() % 2 != 0)
                    throw ERROR("jsize must be multiple of 2");

                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
                platform::limit_blocksize(m_iblocksize, m_jblocksize);
                this->mesh_.print();
                this->mesh_.test();
            }

            inline ~irregular_umesh() {}

            void setup() override {
                cuda_umesh_variant<ValueType, irrumesh_stencil_variant>::setup();

                sneighbours_table table = this->mesh_.get_elements(location::cell).table(location::cell);
                if (cudaMemPrefetchAsync(table.data(), table.size_of_array(), 0) != cudaSuccess)
                    throw ERROR("error in cudaMemPrefetchAsync");

                if (cudaDeviceSynchronize() != cudaSuccess)
                    throw ERROR("error in cudaDeviceSynchronize");
            }

            void on_cells_umesh(unsigned int t) {
                kernel_ij_on_cells_umesh<<<blocks(location::cell),
                    blocksize(),
                    blocksize().x * num_neighbours(location::cell, location::cell) * sizeof(size_t)>>>(
                    this->dst_data(t),
                    this->src_data(t),
                    this->isize(),
                    this->jsize(),
                    this->ksize(),
                    this->istride(),
                    this->jstride(),
                    this->kstride(),
                    this->mesh_.mesh_size(location::cell),
                    this->mesh_.get_elements(location::cell).table(location::cell));
            }

            void copy_umesh(unsigned int t) {
                kernel_ij_copy_umesh<<<blocks(location::cell), blocksize()>>>(this->dst_data(t),
                    this->src_data(t),
                    this->isize(),
                    this->jsize(),
                    this->ksize(),
                    this->istride(),
                    this->jstride(),
                    this->kstride(),
                    this->mesh_.mesh_size(location::cell),
                    this->mesh_.get_elements(location::cell).table(location::cell));
            }

          private:
            inline dim3 blocks_ilp(location loc) const {
                if ((this->jsize() % 2) != 0)
                    throw std::runtime_error("jsize should be a multiple of 2 since umesh contains 2 colors");
                return dim3((this->isize() + m_iblocksize - 1) / m_iblocksize,
                    (this->jsize() / 2 + m_jblocksize - 1) / m_jblocksize);
            }
            inline dim3 blocks(location loc) const {
                return dim3(
                    (this->mesh_.mesh_size(loc) + (m_iblocksize * m_jblocksize) - 1) / (m_iblocksize * m_jblocksize),
                    1,
                    1);
            }

            inline dim3 blocksize() const { return dim3(m_iblocksize, m_jblocksize); }

            int m_iblocksize, m_jblocksize;
        };

    } // namespace cuda

} // namespace platform

#undef LOAD
#undef KERNEL_ILP
#undef KERNEL
#undef KERNEL_CALL
