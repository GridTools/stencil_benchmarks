#pragma once

#include "vadv_stencil_variant.h"

namespace platform {

    namespace cuda {

        template <class ValueType>
        __forceinline__ __device__ void backward_sweep(const ValueType *ccol,
            const ValueType *__restrict__ dcol,
            const ValueType *__restrict__ upos,
            ValueType *__restrict__ utensstage,
            const int isize,
            const int jsize,
            const int ksize,
            const int istride,
            const int jstride,
            const int kstride) {
            constexpr ValueType dtr_stage = 3.0 / 20.0;
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            const int j = blockIdx.y * blockDim.y + threadIdx.y;

            ValueType datacol;
            if (i < isize && j < jsize) {
                // k maximum
                {
                    const int k = ksize - 1;
                    const int index = i * istride + j * jstride + k * kstride;
                    datacol = dcol[index];
                    utensstage[index] = dtr_stage * (datacol - upos[index]);
                }

                // k body
                for (int k = ksize - 2; k >= 0; --k) {
                    int index = i * istride + j * jstride + k * kstride;
                    datacol = dcol[index] - ccol[index] * datacol;
                    utensstage[index] = dtr_stage * (datacol - upos[index]);
                }
            }
        }

        template <class ValueType>
        __forceinline__ __device__ void forward_sweep(const int ishift,
            const int jshift,
            ValueType *__restrict__ ccol,
            ValueType *__restrict__ dcol,
            const ValueType *__restrict__ wcon,
            const ValueType *__restrict__ ustage,
            const ValueType *__restrict__ upos,
            const ValueType *__restrict__ utens,
            const ValueType *__restrict__ utensstage,
            const int isize,
            const int jsize,
            const int ksize,
            const int istride,
            const int jstride,
            const int kstride) {
            constexpr ValueType dtr_stage = 3.0 / 20.0;
            constexpr ValueType beta_v = 0;
            constexpr ValueType bet_m = 0.5 * (1.0 - beta_v);
            constexpr ValueType bet_p = 0.5 * (1.0 + beta_v);
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            const int j = blockIdx.y * blockDim.y + threadIdx.y;

            ValueType ccol0, ccol1;
            ValueType dcol0, dcol1;

            if (i < isize && j < jsize) {
                // k minimum
                {
                    const int k = 0;
                    const int index = i * istride + j * jstride + k * kstride;
                    ValueType gcv = ValueType(0.25) * (wcon[index + ishift * istride + jshift * jstride + kstride] +
                                                          wcon[index + kstride]);
                    ValueType cs = gcv * bet_m;

                    ccol0 = gcv * bet_p;
                    ValueType bcol = dtr_stage - ccol0;

                    ValueType correction_term = -cs * (ustage[index + kstride] - ustage[index]);
                    dcol0 = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    ValueType divided = ValueType(1.0) / bcol;
                    ccol0 = ccol0 * divided;
                    dcol0 = dcol0 * divided;

                    ccol[index] = ccol0;
                    dcol[index] = dcol0;
                }

                // k body
                for (int k = 1; k < ksize - 1; ++k) {
                    ccol1 = ccol0;
                    dcol1 = dcol0;
                    const int index = i * istride + j * jstride + k * kstride;
                    ValueType gav =
                        ValueType(-0.25) * (wcon[index + ishift * istride + jshift * jstride] + wcon[index]);
                    ValueType gcv = ValueType(0.25) * (wcon[index + ishift * istride + jshift * jstride + kstride] +
                                                          wcon[index + kstride]);

                    ValueType as = gav * bet_m;
                    ValueType cs = gcv * bet_m;

                    ValueType acol = gav * bet_p;
                    ccol0 = gcv * bet_p;
                    ValueType bcol = dtr_stage - acol - ccol0;

                    ValueType correction_term = -as * (ustage[index - kstride] - ustage[index]) -
                                                cs * (ustage[index + kstride] - ustage[index]);
                    dcol0 = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    ValueType divided = ValueType(1.0) / (bcol - ccol1 * acol);
                    ccol0 = ccol0 * divided;
                    dcol0 = (dcol0 - dcol1 * acol) * divided;

                    ccol[index] = ccol0;
                    dcol[index] = dcol0;
                }

                // k maximum
                {
                    ccol1 = ccol0;
                    dcol1 = dcol0;
                    const int k = ksize - 1;
                    const int index = i * istride + j * jstride + k * kstride;
                    ValueType gav =
                        ValueType(-0.25) * (wcon[index + ishift * istride + jshift * jstride] + wcon[index]);

                    ValueType as = gav * bet_m;

                    ValueType acol = gav * bet_p;
                    ValueType bcol = dtr_stage - acol;

                    ValueType correction_term = -as * (ustage[index - kstride] - ustage[index]);
                    dcol0 = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    ValueType divided = ValueType(1.0) / (bcol - ccol1 * acol);
                    dcol0 = (dcol0 - dcol1 * acol) * divided;

                    ccol[index] = ccol0;
                    dcol[index] = dcol0;
                }
            }
        }

        template <class ValueType>
        __global__ void kernel_vadv(const ValueType *ustage,
            const ValueType *__restrict__ upos,
            const ValueType *__restrict__ utens,
            ValueType *__restrict__ utensstage,
            const ValueType *__restrict__ vstage,
            const ValueType *__restrict__ vpos,
            const ValueType *__restrict__ vtens,
            ValueType *__restrict__ vtensstage,
            const ValueType *__restrict__ wstage,
            const ValueType *__restrict__ wpos,
            const ValueType *__restrict__ wtens,
            ValueType *__restrict__ wtensstage,
            ValueType *__restrict__ ccol,
            ValueType *__restrict__ dcol,
            const ValueType *__restrict__ wcon,
            const int isize,
            const int jsize,
            const int ksize,
            const int istride,
            const int jstride,
            const int kstride) {
            forward_sweep(1,
                0,
                ccol,
                dcol,
                wcon,
                ustage,
                upos,
                utens,
                utensstage,
                isize,
                jsize,
                ksize,
                istride,
                jstride,
                kstride);
            backward_sweep(ccol, dcol, upos, utensstage, isize, jsize, ksize, istride, jstride, kstride);

            forward_sweep(1,
                0,
                ccol,
                dcol,
                wcon,
                vstage,
                vpos,
                vtens,
                vtensstage,
                isize,
                jsize,
                ksize,
                istride,
                jstride,
                kstride);
            backward_sweep(ccol, dcol, vpos, vtensstage, isize, jsize, ksize, istride, jstride, kstride);

            forward_sweep(1,
                0,
                ccol,
                dcol,
                wcon,
                wstage,
                wpos,
                wtens,
                wtensstage,
                isize,
                jsize,
                ksize,
                istride,
                jstride,
                kstride);
            backward_sweep(ccol, dcol, wpos, wtensstage, isize, jsize, ksize, istride, jstride, kstride);
        }

        template <class Platform, class ValueType>
        class vadv_variant final : public vadv_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;
            using platform = Platform;
            using allocator = typename platform::template allocator<value_type>;

            vadv_variant(const arguments_map &args)
                : vadv_stencil_variant<Platform, ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
                platform::limit_blocksize(m_iblocksize, m_jblocksize);
            }
            ~vadv_variant() {}

            void prerun() override {
                vadv_stencil_variant<platform, value_type>::prerun();

                auto prefetch = [&](const value_type *ptr) {
                    if (cudaMemPrefetchAsync(ptr - this->zero_offset(), this->storage_size() * sizeof(value_type), 0) !=
                        cudaSuccess)
                        throw ERROR("error in cudaMemPrefetchAsync");
                };
                prefetch(this->ustage());
                prefetch(this->upos());
                prefetch(this->utens());
                prefetch(this->utensstage());
                prefetch(this->vstage());
                prefetch(this->vpos());
                prefetch(this->vtens());
                prefetch(this->vtensstage());
                prefetch(this->wstage());
                prefetch(this->wpos());
                prefetch(this->wtens());
                prefetch(this->wtensstage());
                prefetch(this->ccol());
                prefetch(this->dcol());
                prefetch(this->wcon());
                prefetch(this->datacol());

                if (cudaDeviceSynchronize() != cudaSuccess)
                    throw ERROR("error in cudaDeviceSynchronize");
            }

            void vadv() override {
                kernel_vadv<<<blocks(), blocksize()>>>(this->ustage(),
                    this->upos(),
                    this->utens(),
                    this->utensstage(),
                    this->vstage(),
                    this->vpos(),
                    this->vtens(),
                    this->vtensstage(),
                    this->wstage(),
                    this->wpos(),
                    this->wtens(),
                    this->wtensstage(),
                    this->ccol(),
                    this->dcol(),
                    this->wcon(),
                    this->isize(),
                    this->jsize(),
                    this->ksize(),
                    this->istride(),
                    this->jstride(),
                    this->kstride());
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
