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
                int index = i * istride + j * jstride + (ksize - 1) * kstride;
                // k maximum
                {
                    datacol = dcol[index];
                    utensstage[index] = dtr_stage * (datacol - upos[index]);

                    index -= kstride;
                }

                // k body
                for (int k = ksize - 2; k >= 0; --k) {
                    datacol = dcol[index] - ccol[index] * datacol;
                    utensstage[index] = dtr_stage * (datacol - upos[index]);

                    index -= kstride;
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
            ValueType ustage0, ustage1, ustage2;
            ValueType wcon0, wcon1;
            ValueType wcon_shift0, wcon_shift1;

            if (i < isize && j < jsize) {
                int index = i * istride + j * jstride;
                // k minimum
                {
                    wcon_shift0 = wcon[index + ishift * istride + jshift * jstride + kstride];
                    wcon0 = wcon[index + kstride];
                    ValueType gcv = ValueType(0.25) * (wcon_shift0 + wcon0);
                    ValueType cs = gcv * bet_m;

                    ccol0 = gcv * bet_p;
                    ValueType bcol = dtr_stage - ccol0;

                    ustage0 = ustage[index + kstride];
                    ustage1 = ustage[index];
                    ValueType correction_term = -cs * (ustage0 - ustage1);
                    dcol0 = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    ValueType divided = ValueType(1.0) / bcol;
                    ccol0 = ccol0 * divided;
                    dcol0 = dcol0 * divided;

                    ccol[index] = ccol0;
                    dcol[index] = dcol0;

                    index += kstride;
                }

                // k body
                for (int k = 1; k < ksize - 1; ++k) {
                    ccol1 = ccol0;
                    dcol1 = dcol0;
                    ustage2 = ustage1;
                    ustage1 = ustage0;
                    wcon1 = wcon0;
                    wcon_shift1 = wcon_shift0;

                    ValueType gav = ValueType(-0.25) * (wcon_shift1 + wcon1);
                    wcon_shift0 = wcon[index + ishift * istride + jshift * jstride + kstride];
                    wcon0 = wcon[index + kstride];
                    ValueType gcv = ValueType(0.25) * (wcon_shift0 + wcon0);

                    ValueType as = gav * bet_m;
                    ValueType cs = gcv * bet_m;

                    ValueType acol = gav * bet_p;
                    ccol0 = gcv * bet_p;
                    ValueType bcol = dtr_stage - acol - ccol0;

                    ustage0 = ustage[index + kstride];
                    ValueType correction_term = -as * (ustage2 - ustage1) - cs * (ustage0 - ustage1);
                    dcol0 = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    ValueType divided = ValueType(1.0) / (bcol - ccol1 * acol);
                    ccol0 = ccol0 * divided;
                    dcol0 = (dcol0 - dcol1 * acol) * divided;

                    ccol[index] = ccol0;
                    dcol[index] = dcol0;

                    index += kstride;
                }

                // k maximum
                {
                    ccol1 = ccol0;
                    dcol1 = dcol0;
                    ustage2 = ustage1;
                    ustage1 = ustage0;
                    wcon1 = wcon0;
                    wcon_shift1 = wcon_shift0;

                    ValueType gav = ValueType(-0.25) * (wcon_shift1 + wcon1);

                    ValueType as = gav * bet_m;

                    ValueType acol = gav * bet_p;
                    ValueType bcol = dtr_stage - acol;

                    ValueType correction_term = -as * (ustage2 - ustage1);
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

            forward_sweep(0,
                1,
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

            forward_sweep(0,
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

        template <class ValueType>
        class vadv_variant final : public vadv_stencil_variant<cuda, ValueType> {
          public:
            using value_type = ValueType;
            using platform = cuda;
            using allocator = typename platform::template allocator<value_type>;

            vadv_variant(const arguments_map &args)
                : vadv_stencil_variant<cuda, ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
                platform::limit_blocksize(m_iblocksize, m_jblocksize);
            }
            ~vadv_variant() {}

            void setup() override {
                vadv_stencil_variant<platform, value_type>::setup();

                if (cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) != cudaSuccess)
                    throw ERROR("error in cudaDeviceSetCacheConfig");

                if (cudaDeviceSetSharedMemConfig(
                        sizeof(ValueType) == 8 ? cudaSharedMemBankSizeEightByte : cudaSharedMemBankSizeFourByte))
                    throw ERROR("error in cudaDeviceSetSharedMemConfig");

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

            void vadv(unsigned int i) override {
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
