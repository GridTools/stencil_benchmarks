#pragma once

#include "vadv_stencil_variant.h"

namespace platform {

    namespace cuda {

        template <class ValueType>
        __inline__ __device__ void backward_sweep(ValueType *ccol,
            const ValueType *dcol,
            ValueType *datacol,
            const ValueType *upos,
            ValueType *utensstage,
            const int isize,
            const int jsize,
            const int ksize,
            const int istride,
            const int jstride,
            const int kstride) {
            constexpr ValueType dtr_stage = 3.0 / 20.0;
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            const int j = blockIdx.y * blockDim.y + threadIdx.y;

            if (i < isize && j < jsize) {
                // k maximum
                {
                    const int k = ksize - 1;
                    const int index = i * istride + j * jstride + k * kstride;
                    datacol[index] = dcol[index];
                    ccol[index] = datacol[index];
                    utensstage[index] = dtr_stage * (datacol[index] - upos[index]);
                }

                // k body
                for (int k = ksize - 2; k >= 0; --k) {
                    int index = i * istride + j * jstride + k * kstride;
                    datacol[index] = dcol[index] - ccol[index] * datacol[index + kstride];
                    ccol[index] = datacol[index];
                    utensstage[index] = dtr_stage * (datacol[index] - upos[index]);
                }
            }
        }

        template <class ValueType>
        __inline__ __device__ void forward_sweep(const int ishift,
            const int jshift,
            ValueType *ccol,
            ValueType *dcol,
            const ValueType *wcon,
            const ValueType *ustage,
            const ValueType *upos,
            const ValueType *utens,
            const ValueType *utensstage,
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

            if (i < isize && j < jsize) {
                // k minimum
                {
                    const int k = 0;
                    const int index = i * istride + j * jstride + k * kstride;
                    ValueType gcv = ValueType(0.25) * (wcon[index + ishift * istride + jshift * jstride + kstride] +
                                                          wcon[index + kstride]);
                    ValueType cs = gcv * bet_m;

                    ccol[index] = gcv * bet_p;
                    ValueType bcol = dtr_stage - ccol[index];

                    ValueType correction_term = -cs * (ustage[index + kstride] - ustage[index]);
                    dcol[index] = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    ValueType divided = ValueType(1.0) / bcol;
                    ccol[index] = ccol[index] * divided;
                    dcol[index] = dcol[index] * divided;
                }

                // k body
                for (int k = 1; k < ksize - 1; ++k) {
                    const int index = i * istride + j * jstride + k * kstride;
                    ValueType gav =
                        ValueType(-0.25) * (wcon[index + ishift * istride + jshift * jstride] + wcon[index]);
                    ValueType gcv = ValueType(0.25) * (wcon[index + ishift * istride + jshift * jstride + kstride] +
                                                          wcon[index + kstride]);

                    ValueType as = gav * bet_m;
                    ValueType cs = gcv * bet_m;

                    ValueType acol = gav * bet_p;
                    ccol[index] = gcv * bet_p;
                    ValueType bcol = dtr_stage - acol - ccol[index];

                    ValueType correction_term = -as * (ustage[index - kstride] - ustage[index]) -
                                                cs * (ustage[index + kstride] - ustage[index]);
                    dcol[index] = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    ValueType divided = ValueType(1.0) / (bcol - ccol[index - kstride] * acol);
                    ccol[index] = ccol[index] * divided;
                    dcol[index] = (dcol[index] - dcol[index - kstride] * acol) * divided;
                }

                // k maximum
                {
                    const int k = ksize - 1;
                    const int index = i * istride + j * jstride + k * kstride;
                    ValueType gav =
                        ValueType(-0.25) * (wcon[index + ishift * istride + jshift * jstride] + wcon[index]);

                    ValueType as = gav * bet_m;

                    ValueType acol = gav * bet_p;
                    ValueType bcol = dtr_stage - acol;

                    ValueType correction_term = -as * (ustage[index - kstride] - ustage[index]);
                    dcol[index] = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    ValueType divided = ValueType(1.0) / (bcol - ccol[index - kstride] * acol);
                    dcol[index] = (dcol[index] - dcol[index - kstride] * acol) * divided;
                }
            }
        }

        template <class ValueType>
        __global__ void kernel_vadv(const ValueType *ustage,
            const ValueType *upos,
            const ValueType *utens,
            ValueType *utensstage,
            const ValueType *vstage,
            const ValueType *vpos,
            const ValueType *vtens,
            ValueType *vtensstage,
            const ValueType *wstage,
            const ValueType *wpos,
            const ValueType *wtens,
            ValueType *wtensstage,
            ValueType *ccol,
            ValueType *dcol,
            const ValueType *wcon,
            ValueType *datacol,
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
            backward_sweep(ccol, dcol, datacol, upos, utensstage, isize, jsize, ksize, istride, jstride, kstride);

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
            backward_sweep(ccol, dcol, datacol, vpos, vtensstage, isize, jsize, ksize, istride, jstride, kstride);

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
            backward_sweep(ccol, dcol, datacol, wpos, wtensstage, isize, jsize, ksize, istride, jstride, kstride);
        }

        template <class Platform, class ValueType>
        class variant_vadv final : public vadv_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;
            using platform = Platform;
            using allocator = typename platform::template allocator<value_type>;

            variant_vadv(const arguments_map &args)
                : vadv_stencil_variant<Platform, ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
            }
            ~variant_vadv() {}

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
                    this->datacol(),
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
