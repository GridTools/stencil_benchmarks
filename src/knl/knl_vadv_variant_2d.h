#pragma once

#include "knl/knl_variant_vadv.h"

namespace platform {

    namespace knl {

        template <class Platform, class ValueType>
        class variant_vadv_2d final : public knl_vadv_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;
            using platform = Platform;
            using allocator = typename platform::template allocator<value_type>;

            variant_vadv_2d(const arguments_map &args) : knl_vadv_stencil_variant<Platform, ValueType>(args) {}
            ~variant_vadv_2d() {}

            void vadv(counter &ctr) override {
                const value_type *__restrict__ ustage = this->ustage();
                const value_type *__restrict__ upos = this->upos();
                const value_type *__restrict__ utens = this->utens();
                value_type *__restrict__ utensstage = this->utensstage();
                const value_type *__restrict__ vstage = this->vstage();
                const value_type *__restrict__ vpos = this->vpos();
                const value_type *__restrict__ vtens = this->vtens();
                value_type *__restrict__ vtensstage = this->vtensstage();
                const value_type *__restrict__ wstage = this->wstage();
                const value_type *__restrict__ wpos = this->wpos();
                const value_type *__restrict__ wtens = this->wtens();
                value_type *__restrict__ wtensstage = this->wtensstage();
                value_type *__restrict__ ccol = this->ccol();
                value_type *__restrict__ dcol = this->dcol();
                const value_type *__restrict__ wcon = this->wcon();
                value_type *__restrict__ datacol = this->datacol();
                const int isize = this->isize();
                const int jsize = this->jsize();
                const int ksize = this->ksize();
                const int istride = this->istride();
                const int jstride = this->jstride();
                const int kstride = this->kstride();

                const int last = this->index(isize - 1, jsize - 1, ksize - 1);
#pragma omp parallel
                {
                    ctr.start();
#pragma omp for collapse(2)
                    for (int j = 0; j < jsize; ++j) {
                        for (int i = 0; i < isize; ++i) {
                            kernel_vadv(i,
                                j,
                                ustage,
                                upos,
                                utens,
                                utensstage,
                                vstage,
                                vpos,
                                vtens,
                                vtensstage,
                                wstage,
                                wpos,
                                wtens,
                                wtensstage,
                                ccol,
                                dcol,
                                wcon,
                                datacol,
                                isize,
                                jsize,
                                ksize,
                                istride,
                                jstride,
                                kstride);
                        }
                    }
                    ctr.stop();
                }
            }

          private:
            void backward_sweep(const int i,
                const int j,
                value_type *__restrict__ ccol,
                const value_type *__restrict__ dcol,
                value_type *__restrict__ datacol,
                const value_type *__restrict__ upos,
                value_type *__restrict__ utensstage,
                const int isize,
                const int jsize,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride) {
                constexpr value_type dtr_stage = 3.0 / 20.0;

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

            void forward_sweep(const int i,
                const int j,
                const int ishift,
                const int jshift,
                value_type *__restrict__ ccol,
                value_type *__restrict__ dcol,
                const value_type *__restrict__ wcon,
                const value_type *__restrict__ ustage,
                const value_type *__restrict__ upos,
                const value_type *__restrict__ utens,
                const value_type *__restrict__ utensstage,
                const int isize,
                const int jsize,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride) {
                constexpr value_type dtr_stage = 3.0 / 20.0;
                constexpr value_type beta_v = 0;
                constexpr value_type bet_m = 0.5 * (1.0 - beta_v);
                constexpr value_type bet_p = 0.5 * (1.0 + beta_v);

                if (i < isize && j < jsize) {
                    // k minimum
                    {
                        const int k = 0;
                        const int index = i * istride + j * jstride + k * kstride;
                        value_type gcv =
                            value_type(0.25) *
                            (wcon[index + ishift * istride + jshift * jstride + kstride] + wcon[index + kstride]);
                        value_type cs = gcv * bet_m;

                        ccol[index] = gcv * bet_p;
                        value_type bcol = dtr_stage - ccol[index];

                        value_type correction_term = -cs * (ustage[index + kstride] - ustage[index]);
                        dcol[index] = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                        value_type divided = value_type(1.0) / bcol;
                        ccol[index] = ccol[index] * divided;
                        dcol[index] = dcol[index] * divided;
                    }

                    // k body
                    for (int k = 1; k < ksize - 1; ++k) {
                        const int index = i * istride + j * jstride + k * kstride;
                        value_type gav =
                            value_type(-0.25) * (wcon[index + ishift * istride + jshift * jstride] + wcon[index]);
                        value_type gcv =
                            value_type(0.25) *
                            (wcon[index + ishift * istride + jshift * jstride + kstride] + wcon[index + kstride]);

                        value_type as = gav * bet_m;
                        value_type cs = gcv * bet_m;

                        value_type acol = gav * bet_p;
                        ccol[index] = gcv * bet_p;
                        value_type bcol = dtr_stage - acol - ccol[index];

                        value_type correction_term = -as * (ustage[index - kstride] - ustage[index]) -
                                                     cs * (ustage[index + kstride] - ustage[index]);
                        dcol[index] = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                        value_type divided = value_type(1.0) / (bcol - ccol[index - kstride] * acol);
                        ccol[index] = ccol[index] * divided;
                        dcol[index] = (dcol[index] - dcol[index - kstride] * acol) * divided;
                    }

                    // k maximum
                    {
                        const int k = ksize - 1;
                        const int index = i * istride + j * jstride + k * kstride;
                        value_type gav =
                            value_type(-0.25) * (wcon[index + ishift * istride + jshift * jstride] + wcon[index]);

                        value_type as = gav * bet_m;

                        value_type acol = gav * bet_p;
                        value_type bcol = dtr_stage - acol;

                        value_type correction_term = -as * (ustage[index - kstride] - ustage[index]);
                        dcol[index] = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                        value_type divided = value_type(1.0) / (bcol - ccol[index - kstride] * acol);
                        dcol[index] = (dcol[index] - dcol[index - kstride] * acol) * divided;
                    }
                }
            }

            void kernel_vadv(const int i,
                const int j,
                const value_type *__restrict__ ustage,
                const value_type *__restrict__ upos,
                const value_type *__restrict__ utens,
                value_type *__restrict__ utensstage,
                const value_type *__restrict__ vstage,
                const value_type *__restrict__ vpos,
                const value_type *__restrict__ vtens,
                value_type *__restrict__ vtensstage,
                const value_type *__restrict__ wstage,
                const value_type *__restrict__ wpos,
                const value_type *__restrict__ wtens,
                value_type *__restrict__ wtensstage,
                value_type *__restrict__ ccol,
                value_type *__restrict__ dcol,
                const value_type *__restrict__ wcon,
                value_type *__restrict__ datacol,
                const int isize,
                const int jsize,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride) {
                forward_sweep(i,
                    j,
                    1,
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
                backward_sweep(
                    i, j, ccol, dcol, datacol, upos, utensstage, isize, jsize, ksize, istride, jstride, kstride);

                forward_sweep(i,
                    j,
                    1,
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
                backward_sweep(
                    i, j, ccol, dcol, datacol, vpos, vtensstage, isize, jsize, ksize, istride, jstride, kstride);

                forward_sweep(i,
                    j,
                    1,
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
                backward_sweep(
                    i, j, ccol, dcol, datacol, wpos, wtensstage, isize, jsize, ksize, istride, jstride, kstride);
            }
        };

    } // knl

} // namespace platform
