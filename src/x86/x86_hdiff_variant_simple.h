#pragma once

#include "x86/x86_hdiff_stencil_variant.h"

namespace platform {

    namespace x86 {

        template <class Platform, class ValueType>
        class hdiff_variant_simple final : public x86_hdiff_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;

            hdiff_variant_simple(const arguments_map &args) : x86_hdiff_stencil_variant<Platform, ValueType>(args) {}

            void hdiff() override {

                const value_type *__restrict__ in = this->in();
                const value_type *__restrict__ coeff = this->coeff();
                value_type *__restrict__ lap = this->lap();
                value_type *__restrict__ flx = this->flx();
                value_type *__restrict__ fly = this->fly();
                value_type *__restrict__ out = this->out();  

                const int istride = this->istride();
                const int jstride = this->jstride();
                const int kstride = this->kstride();
                const int h = this->halo();
                const int isize = this->isize();
                const int jsize = this->jsize();
                const int ksize = this->ksize();

                for (int k = 0; k < ksize; ++k) {
                    for (int j = -1; j < jsize+1; ++j) {
                        for (int i = -1; i < isize+1; ++i) {
                            lap[this->index(i, j, k)] =
                                4 * in[this->index(i, j, k)] -
                                (in[this->index(i - 1, j, k)] + in[this->index(i + 1, j, k)] + in[this->index(i, j - 1, k)] + in[this->index(i, j + 1, k)]);
                        }
                    }

                    for (int j = 0; j < jsize; ++j) {
                        for (int i = -1; i < isize; ++i) {
                            flx[this->index(i, j, k)] = lap[this->index(i + 1, j, k)] - lap[this->index(i, j, k)];
                            if (flx[this->index(i, j, k)] * (in[this->index(i + 1, j, k)] - in[this->index(i, j, k)]) > 0)
                                flx[this->index(i, j, k)] = 0.;
                        }
                    }

                    for (int j = -1; j < jsize; ++j) {
                        for (int i = 0; i < isize; ++i) {
                            fly[this->index(i, j, k)] = lap[this->index(i, j + 1, k)] - lap[this->index(i, j, k)];
                            if (fly[this->index(i, j, k)] * (in[this->index(i, j + 1, k)] - in[this->index(i, j, k)]) > 0)
                                fly[this->index(i, j, k)] = 0.;
                        }
                    }

                    for (int i = 0; i < isize; ++i) {
                        for (int j = 0; j < jsize; ++j) {
                            out[this->index(i, j, k)] =
                                in[this->index(i, j, k)] - coeff[this->index(i, j, k)] *
                                    (flx[this->index(i, j, k)] - flx[this->index(i - 1, j, k)] + fly[this->index(i, j, k)] - fly[this->index(i, j - 1, k)]);
                        }
                    }
                }                
            }
        };

    } // namespace x86

} // namespace platform
