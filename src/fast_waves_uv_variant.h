#pragma once

#include <iterator>
#include <random>

#include "except.h"
#include "variant_base.h"

namespace platform {

    template <class Platform, class ValueType>
    class fast_waves_uv_variant : public variant_base {
      public:
        using platform = Platform;
        using value_type = ValueType;
        using allocator = typename platform::template allocator<value_type>;

        fast_waves_uv_variant(const arguments_map &args);
        virtual ~fast_waves_uv_variant() {}

        std::vector<std::string> stencil_list() const override;

        void prerun() override;

        virtual void fast_waves_uv() = 0;

      protected:
        value_type *u_ref() { return m_u_ref.data() + zero_offset(); }
        value_type *v_ref() { return m_v_ref.data() + zero_offset(); }
        value_type *u_out() { return m_u_out.data() + zero_offset(); }
        value_type *v_out() { return m_v_out.data() + zero_offset(); }
        value_type *u_pos() { return m_u_pos.data() + zero_offset(); }
        value_type *v_pos() { return m_v_pos.data() + zero_offset(); }
        value_type *u_tens() { return m_u_tens.data() + zero_offset(); }
        value_type *v_tens() { return m_v_tens.data() + zero_offset(); }
        value_type *rho() { return m_rho.data() + zero_offset(); }
        value_type *ppuv() { return m_ppuv.data() + zero_offset(); }
        value_type *fx() { return m_fx.data() + zero_offset(); }
        value_type *rho0() { return m_rho0.data() + zero_offset(); }
        value_type *cwp() { return m_cwp.data() + zero_offset(); }
        value_type *p0() { return m_p0.data() + zero_offset(); }
        value_type *wbbctens_stage() { return m_wbbctens_stage.data() + zero_offset(); }
        value_type *wgtfac() { return m_wgtfac.data() + zero_offset(); }
        value_type *hhl() { return m_hhl.data() + zero_offset(); }
        value_type *xlhsx() { return m_xlhsx.data() + zero_offset(); }
        value_type *xlhsy() { return m_xlhsy.data() + zero_offset(); }
        value_type *xdzdx() { return m_xdzdx.data() + zero_offset(); }
        value_type *xdzdy() { return m_xdzdy.data() + zero_offset(); }
        value_type *xrhsx_ref() { return m_xrhsx_ref.data() + zero_offset(); }
        value_type *xrhsy_ref() { return m_xrhsy_ref.data() + zero_offset(); }
        value_type *xrhsz_ref() { return m_xrhsz_ref.data() + zero_offset(); }
        value_type *ppgradcor() { return m_ppgradcor.data() + zero_offset(); }
        value_type *ppgradu() { return m_ppgradu.data() + zero_offset(); }
        value_type *ppgradv() { return m_ppgradv.data() + zero_offset(); }

        std::function<void()> stencil_function(const std::string &stencil) override;

        bool verify(const std::string &stencil) override;

        std::size_t touched_elements(const std::string &stencil) const override;
        std::size_t bytes_per_element() const override { return sizeof(value_type); }

        std::vector<value_type, allocator> m_u_ref, m_v_ref, m_u_out, m_v_out, m_u_pos, m_v_pos, m_u_tens, m_v_tens,
            m_rho, m_ppuv, m_fx, m_rho0, m_cwp, m_p0, m_wbbctens_stage, m_wgtfac, m_hhl, m_xlhsx, m_xlhsy, m_xdzdx,
            m_xdzdy, m_xrhsy_ref, m_xrhsx_ref, m_xrhsz_ref, m_ppgradcor, m_ppgradu, m_ppgradv;
    };

    template <class Platform, class ValueType>
    fast_waves_uv_variant<Platform, ValueType>::fast_waves_uv_variant(const arguments_map &args)
        : variant_base(args), m_u_ref(storage_size()), m_v_ref(storage_size()), m_u_out(storage_size()),
          m_v_out(storage_size()), m_u_pos(storage_size()), m_v_pos(storage_size()), m_u_tens(storage_size()),
          m_v_tens(storage_size()), m_rho(storage_size()), m_ppuv(storage_size()), m_fx(storage_size()),
          m_rho0(storage_size()), m_cwp(storage_size()), m_p0(storage_size()), m_wbbctens_stage(storage_size()),
          m_wgtfac(storage_size()), m_hhl(storage_size()), m_xlhsx(storage_size()), m_xlhsy(storage_size()),
          m_xdzdx(storage_size()), m_xdzdy(storage_size()), m_xrhsy_ref(storage_size()), m_xrhsx_ref(storage_size()),
          m_xrhsz_ref(storage_size()), m_ppgradcor(storage_size()), m_ppgradu(storage_size()),
          m_ppgradv(storage_size()) {
#pragma omp parallel
        {
            std::minstd_rand eng;
            std::uniform_real_distribution<value_type> dist(-1, 1);

            int total_size = storage_size();
#pragma omp for
            for (int i = 0; i < total_size; ++i) {
                m_u_ref.at(i) = dist(eng);
                m_v_ref.at(i) = dist(eng);
                m_u_out.at(i) = dist(eng);
                m_v_out.at(i) = dist(eng);
                m_u_pos.at(i) = dist(eng);
                m_v_pos.at(i) = dist(eng);
                m_u_tens.at(i) = dist(eng);
                m_v_tens.at(i) = dist(eng);
                m_rho.at(i) = dist(eng);
                m_ppuv.at(i) = dist(eng);
                m_fx.at(i) = dist(eng);
                m_rho0.at(i) = dist(eng);
                m_cwp.at(i) = dist(eng);
                m_p0.at(i) = dist(eng);
                m_wbbctens_stage.at(i) = dist(eng);
                m_wgtfac.at(i) = dist(eng);
                m_hhl.at(i) = dist(eng);
                m_xlhsx.at(i) = dist(eng);
                m_xlhsy.at(i) = dist(eng);
                m_xdzdx.at(i) = dist(eng);
                m_xdzdy.at(i) = dist(eng);
                m_xrhsy_ref.at(i) = dist(eng);
                m_xrhsx_ref.at(i) = dist(eng);
                m_xrhsz_ref.at(i) = dist(eng);
                m_ppgradcor.at(i) = dist(eng);
                m_ppgradu.at(i) = dist(eng);
                m_ppgradv.at(i) = dist(eng);
            }
        }
    }

    template <class Platform, class ValueType>
    std::vector<std::string> fast_waves_uv_variant<Platform, ValueType>::stencil_list() const {
        return {"fast-waves-uv"};
    }

    template <class Platform, class ValueType>
    void fast_waves_uv_variant<Platform, ValueType>::prerun() {
        variant_base::prerun();
        const int istride = this->istride();
        const int jstride = this->jstride();
        const int kstride = this->kstride();
        const int h = this->halo();
        const int isize = this->isize();
        const int jsize = this->jsize();
        const int ksize = this->ksize();

        auto fill_field = [&](ValueType *ptr,
            ValueType offset1,
            ValueType offset2,
            ValueType base1,
            ValueType base2,
            ValueType spreadx,
            ValueType spready) {
            const unsigned int i_begin = 0;
            const unsigned int i_end = isize + h * 2;
            const unsigned int j_begin = 0;
            const unsigned int j_end = jsize + h * 2;
            const unsigned int k_begin = 0;
            const unsigned int k_end = ksize;

            ValueType dx = 1. / (ValueType)(i_end - i_begin);
            ValueType dy = 1. / (ValueType)(j_end - j_begin);
            ValueType dz = 1. / (ValueType)(k_end - k_begin);

            for (int j = j_begin; j < j_end; j++) {
                for (int i = i_begin; i < i_end; i++) {
                    double x = dx * (double)(i - i_begin);
                    double y = dy * (double)(j - j_begin);
                    for (int k = k_begin; k < k_end; k++) {
                        double z = dz * (double)(k - k_begin);

                        // u values between 5 and 9
                        ptr[index(i, j, k)] = offset1 +
                                              base1 * (offset2 + cos(M_PI * (spreadx * x + spready * y)) +
                                                          base2 * sin(2 * M_PI * (spreadx * x + spready * y) * z)) /
                                                  4.;
                    }
                }
            }
        };

        fill_field(u_ref(), 3.0, 2.5, 1.25, 0.78, 18.4, 20.3);
        fill_field(v_ref(), 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
        fill_field(u_out(), 3.0, 2.5, 1.25, 0.78, 18.4, 20.3);
        fill_field(v_out(), 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
        fill_field(u_pos(), 3.4, 5.3, 0.87, 1.11, 1.4, 2.3);
        fill_field(v_pos(), 2.4, 1.3, 0.77, 1.11, 1.4, 2.3);
        fill_field(u_tens(), 4.3, 0.3, 0.97, 1.11, 1.4, 2.3);
        fill_field(v_tens(), 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
        fill_field(ppuv(), 1.4, 5.3, 0.87, 1.11, 1.4, 2.3);
        fill_field(rho(), 1.4, 4.3, 0.87, 1.11, 1.4, 2.3);
        fill_field(rho0(), 3.4, 1.3, 0.87, 1.11, 1.4, 2.3);
        fill_field(p0(), 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
        fill_field(hhl(), 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
        fill_field(wgtfac(), 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
        fill_field(fx(), 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
        fill_field(cwp(), 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
        fill_field(xdzdx(), 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
        fill_field(xdzdy(), 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
        fill_field(xlhsx(), 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
        fill_field(xlhsy(), 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
        fill_field(wbbctens_stage(), 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    }

    template <class Platform, class ValueType>
    std::function<void()> fast_waves_uv_variant<Platform, ValueType>::stencil_function(const std::string &stencil) {
        if (stencil == "fast-waves-uv")
            return std::bind(&fast_waves_uv_variant::fast_waves_uv, this);
        throw ERROR("unknown stencil '" + stencil + "'");
    }

    template <class Platform, class ValueType>
    bool fast_waves_uv_variant<Platform, ValueType>::verify(const std::string &stencil) {
        if (stencil != "fast-waves-uv")
            throw ERROR("unknown stencil '" + stencil + "'");

        const ValueType dt_small = 10;
        const ValueType edadlat = 1;

        int total_size = storage_size();

        ValueType *frefUField = u_ref();
        ValueType *frefVField = v_ref();

        ValueType *fuin = u_out();
        ValueType *fvin = v_out();
        ValueType *fupos = u_pos();
        ValueType *fvpos = v_pos();

        ValueType *futensstage = u_tens();
        ValueType *fvtensstage = v_tens();

        ValueType *frho = rho();
        ValueType *fppuv = ppuv();
        ValueType *ffx = fx();

        ValueType *frho0 = rho0();
        ValueType *fcwp = cwp();
        ValueType *fp0 = p0();
        ValueType *fwbbctens_stage = wbbctens_stage();
        ValueType *fwgtfac = wgtfac();
        ValueType *fhhl = hhl();
        ValueType *fxlhsx = xlhsx();
        ValueType *fxlhsy = xlhsy();
        ValueType *fxdzdx = xdzdx();
        ValueType *fxdzdy = xdzdy();

        ValueType *fxrhsy = xrhsy_ref();
        ValueType *fxrhsx = xrhsx_ref();
        ValueType *fxrhsz = xrhsz_ref();
        ValueType *fppgradcor = ppgradcor();
        ValueType *fppgradu = ppgradu();
        ValueType *fppgradv = ppgradv();

        const int cFlatLimit = 10;
        const int istride = this->istride();
        const int jstride = this->jstride();
        const int kstride = this->kstride();
        const int h = this->halo();
        const int isize = this->isize();
        const int jsize = this->jsize();
        const int ksize = this->ksize();

        auto computePPGradCor = [&](int i, int j, int k) {
            fppgradcor[index(i, j, k)] = fwgtfac[index(i, j, k)] * fppuv[index(i, j, k)] +
                                         ((ValueType)1.0 - fwgtfac[index(i, j, k)]) * fppuv[index(i, j, k - 1)];
        };

        // PPGradCorStage
        int k = cFlatLimit;
        for (int i = 0; i < isize + 1; ++i) {
            for (int j = 0; j < jsize + 1; ++j) {
                computePPGradCor(i, j, k);
            }
        }

        for (k = cFlatLimit + 1; k < ksize; ++k) {
            for (int i = 0; i < isize + 1; ++i) {
                for (int j = 0; j < jsize + 1; ++j) {
                    computePPGradCor(i, j, k);
                    fppgradcor[index(i, j, k - 1)] = (fppgradcor[index(i, j, k)] - fppgradcor[index(i, j, k - 1)]);
                }
            }
        }

        // XRHSXStage
        // FullDomain
        k = ksize - 1;
        for (int i = -1; i < isize; ++i) {
            for (int j = 0; j < jsize + 1; ++j) {
                fxrhsx[index(i, j, k)] = -ffx[index(i, j, k)] /
                                             ((ValueType)0.5 * (frho[index(i, j, k)] + frho[index(i + 1, j, k)])) *
                                             (fppuv[index(i + 1, j, k)] - fppuv[index(i, j, k)]) +
                                         futensstage[index(i, j, k)];
                fxrhsy[index(i, j, k)] = -edadlat /
                                             ((ValueType)0.5 * (frho[index(i, j + 1, k)] + frho[index(i, j, k)])) *
                                             (fppuv[index(i, j + 1, k)] - fppuv[index(i, j, k)]) +
                                         fvtensstage[index(i, j, k)];
            }
        }
        for (int i = 0; i < isize + 1; ++i) {
            for (int j = -1; j < jsize; ++j) {
                fxrhsy[index(i, j, k)] = -edadlat /
                                             ((ValueType)0.5 * (frho[index(i, j + 1, k)] + frho[index(i, j, k)])) *
                                             (fppuv[index(i, j + 1, k)] - fppuv[index(i, j, k)]) +
                                         fvtensstage[index(i, j, k)];
            }
        }
        for (int i = 0; i < isize + 1; ++i) {
            for (int j = 0; j < jsize + 1; ++j) {
                fxrhsz[index(i, j, k)] =
                    frho0[index(i, j, k)] / frho[index(i, j, k)] * 9.8 *
                        ((ValueType)1.0 - fcwp[index(i, j, k)] * (fp0[index(i, j, k)] + fppuv[index(i, j, k)])) +
                    fwbbctens_stage[index(i, j, k + 1)];
            }
        }

        // PPGradStage
        for (k = 0; k < ksize - 1; ++k) {
            for (int i = 0; i < isize; ++i) {
                for (int j = 0; j < jsize; ++j) {
                    if (k < cFlatLimit) {
                        fppgradu[index(i, j, k)] = (fppuv[index(i + 1, j, k)] - fppuv[index(i, j, k)]);
                        fppgradv[index(i, j, k)] = (fppuv[index(i, j + 1, k)] - fppuv[index(i, j, k)]);
                    } else {
                        fppgradu[index(i, j, k)] = (fppuv[index(i + 1, j, k)] - fppuv[index(i, j, k)]) +
                                                   (fppgradcor[index(i + 1, j, k)] + fppgradcor[index(i, j, k)]) *
                                                       (ValueType)0.5 *
                                                       ((fhhl[index(i, j, k + 1)] + fhhl[index(i, j, k)]) -
                                                           (fhhl[index(i + 1, j, k + 1)] + fhhl[index(i + 1, j, k)])) /
                                                       ((fhhl[index(i, j, k + 1)] - fhhl[index(i, j, k)]) +
                                                           (fhhl[index(i + 1, j, k + 1)] - fhhl[index(i + 1, j, k)]));
                        fppgradv[index(i, j, k)] = (fppuv[index(i, j + 1, k)] - fppuv[index(i, j, k)]) +
                                                   (fppgradcor[index(i, j + 1, k)] + fppgradcor[index(i, j, k)]) *
                                                       (ValueType)0.5 *
                                                       ((fhhl[index(i, j, k + 1)] + fhhl[index(i, j, k)]) -
                                                           (fhhl[index(i, j + 1, k + 1)] + fhhl[index(i, j + 1, k)])) /
                                                       ((fhhl[index(i, j, k + 1)] - fhhl[index(i, j, k)]) +
                                                           (fhhl[index(i, j + 1, k + 1)] - fhhl[index(i, j + 1, k)]));
                    }
                }
            }
        }

        // UVStage
        // FullDomain
        for (k = 0; k < ksize - 1; ++k) {
            for (int i = 0; i < isize; ++i) {
                for (int j = 0; j < jsize; ++j) {
                    ValueType rhou =
                        ffx[index(i, j, k)] / ((ValueType)0.5 * (frho[index(i + 1, j, k)] + frho[index(i, j, k)]));
                    ValueType rhov = edadlat / ((ValueType)0.5 * (frho[index(i, j + 1, k)] + frho[index(i, j, k)]));

                    frefUField[index(i, j, k)] =
                        fupos[index(i, j, k)] +
                        (futensstage[index(i, j, k)] - fppgradu[index(i, j, k)] * rhou) * dt_small;
                    frefVField[index(i, j, k)] =
                        fvpos[index(i, j, k)] +
                        (fvtensstage[index(i, j, k)] - fppgradv[index(i, j, k)] * rhov) * dt_small;
                }
            }
        }

        k = ksize - 1;

        for (int i = 0; i < isize; ++i) {
            for (int j = 0; j < jsize; ++j) {
                ValueType bottU =
                    fxlhsx[index(i, j, k)] * fxdzdx[index(i, j, k)] *
                        ((ValueType)0.5 * (fxrhsz[index(i + 1, j, k)] + fxrhsz[index(i, j, k)]) -
                            fxdzdx[index(i, j, k)] * fxrhsx[index(i, j, k)] -
                            (ValueType)0.5 *
                                ((ValueType)0.5 * (fxdzdy[index(i + 1, j - 1, k)] + fxdzdy[index(i + 1, j, k)]) +
                                    (ValueType)0.5 * (fxdzdy[index(i, j - 1, k)] + fxdzdy[index(i, j, k)])) *
                                (ValueType)0.5 *
                                ((ValueType)0.5 * (fxrhsy[index(i + 1, j - 1, k)] + fxrhsy[index(i + 1, j, k)]) +
                                    (ValueType)0.5 * (fxrhsy[index(i, j - 1, k)] + fxrhsy[index(i, j, k)]))) +
                    fxrhsx[index(i, j, k)];
                frefUField[index(i, j, k)] = fupos[index(i, j, k)] + bottU * dt_small;
                ValueType bottV =
                    fxlhsy[index(i, j, k)] * fxdzdy[index(i, j, k)] *
                        ((ValueType)0.5 * (fxrhsz[index(i, j + 1, k)] + fxrhsz[index(i, j, k)]) -
                            fxdzdy[index(i, j, k)] * fxrhsy[index(i, j, k)] -
                            (ValueType)0.5 *
                                ((ValueType)0.5 * (fxdzdx[index(i - 1, j + 1, k)] + fxdzdx[index(i, j + 1, k)]) +
                                    (ValueType)0.5 * (fxdzdx[index(i - 1, j, k)] + fxdzdx[index(i, j, k)])) *
                                (ValueType)0.5 *
                                ((ValueType)0.5 * (fxrhsx[index(i - 1, j + 1, k)] + fxrhsx[index(i, j + 1, k)]) +
                                    (ValueType)0.5 * (fxrhsx[index(i - 1, j, k)] + fxrhsx[index(i, j, k)]))) +
                    fxrhsy[index(i, j, k)];
                frefVField[index(i, j, k)] = fvpos[index(i, j, k)] + bottV * dt_small;
            }
        }

        auto eq = [](value_type a, value_type b) {
            value_type diff = std::abs(a - b);
            a = std::abs(a);
            b = std::abs(b);
            return diff <= (a > b ? a : b) * 1e-3;
        };

        bool success = true;
#pragma omp parallel for collapse(3) reduction(&& : success)
        for (int k = 0; k < ksize; ++k)
            for (int j = 0; j < jsize; ++j)
                for (int i = 0; i < isize; ++i) {
                    success = success && eq(u_ref()[index(i, j, k)], u_out()[index(i, j, k)]);
                    success = success && eq(v_ref()[index(i, j, k)], v_out()[index(i, j, k)]);
                }
        return success;
    }

    template <class Platform, class ValueType>
    std::size_t fast_waves_uv_variant<Platform, ValueType>::touched_elements(const std::string &stencil) const {
        if (stencil != "fast-waves-uv")
            throw ERROR("unknown stencil '" + stencil + "'");
        return -1;
    }

} // namespace platform
