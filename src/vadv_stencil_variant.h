#pragma once

#include <cmath>
#include <iterator>
#include <random>

#include "except.h"
#include "variant_base.h"

namespace platform {

    template <class Platform, class ValueType>
    class vadv_stencil_variant : public variant_base {
      public:
        using platform = Platform;
        using value_type = ValueType;
        using allocator = typename platform::template allocator<value_type>;

        vadv_stencil_variant(const arguments_map &args);
        virtual ~vadv_stencil_variant() {}

        std::vector<std::string> stencil_list() const override;

        void prerun_init() override;
        void prerun() override;

        virtual void vadv() = 0;

      protected:
        value_type *ustage() { return m_ustage.data() + zero_offset(); }
        value_type *upos() { return m_upos.data() + zero_offset(); }
        value_type *utens() { return m_utens.data() + zero_offset(); }
        value_type *utensstage() { return m_utensstage.data() + zero_offset(); }
        value_type *vstage() { return m_vstage.data() + zero_offset(); }
        value_type *vpos() { return m_vpos.data() + zero_offset(); }
        value_type *vtens() { return m_vtens.data() + zero_offset(); }
        value_type *vtensstage() { return m_vtensstage.data() + zero_offset(); }
        value_type *wstage() { return m_wstage.data() + zero_offset(); }
        value_type *wpos() { return m_wpos.data() + zero_offset(); }
        value_type *wtens() { return m_wtens.data() + zero_offset(); }
        value_type *wtensstage() { return m_wtensstage.data() + zero_offset(); }
        value_type *ccol() { return m_ccol.data() + zero_offset(); }
        value_type *dcol() { return m_dcol.data() + zero_offset(); }
        value_type *wcon() { return m_wcon.data() + zero_offset(); }
        value_type *datacol() { return m_datacol.data() + zero_offset(); }
        value_type *utensstage_ref() { return m_utensstage_ref.data() + zero_offset(); }
        value_type *vtensstage_ref() { return m_vtensstage_ref.data() + zero_offset(); }
        value_type *wtensstage_ref() { return m_wtensstage_ref.data() + zero_offset(); }

        std::function<void()> stencil_function(const std::string &stencil) override;

        bool verify(const std::string &stencil) override;

        std::size_t touched_elements(const std::string &stencil) const override;
        std::size_t bytes_per_element() const override { return sizeof(value_type); }

      private:
        std::vector<value_type, allocator> m_ustage, m_upos, m_utens, m_utensstage;
        std::vector<value_type, allocator> m_vstage, m_vpos, m_vtens, m_vtensstage;
        std::vector<value_type, allocator> m_wstage, m_wpos, m_wtens, m_wtensstage;
        std::vector<value_type, allocator> m_ccol, m_dcol, m_wcon, m_datacol;
        std::vector<value_type> m_utensstage_ref, m_vtensstage_ref, m_wtensstage_ref;
    };

    template <class Platform, class ValueType>
    vadv_stencil_variant<Platform, ValueType>::vadv_stencil_variant(const arguments_map &args)
        : variant_base(args), m_ustage(storage_size()), m_upos(storage_size()), m_utens(storage_size()),
          m_utensstage(storage_size()), m_vstage(storage_size()), m_vpos(storage_size()), m_vtens(storage_size()),
          m_vtensstage(storage_size()), m_wstage(storage_size()), m_wpos(storage_size()), m_wtens(storage_size()),
          m_wtensstage(storage_size()), m_ccol(storage_size()), m_dcol(storage_size()), m_wcon(storage_size()),
          m_datacol(storage_size()), m_utensstage_ref(storage_size()), m_vtensstage_ref(storage_size()),
          m_wtensstage_ref(storage_size()) {
    }

    template <class Platform, class ValueType>
    void vadv_stencil_variant<Platform, ValueType>::prerun_init() {
        const int isize = this->isize();
        const int jsize = this->jsize();
        const int ksize = this->ksize();
        const int h = this->halo();
        const value_type dx = 1.0 / isize;
        const value_type dy = 1.0 / jsize;
        const value_type dz = 1.0 / ksize;

        value_type x, y, z;

        auto val = [&](value_type offset1,
            value_type offset2,
            value_type base1,
            value_type base2,
            value_type ispread,
            value_type jspread) {
            return offset1 +
                   base1 * (offset2 + std::cos(M_PI * (ispread * x + ispread * y)) +
                               base2 * std::sin(2 * M_PI * (ispread * x + jspread * y) * z)) /
                       4.0;
        };

#pragma omp parallel for collapse(3)
        for (int k = -h; k < ksize + h; ++k)
            for (int j = -h; j < jsize + h; ++j)
                for (int i = -h; i < isize + h; ++i) {
                    const int idx = this->zero_offset() + this->index(i, j, k);
                    x = i * dx;
                    y = j * dy;
                    z = k * dz;
                    m_ustage.at(idx) = val(2.2, 1.5, 0.95, 1.18, 18.4, 20.3);
                    m_upos.at(idx) = val(3.4, 0.7, 1.07, 1.51, 1.4, 2.3);
                    m_utens.at(idx) = val(7.4, 4.3, 1.17, 0.91, 1.4, 2.3);
                    m_utensstage.at(idx) = val(3.2, 2.5, 0.95, 1.18, 18.4, 20.3);

                    m_vstage.at(idx) = val(2.3, 1.5, 0.95, 1.14, 18.4, 20.3);
                    m_vpos.at(idx) = val(3.3, 0.7, 1.07, 1.71, 1.4, 2.3);
                    m_vtens.at(idx) = val(7.3, 4.3, 1.17, 0.71, 1.4, 2.3);
                    m_vtensstage.at(idx) = val(3.3, 2.4, 0.95, 1.18, 18.4, 20.3);

                    m_wstage.at(idx) = val(2.3, 1.5, 0.95, 1.14, 18.4, 20.3);
                    m_wpos.at(idx) = val(3.3, 0.7, 1.07, 1.71, 1.4, 2.3);
                    m_wtens.at(idx) = val(7.3, 4.3, 1.17, 0.71, 1.4, 2.3);
                    m_wtensstage.at(idx) = val(3.3, 2.4, 0.95, 1.18, 18.4, 20.3);

                    m_wcon.at(idx) = val(1.3, 0.3, 0.87, 1.14, 1.4, 2.3);

                    m_ccol.at(idx) = -1;
                    m_dcol.at(idx) = -1;
                    m_datacol.at(idx) = -1;
                }
    }

    template <class Platform, class ValueType>
    std::vector<std::string> vadv_stencil_variant<Platform, ValueType>::stencil_list() const {
        return {"vadv"};
    }

    template <class Platform, class ValueType>
    void vadv_stencil_variant<Platform, ValueType>::prerun() {
        variant_base::prerun();
        int total_size = storage_size();
#pragma omp parallel for
        for (int i = 0; i < total_size; ++i) {
            m_utensstage_ref.at(i) = m_utensstage.at(i);
            m_vtensstage_ref.at(i) = m_vtensstage.at(i);
            m_wtensstage_ref.at(i) = m_wtensstage.at(i);
            m_ccol.at(i) = -1;
            m_dcol.at(i) = -1;
            m_datacol.at(i) = -1;
        }
    }

    template <class Platform, class ValueType>
    std::function<void()> vadv_stencil_variant<Platform, ValueType>::stencil_function(const std::string &stencil) {
        if (stencil == "vadv")
            return std::bind(&vadv_stencil_variant::vadv, this);
        throw ERROR("unknown stencil '" + stencil + "'");
    }

    template <class Platform, class ValueType>
    bool vadv_stencil_variant<Platform, ValueType>::verify(const std::string &stencil) {
        if (stencil != "vadv")
            throw ERROR("unknown stencil '" + stencil + "'");
        constexpr value_type dtr_stage = 3.0 / 20.0;
        constexpr value_type beta_v = 0;
        constexpr value_type bet_m = 0.5 * (1.0 - beta_v);
        constexpr value_type bet_p = 0.5 * (1.0 + beta_v);
        const int isize = this->isize();
        const int jsize = this->jsize();
        const int ksize = this->ksize();
        const int istride = this->istride();
        const int jstride = this->jstride();
        const int kstride = this->kstride();

        auto backward_sweep = [ksize, istride, jstride, kstride](int i,
            int j,
            value_type *ccol,
            const value_type *dcol,
            value_type *datacol,
            const value_type *upos,
            value_type *utensstage) {
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
                const int index = i * istride + j * jstride + k * kstride;
                datacol[index] = dcol[index] - ccol[index] * datacol[index + kstride];
                ccol[index] = datacol[index];
                utensstage[index] = dtr_stage * (datacol[index] - upos[index]);
            }
        };

        auto forward_sweep = [ksize, istride, jstride, kstride](int i,
            int j,
            int ishift,
            int jshift,
            value_type *ccol,
            value_type *dcol,
            const value_type *wcon,
            const value_type *ustage,
            const value_type *upos,
            const value_type *utens,
            const value_type *utensstage) {
            // k minimum
            {
                const int k = 0;
                const int index = i * istride + j * jstride + k * kstride;
                value_type gcv = value_type(0.25) *
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
                value_type gav = value_type(-0.25) * (wcon[index + ishift * istride + jshift * jstride] + wcon[index]);
                value_type gcv = value_type(0.25) *
                                 (wcon[index + ishift * istride + jshift * jstride + kstride] + wcon[index + kstride]);

                value_type as = gav * bet_m;
                value_type cs = gcv * bet_m;

                value_type acol = gav * bet_p;
                ccol[index] = gcv * bet_p;
                value_type bcol = dtr_stage - acol - ccol[index];

                value_type correction_term =
                    -as * (ustage[index - kstride] - ustage[index]) - cs * (ustage[index + kstride] - ustage[index]);
                dcol[index] = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                value_type divided = value_type(1.0) / (bcol - ccol[index - kstride] * acol);
                ccol[index] = ccol[index] * divided;
                dcol[index] = (dcol[index] - dcol[index - kstride] * acol) * divided;
            }

            // k maximum
            {
                const int k = ksize - 1;
                const int index = i * istride + j * jstride + k * kstride;
                value_type gav = value_type(-0.25) * (wcon[index + ishift * istride + jshift * jstride] + wcon[index]);

                value_type as = gav * bet_m;

                value_type acol = gav * bet_p;
                value_type bcol = dtr_stage - acol;

                value_type correction_term = -as * (ustage[index - kstride] - ustage[index]);
                dcol[index] = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                value_type divided = value_type(1.0) / (bcol - ccol[index - kstride] * acol);
                dcol[index] = (dcol[index] - dcol[index - kstride] * acol) * divided;
            }
        };

        int total_size = storage_size();
#pragma omp parallel for
        for (int i = 0; i < total_size; ++i) {
            m_ccol.at(i) = -1;
            m_dcol.at(i) = -1;
            m_datacol.at(i) = -1;
        }

// generate u
#pragma omp parallel for collapse(2)
        for (int j = 0; j < jsize; ++j)
            for (int i = 0; i < isize; ++i) {
                forward_sweep(i, j, 1, 0, ccol(), dcol(), wcon(), ustage(), upos(), utens(), utensstage_ref());
                backward_sweep(i, j, ccol(), dcol(), datacol(), upos(), utensstage_ref());
            }

// generate v
#pragma omp parallel for collapse(2)
        for (int j = 0; j < jsize; ++j)
            for (int i = 0; i < isize; ++i) {
                forward_sweep(i, j, 0, 1, ccol(), dcol(), wcon(), vstage(), vpos(), vtens(), vtensstage_ref());
                backward_sweep(i, j, ccol(), dcol(), datacol(), vpos(), vtensstage_ref());
            }

// generate w
#pragma omp parallel for collapse(2)
        for (int j = 0; j < jsize; ++j)
            for (int i = 0; i < isize; ++i) {
                forward_sweep(i, j, 0, 0, ccol(), dcol(), wcon(), wstage(), wpos(), wtens(), wtensstage_ref());
                backward_sweep(i, j, ccol(), dcol(), datacol(), wpos(), wtensstage_ref());
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
                    bool usuccess = eq(utensstage()[index(i, j, k)], utensstage_ref()[index(i, j, k)]);
                    bool vsuccess = eq(vtensstage()[index(i, j, k)], vtensstage_ref()[index(i, j, k)]);
                    bool wsuccess = eq(wtensstage()[index(i, j, k)], wtensstage_ref()[index(i, j, k)]);
                    success = success && usuccess && vsuccess && wsuccess;
                }
        return success;
    }

    template <class Platform, class ValueType>
    std::size_t vadv_stencil_variant<Platform, ValueType>::touched_elements(const std::string &stencil) const {
        if (stencil != "vadv")
            throw ERROR("unknown stencil '" + stencil + "'");
        std::size_t i = isize();
        std::size_t j = jsize();
        std::size_t k = ksize();
        // TODO: better estimate
        return i * j * k * 16;
    }

} // namespace platform
