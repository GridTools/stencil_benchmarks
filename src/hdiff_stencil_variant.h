#pragma once

#include <iterator>
#include <random>

#include "except.h"
#include "variant_base.h"

namespace platform {

    template <class Platform, class ValueType>
    class hdiff_stencil_variant : public variant_base {
      public:
        using platform = Platform;
        using value_type = ValueType;
        using allocator = typename platform::template allocator<value_type>;

        hdiff_stencil_variant(const arguments_map &args);
        virtual ~hdiff_stencil_variant() {}

        std::vector<std::string> stencil_list() const override;

        void prerun() override;

        virtual void hdiff() = 0;

      protected:
        value_type *in() { return m_in.data() + zero_offset(); }
        value_type *coeff() { return m_coeff.data() + zero_offset(); }

        value_type *out() { return m_out.data() + zero_offset(); }
        value_type *lap() { return m_lap.data() + zero_offset(); }
        value_type *flx() { return m_flx.data() + zero_offset(); }
        value_type *fly() { return m_fly.data() + zero_offset(); }

        value_type *out_ref() { return m_out_ref.data() + zero_offset(); }
        value_type *lap_ref() { return m_lap_ref.data() + zero_offset(); }
        value_type *flx_ref() { return m_flx_ref.data() + zero_offset(); }
        value_type *fly_ref() { return m_fly_ref.data() + zero_offset(); }

        std::function<void()> stencil_function(const std::string &stencil) override;

        bool verify(const std::string &stencil) override;

        std::size_t touched_elements(const std::string &stencil) const override;
        std::size_t bytes_per_element() const override { return sizeof(value_type); }

      private:
        std::vector<value_type, allocator> m_in, m_coeff;
        std::vector<value_type, allocator> m_lap, m_flx, m_fly, m_out;
        std::vector<value_type> m_lap_ref, m_flx_ref, m_fly_ref, m_out_ref;
    };

    template <class Platform, class ValueType>
    hdiff_stencil_variant<Platform, ValueType>::hdiff_stencil_variant(const arguments_map &args)
        : variant_base(args), m_in(storage_size()), m_coeff(storage_size()), m_lap(storage_size()),
          m_flx(storage_size()), m_fly(storage_size()), m_out(storage_size()), m_lap_ref(storage_size()),
          m_flx_ref(storage_size()), m_fly_ref(storage_size()), m_out_ref(storage_size()) {
#pragma omp parallel
        {
            std::minstd_rand eng;
            std::uniform_real_distribution<value_type> dist(-1, 1);

            int total_size = storage_size();
#pragma omp for
            for (int i = 0; i < total_size; ++i) {
                m_in.at(i) = dist(eng);
                m_out.at(i) = dist(eng);
                m_coeff.at(i) = dist(eng);
                m_lap.at(i) = dist(eng);
                m_flx.at(i) = dist(eng);
                m_fly.at(i) = dist(eng);
                m_out_ref.at(i) = dist(eng);
                m_flx_ref.at(i) = dist(eng);
                m_fly_ref.at(i) = dist(eng);
                m_lap_ref.at(i) = dist(eng);
            }
        }
    }

    template <class Platform, class ValueType>
    std::vector<std::string> hdiff_stencil_variant<Platform, ValueType>::stencil_list() const {
        return {"hdiff"};
    }

    template <class Platform, class ValueType>
    void hdiff_stencil_variant<Platform, ValueType>::prerun() {
        variant_base::prerun();
        int total_size = storage_size();
        int cnt = 0;
        double dx = 1. / (double)(isize());
        double dy = 1. / (double)(jsize());
        double dz = 1. / (double)(ksize());
        for (int j = 0; j < isize(); j++) {
            for (int i = 0; i < jsize(); i++) {
                double x = dx * (double)(i);
                double y = dy * (double)(j);
                for (int k = 0; k < ksize(); k++) {
                    double z = dz * (double)(k);
                    // u values between 5 and 9
                    m_in[cnt] = 3.0 +
                                1.25 * (2.5 + cos(M_PI * (18.4 * x + 20.3 * y)) +
                                           0.78 * sin(2 * M_PI * (18.4 * x + 20.3 * y) * z)) /
                                    4.;
                    m_coeff[cnt] = 1.4 +
                                   0.87 * (0.3 + cos(M_PI * (1.4 * x + 2.3 * y)) +
                                              1.11 * sin(2 * M_PI * (1.4 * x + 2.3 * y) * z)) /
                                       4.;
                    m_out[cnt] = 5.4;
                    m_out_ref[cnt] = 5.4;
                    m_flx[cnt] = 0.0;
                    m_fly[cnt] = 0.0;
                    m_lap[cnt] = 0.0;
                    m_flx_ref[cnt] = 0.0;
                    m_fly_ref[cnt] = 0.0;
                    m_lap_ref[cnt] = 0.0;
                    cnt++;
                }
            }
        }
    }

    template <class Platform, class ValueType>
    std::function<void()> hdiff_stencil_variant<Platform, ValueType>::stencil_function(const std::string &stencil) {
        if (stencil == "hdiff")
            return std::bind(&hdiff_stencil_variant::hdiff, this);
        throw ERROR("unknown stencil '" + stencil + "'");
    }

    template <class Platform, class ValueType>
    bool hdiff_stencil_variant<Platform, ValueType>::verify(const std::string &stencil) {
        if (stencil != "hdiff")
            throw ERROR("unknown stencil '" + stencil + "'");

        const int istride = this->istride();
        const int jstride = this->jstride();
        const int kstride = this->kstride();
        const int h = this->halo();
        const int isize = this->isize();
        const int jsize = this->jsize();
        const int ksize = this->ksize();

        for (int k = 0; k < ksize; ++k) {
            for (int j = -1; j < jsize + 1; ++j) {
                for (int i = -1; i < isize + 1; ++i) {
                    lap_ref()[index(i, j, k)] =
                        4 * in()[index(i, j, k)] - (in()[index(i - 1, j, k)] + in()[index(i + 1, j, k)] +
                                                       in()[index(i, j - 1, k)] + in()[index(i, j + 1, k)]);
                }
            }

            for (int j = 0; j < jsize; ++j) {
                for (int i = -1; i < isize; ++i) {
                    flx_ref()[index(i, j, k)] = lap_ref()[index(i + 1, j, k)] - lap_ref()[index(i, j, k)];
                    if (flx_ref()[index(i, j, k)] * (in()[index(i + 1, j, k)] - in()[index(i, j, k)]) > 0)
                        flx_ref()[index(i, j, k)] = 0.;
                }
            }

            for (int j = -1; j < jsize; ++j) {
                for (int i = 0; i < isize; ++i) {
                    fly_ref()[index(i, j, k)] = lap_ref()[index(i, j + 1, k)] - lap_ref()[index(i, j, k)];
                    if (fly_ref()[index(i, j, k)] * (in()[index(i, j + 1, k)] - in()[index(i, j, k)]) > 0)
                        fly_ref()[index(i, j, k)] = 0.;
                }
            }

            for (int i = 0; i < isize; ++i) {
                for (int j = 0; j < jsize; ++j) {
                    out_ref()[index(i, j, k)] =
                        in()[index(i, j, k)] -
                        coeff()[index(i, j, k)] * (flx_ref()[index(i, j, k)] - flx_ref()[index(i - 1, j, k)] +
                                                      fly_ref()[index(i, j, k)] - fly_ref()[index(i, j - 1, k)]);
                }
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
                for (int i = 0; i < isize; ++i)
                    success = success && eq(out_ref()[index(i, j, k)], out()[index(i, j, k)]);
        return success;
    }

    template <class Platform, class ValueType>
    std::size_t hdiff_stencil_variant<Platform, ValueType>::touched_elements(const std::string &stencil) const {
        if (stencil != "hdiff")
            throw ERROR("unknown stencil '" + stencil + "'");
        std::size_t i = isize();
        std::size_t j = jsize();
        std::size_t k = ksize();
        // TODO: better estimate
        return i * j * k * 6;
    }

} // namespace platform
