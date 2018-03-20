#pragma once

#include <cmath>
#include <limits>
#include <random>

#include "data_field.h"
#include "defs.h"
#include "except.h"
#include "variant_base.h"

namespace platform {

    template <class Platform, class ValueType>
    class irrstrgrid_stencil_variant : public variant_base {
      public:
        using platform = Platform;
        using value_type = ValueType;
        using allocator = typename platform::template allocator<value_type>;

        irrstrgrid_stencil_variant(const arguments_map &args);
        virtual ~irrstrgrid_stencil_variant() {}

        std::vector<std::string> stencil_list() const override;

        virtual void copyu_ilp(unsigned int) = 0;
        virtual void copyu(unsigned int) = 0;
        virtual void on_cells_ilp(unsigned int) = 0;
        virtual void on_cells(unsigned int) = 0;

      protected:
        value_type *src_data(unsigned int field = 0) {
            return m_src_data.at(field % num_storages_per_field).m_data.data();
        }
        value_type *dst_data(unsigned int field = 0) {
            return m_dst_data.at(field % num_storages_per_field).m_data.data();
        }

        value_type *src(unsigned int field = 0) {
            return m_src_data.at(field % num_storages_per_field).m_data.data() + zero_offset();
        }
        value_type *dst(unsigned int field = 0) {
            return m_dst_data.at(field % num_storages_per_field).m_data.data() + zero_offset();
        }

        std::function<void(unsigned int)> stencil_function(const std::string &stencil) override;

        bool verify(const std::string &stencil) override;

        std::size_t touched_elements(const std::string &stencil) const override;
        std::size_t bytes_per_element() const override { return sizeof(value_type); }
        const unsigned int num_storages_per_field;

        void initialize_fields();
        void setup() override;

      private:
        std::vector<data_field<value_type, allocator>> m_src_data, m_dst_data;
        value_type *m_src, *m_dst;
    };

    template <class Platform, class ValueType>
    irrstrgrid_stencil_variant<Platform, ValueType>::irrstrgrid_stencil_variant(const arguments_map &args)
        : variant_base(args),
          num_storages_per_field(std::max(1, (int)(CACHE_SIZE / (storage_size() * sizeof(value_type))))),
          m_src_data(num_storages_per_field, storage_size()), m_dst_data(num_storages_per_field, storage_size()) {
        initialize_fields();
    }

    template <class Platform, class ValueType>
    void irrstrgrid_stencil_variant<Platform, ValueType>::initialize_fields() {
#pragma omp parallel
        {
            std::minstd_rand eng;
            std::uniform_real_distribution<value_type> dist(-100, 100);

            int total_size = storage_size();
            for (int f = 0; f < num_storages_per_field; ++f) {
                auto src_field = src_data(f);
                auto dst_field = dst_data(f);
#pragma omp for
                for (int i = 0; i < total_size; ++i) {
                    src_field[i] = dist(eng);
                    dst_field[i] = -1;
                }
            }
        }
    }

    template <class Platform, class ValueType>
    void irrstrgrid_stencil_variant<Platform, ValueType>::setup() {
        initialize_fields();
        variant_base::setup();
    }

    template <class Platform, class ValueType>
    std::vector<std::string> irrstrgrid_stencil_variant<Platform, ValueType>::stencil_list() const {
        return {"copyu_ilp", "copyu", "on_cells_ilp", "on_cells"};
    }

    template <class Platform, class ValueType>
    std::function<void(unsigned int)> irrstrgrid_stencil_variant<Platform, ValueType>::stencil_function(
        const std::string &stencil) {
        if (stencil == "copyu_ilp")
            return std::bind(&irrstrgrid_stencil_variant::copyu_ilp, this, std::placeholders::_1);
        else if (stencil == "copyu")
            return std::bind(&irrstrgrid_stencil_variant::copyu, this, std::placeholders::_1);
        else if (stencil == "on_cells_ilp")
            return std::bind(&irrstrgrid_stencil_variant::on_cells_ilp, this, std::placeholders::_1);
        else if (stencil == "on_cells")
            return std::bind(&irrstrgrid_stencil_variant::on_cells, this, std::placeholders::_1);
        throw ERROR("unknown stencil '" + stencil + "'");
    }

    template <class Platform, class ValueType>
    bool irrstrgrid_stencil_variant<Platform, ValueType>::verify(const std::string &stencil) {
        std::function<bool(int, int, int)> f;
        auto s = [&](int i, int j, int k) { return src()[index(i, j, k)]; };
        auto d = [&](int i, int j, int k) { return dst()[index(i, j, k)]; };

        if (stencil == "copyu_ilp" || stencil == "copyu") {
            f = [&](int i, int j, int k) { return d(i, j, k) == s(i, j, k); };
        } else if (stencil == "on_cells_ilp" || stencil == "on_cells") {

            f = [&](int i, int j, int k) {
                return (j % 2 == 0) ? (d(i, j, k) == s(i - 1, j + 1, k) + s(i, j + 1, k) + s(i, j - 1, k))
                                    : (d(i, j, k) == s(i, j - 1, k) + s(i + 1, j - 1, k) + s(i, j + 1, k));
            };
        } else {
            throw ERROR("unknown stencil '" + stencil + "'");
        }

        const int isize = this->isize();
        const int jsize = this->jsize();
        const int ksize = this->ksize();
        bool success = true;
#pragma omp parallel for collapse(3) reduction(&& : success)
        for (int k = 0; k < ksize; ++k)
            for (int j = 0; j < jsize; ++j)
                for (int i = 0; i < isize; ++i)
                    success = success && f(i, j, k);
        return success;
    }

    template <class Platform, class ValueType>
    std::size_t irrstrgrid_stencil_variant<Platform, ValueType>::touched_elements(const std::string &stencil) const {
        std::size_t i = isize();
        std::size_t j = jsize();
        std::size_t k = ksize();
        if (stencil == "copyu_ilp" || stencil == "copyu" || stencil == "on_cells_ilp" || stencil == "on_cells")
            return i * j * k * 2;
        throw ERROR("unknown stencil '" + stencil + "'");
    }

} // platform
