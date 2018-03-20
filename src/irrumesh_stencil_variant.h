#pragma once

#include <cmath>
#include <limits>
#include <random>

#include "data_field.h"
#include "defs.h"
#include "except.h"
#include "unstructured/mesh.hpp"
#include "variant_base.h"

namespace platform {

    template <class Platform, class ValueType>
    class irrumesh_stencil_variant : public variant_base {
      public:
        using platform = Platform;
        using value_type = ValueType;
        using allocator = typename platform::template allocator<value_type>;

        irrumesh_stencil_variant(const arguments_map &args);
        virtual ~irrumesh_stencil_variant() {}

        std::vector<std::string> stencil_list() const override;

        virtual void copy_umesh(unsigned int) = 0;
        virtual void on_cells_umesh(unsigned int) = 0;

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

        void initialize_fields();

        void setup() override;

        bool verify(const std::string &stencil) override;

        std::size_t touched_elements(const std::string &stencil) const override;
        std::size_t bytes_per_element() const override { return sizeof(value_type); }
        const unsigned int num_storages_per_field;

      private:
        std::vector<data_field<value_type, allocator>> m_src_data, m_dst_data;
        value_type *m_src, *m_dst;

      protected:
        mesh mesh_;
    };

    template <class Platform, class ValueType>
    irrumesh_stencil_variant<Platform, ValueType>::irrumesh_stencil_variant(const arguments_map &args)
        : variant_base(args),
          num_storages_per_field(std::max(1, (int)(CACHE_SIZE / (storage_size() * sizeof(value_type))))),
          m_src_data(num_storages_per_field, storage_size()), m_dst_data(num_storages_per_field, storage_size()),
          mesh_(this->isize(), this->jsize() / 2, this->halo()) {
        initialize_fields();
    }

    template <class Platform, class ValueType>
    void irrumesh_stencil_variant<Platform, ValueType>::initialize_fields() {
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
    void irrumesh_stencil_variant<Platform, ValueType>::setup() {
        initialize_fields();
        variant_base::setup();
    }
    template <class Platform, class ValueType>
    std::vector<std::string> irrumesh_stencil_variant<Platform, ValueType>::stencil_list() const {
        return {"copy_umesh", "on_cells_umesh"};
    }

    template <class Platform, class ValueType>
    std::function<void(unsigned int)> irrumesh_stencil_variant<Platform, ValueType>::stencil_function(
        const std::string &stencil) {
        if (stencil == "copy_umesh")
            return std::bind(&irrumesh_stencil_variant::copy_umesh, this, std::placeholders::_1);
        else if (stencil == "on_cells_umesh")
            return std::bind(&irrumesh_stencil_variant::on_cells_umesh, this, std::placeholders::_1);
        throw ERROR("unknown stencil '" + stencil + "'");
    }

    template <class Platform, class ValueType>
    bool irrumesh_stencil_variant<Platform, ValueType>::verify(const std::string &stencil) {
        std::function<bool(int, int)> f;
        auto s = [&](int idx, int k) { return src_data()[idx + k * kstride()]; };
        auto d = [&](int idx, int k) { return dst_data()[idx + k * kstride()]; };
        auto &table = mesh_.get_elements(location::cell).table(location::cell);

        if (stencil == "copy_umesh") {
            f = [&](int idx, int k) { return d(idx, k) == s(idx, k); };
        } else if (stencil == "on_cells_umesh") {
            f = [&](int idx, int k) {
                return (d(idx, k) ==
                        src_data()[table(idx, 0) + k * kstride()] + src_data()[table(idx, 1) + k * kstride()] +
                            src_data()[table(idx, 2) + k * kstride()]);
            };
        } else {
            throw ERROR("unknown stencil '" + stencil + "'");
        }

        const int isize = this->isize();
        const int jsize = this->jsize();
        const int ksize = this->ksize();
        bool success = true;
#pragma omp parallel for collapse(3) reduction(&& : success)
        for (int k = 0; k < ksize; ++k) {
            for (size_t idx = 0; idx < mesh_.compd_size(); ++idx) {
                success = success && f(idx, k);
            }
        }
        return success;
    }

    template <class Platform, class ValueType>
    std::size_t irrumesh_stencil_variant<Platform, ValueType>::touched_elements(const std::string &stencil) const {
        std::size_t i = isize();
        std::size_t j = jsize();
        std::size_t k = ksize();
        if (stencil == "copy_umesh" || stencil == "on_cells_umesh")
            return i * j * k * 2;
        throw ERROR("unknown stencil '" + stencil + "'");
    }

} // platform
