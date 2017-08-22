#pragma once

#include <chrono>
#include <functional>
#include <stdexcept>

#include "arguments.h"
#include "result.h"

namespace platform {

    class variant_base {
      public:
        variant_base(const arguments_map &args);
        virtual ~variant_base() {}

        virtual std::vector<std::string> stencil_list() const = 0;
        std::vector<result> run(const std::string &kernel, int runs = 20);

      protected:
        using stencil_fptr = void (variant_base::*)();

        inline int index(int i, int j, int k) const { return i * m_istride + j * m_jstride + k * m_kstride; }

        inline int zero_offset() const { return m_data_offset + index(m_halo, m_halo, m_halo); }

        inline int halo() const { return m_halo; }
        inline int isize() const { return m_isize; }
        inline int jsize() const { return m_jsize; }
        inline int ksize() const { return m_ksize; }
        inline int ilayout() const { return m_ilayout; }
        inline int jlayout() const { return m_jlayout; }
        inline int klayout() const { return m_klayout; }
        inline int istride() const { return m_istride; }
        inline int jstride() const { return m_jstride; }
        inline int kstride() const { return m_kstride; }
        inline int storage_size() const { return m_storage_size; }

        virtual std::function<void()> stencil_function(const std::string &kernel) = 0;

        virtual void prerun() {}
        virtual void postrun() {}
        virtual bool verify(const std::string &kernel) = 0;

        virtual std::size_t touched_elements(const std::string &stencil) const = 0;
        virtual std::size_t bytes_per_element() const = 0;

      private:
        std::size_t touched_bytes(const std::string &stencil) const {
            return touched_elements(stencil) * bytes_per_element();
        }

        int m_halo, m_alignment;
        int m_isize, m_jsize, m_ksize;
        int m_ilayout, m_jlayout, m_klayout;
        int m_istride, m_jstride, m_kstride;
        int m_data_offset, m_storage_size;
#ifdef WITH_PAPI
        int m_papi_event_code;
#endif
    };

} // namespace platform
