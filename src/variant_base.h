#pragma once

#include <chrono>
#include <stdexcept>

#include "arguments.h"
#include "result.h"

namespace platform {

class variant_base {
  using stencil_fptr = void (variant_base::*)();

 public:
  variant_base(const arguments_map& args)
      : m_isize(args.get<int>("i-size")),
        m_jsize(args.get<int>("j-size")),
        m_ksize(args.get<int>("k-size")),
        m_ilayout(args.get<int>("i-layout")),
        m_jlayout(args.get<int>("j-layout")),
        m_klayout(args.get<int>("k-layout")),
        m_halo(args.get<int>("halo")),
        m_pad(args.get<int>("padding")) {
    if (m_isize <= 0 || m_jsize <= 0 || m_ksize <= 0)
      throw std::logic_error("invalid domain size");
    if (m_halo <= 0) throw std::logic_error("invalid m_halo size");
    if (m_pad <= 0) throw std::logic_error("invalid padding");

    int ish = m_isize + 2 * m_halo;
    int jsh = m_jsize + 2 * m_halo;
    int ksh = m_ksize + 2 * m_halo;

    int s = 1;
    if (m_ilayout == 2) {
      m_istride = s;
      s *= ish;
    } else if (m_jlayout == 2) {
      m_jstride = s;
      s *= jsh;
    } else if (m_klayout == 2) {
      m_kstride = s;
      s *= ksh;
    } else {
      throw std::logic_error("invalid layout");
    }

    s = ((s + m_pad - 1) / m_pad) * m_pad;

    if (m_ilayout == 1) {
      m_istride = s;
      s *= ish;
    } else if (m_jlayout == 1) {
      m_jstride = s;
      s *= jsh;
    } else if (m_klayout == 1) {
      m_kstride = s;
      s *= ksh;
    } else {
      throw std::logic_error("invalid layout");
    }

    if (m_ilayout == 0) {
      m_istride = s;
      s *= ish;
    } else if (m_jlayout == 0) {
      m_jstride = s;
      s *= jsh;
    } else if (m_klayout == 0) {
      m_kstride = s;
      s *= ksh;
    } else {
      throw std::logic_error("invalid layout");
    }

    m_storage_size = s;
  }

  result run(const std::string& kernel, int runs) {
    using clock = std::chrono::high_resolution_clock;

    stencil_fptr f = stencil_function(kernel);

    result res;

    for (int i = 0; i < runs + 1; ++i) {
      prerun();

      auto tstart = clock::now();
      (this->*f)();
      auto tend = clock::now();

      postrun();

      if (i == 0) {
        verify(kernel);
      } else {
        double t = std::chrono::duration<double>(tend - tstart).count();
        res.push_back(t, bytes(kernel) / (1024.0 * 1024.0 * 1024.0));
      }
    }

    return res;
  }

  virtual void prerun() = 0;
  virtual void postrun() = 0;

  virtual void copy() = 0;

 protected:
  int index(int i, int j, int k) const {
    return i * m_istride + j * m_jstride + k * m_kstride;
  }

  int zero_offset() const { return index(m_halo, m_halo, m_halo); }

  int isize() const { return m_isize; }
  int jsize() const { return m_jsize; }
  int ksize() const { return m_ksize; }
  int ilayout() const { return m_ilayout; }
  int jlayout() const { return m_jlayout; }
  int klayout() const { return m_klayout; }
  int istride() const { return m_istride; }
  int jstride() const { return m_jstride; }
  int kstride() const { return m_kstride; }
  int storage_size() const { return m_storage_size; }

  virtual bool verify(const std::string& kernel) const = 0;
  virtual std::size_t bytes(const std::string& kernel) const = 0;

  template <class F>
  bool verify_loop(F f) const {
    bool success = true;
    for (int k = 0; k < m_ksize; ++k)
      for (int j = 0; j < m_jsize; ++j)
        for (int i = 0; i < m_isize; ++i) success &= f(i, j, k);
    return success;
  }

 private:
  int m_isize, m_jsize, m_ksize;
  int m_ilayout, m_jlayout, m_klayout;
  int m_istride, m_jstride, m_kstride;
  int m_halo, m_pad, m_storage_size;

  stencil_fptr stencil_function(const std::string& kernel) {
    if (kernel == "copy") return &variant_base::copy;
    throw std::logic_error("Error: unknown stencil '" + kernel + "'");
  }
};

}  // namespace platform
