#include "variant_base.h"

#include <chrono>
#include <stdexcept>

#include "arguments.h"
#include "result.h"

namespace platform {

variant_base::variant_base(const arguments_map& args)
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

result variant_base::run(const std::string& kernel, int runs) {
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

std::vector<std::string> variant_base::kernel_list() {
  return {"copy", "copyi", "copyj", "copyk", "avgi", "avgj",
          "avgk", "sumi",  "sumj",  "sumk",  "lapij"};
}

variant_base::stencil_fptr variant_base::stencil_function(
    const std::string& kernel) {
  if (kernel == "copy") return &variant_base::copy;
  if (kernel == "copyi") return &variant_base::copyi;
  if (kernel == "copyj") return &variant_base::copyj;
  if (kernel == "copyk") return &variant_base::copyk;
  if (kernel == "avgi") return &variant_base::avgi;
  if (kernel == "avgj") return &variant_base::avgj;
  if (kernel == "avgk") return &variant_base::avgk;
  if (kernel == "sumi") return &variant_base::sumi;
  if (kernel == "sumj") return &variant_base::sumj;
  if (kernel == "sumk") return &variant_base::sumk;
  if (kernel == "lapij") return &variant_base::lapij;
  throw std::logic_error("Error: unknown stencil '" + kernel + "'");
}

}  // namespace platform
