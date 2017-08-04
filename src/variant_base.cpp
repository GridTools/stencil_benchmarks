#include "variant_base.h"

#include <chrono>
#include <stdexcept>

#include "arguments.h"
#include "except.h"
#include "result.h"

namespace platform {

variant_base::variant_base(const arguments_map& args)
    : m_halo(args.get<int>("halo")),
      m_alignment(args.get<int>("alignment")),
      m_isize(args.get<int>("i-size")),
      m_jsize(args.get<int>("j-size")),
      m_ksize(args.get<int>("k-size")),
      m_ilayout(args.get<int>("i-layout")),
      m_jlayout(args.get<int>("j-layout")),
      m_klayout(args.get<int>("k-layout")),
      m_data_offset(((m_halo + m_alignment - 1) / m_alignment) * m_alignment -
                    m_halo) {
  if (m_isize <= 0 || m_jsize <= 0 || m_ksize <= 0)
    throw ERROR("invalid domain size");
  if (m_halo <= 0) throw ERROR("invalid m_halo size");
  if (m_alignment <= 0) throw ERROR("invalid alignment");

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
    throw ERROR("invalid layout");
  }

  s = ((s + m_alignment - 1) / m_alignment) * m_alignment;

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
    throw ERROR("invalid layout");
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
    throw ERROR("invalid layout");
  }

  m_storage_size = m_data_offset + s;
}

result variant_base::run(const std::string& kernel, int runs) {
  using clock = std::chrono::high_resolution_clock;
  constexpr int dry = 2;

  stencil_fptr f = stencil_function(kernel);

  result res;

  for (int i = 0; i < runs + dry; ++i) {
    prerun();

    auto tstart = clock::now();
    (this->*f)();
    auto tend = clock::now();

    postrun();

    if (i == 0) {
      if (!verify(kernel))
        throw ERROR("result of kernel '" + kernel + "' is wrong");
    } else if (i >= dry) {
      double t = std::chrono::duration<double>(tend - tstart).count();
      res.push_back(t, touched_bytes(kernel) / (1024.0 * 1024.0 * 1024.0));
    }
  }

  return res;
}

std::vector<std::string> variant_base::stencil_list() {
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
  throw ERROR("unknown stencil '" + kernel + "'");
}

std::size_t variant_base::touched_elements(const std::string& kernel) const {
  std::size_t i = m_isize;
  std::size_t j = m_jsize;
  std::size_t k = m_ksize;
  if (kernel == "copy" || kernel == "copyi" || kernel == "copyj" ||
      kernel == "copyk")
    return i * j * k * 2;
  if (kernel == "avgi") return i * j * k + (i + 2) * j * k;
  if (kernel == "avgj") return i * j * k + i * (j + 2) * k;
  if (kernel == "avgk") return i * j * k + i * j * (k + 2);
  if (kernel == "sumi") return i * j * k + (i + 1) * j * k;
  if (kernel == "sumj") return i * j * k + i * (j + 1) * k;
  if (kernel == "sumk") return i * j * k + i * j * (k + 1);
  if (kernel == "lapij") return i * j * k + (i + 2) * (j + 2) * (k + 2);
  throw ERROR("unknown stencil '" + kernel + "'");
}

}  // namespace platform
