#pragma once

#include "variant_base.h"

namespace platform {

template <class Platform, class ValueType>
class variant : public variant_base {
 public:
  using platform = Platform;
  using value_type = ValueType;
  using allocator = typename platform::allocator<value_type>;

  variant(const arguments_map& args)
      : variant_base(args),
        m_src_data(storage_size()),
        m_dst_data(storage_size()) {
    int imin = -m_halo, imax = m_isize + m_halo;
    int jmin = -m_halo, jmax = m_jsize + m_halo;
    int kmin = -m_halo, kmax = m_ksize + m_halo;
#pragma omp parallel for collapse(3)
    for (int k = kmin; k < kmax; ++k)
      for (int j = jmin; j < jmax; ++j)
        for (int i = imin; i < imax; ++i) {
          m_src_data.at(zero_offset() + index(i, j, k)) = index(i, j, k);
        }
    set_ptrs(src_data() + zero_offset(), dst_data() + zero_offset());
  }

  virtual ~variant() {}

 protected:
  value_type* src_data() { return m_src_data.data(); }
  value_type* dst_data() { return m_dst_data.data(); }
  inline value_type* src() {
    if (!m_src) throw ERROR("src is nullptr");
    return m_src;
  }
  inline value_type* dst() {
    if (!m_dst) throw ERROR("dst is nullptr");
    return m_dst;
  }

  void set_ptrs(value_type* src, value_type* dst) {
    m_src = src;
    m_dst = dst;
  }

 private:
  const value_type& src(int i, int j, int k) const {
    return m_src_data[zero_offset() + index(i, j, k)];
  }

  const value_type& dst(int i, int j, int k) const {
    return m_dst_data[zero_offset() + index(i, j, k)];
  }

  bool verify(const std::string& kernel) const override {
    auto equal = [](value_type a, value_type b) { return a == b; };
    if (kernel == "copy") {
      return verify_loop([&](int i, int j, int k) {
        return equal(dst(i, j, k), src(i, j, k));
      });
    }
    if (kernel == "copyi") {
      return verify_loop([&](int i, int j, int k) {
        return equal(dst(i, j, k), src(i + 1, j, k));
      });
    }
    if (kernel == "copyj") {
      return verify_loop([&](int i, int j, int k) {
        return equal(dst(i, j, k), src(i, j + 1, k));
      });
    }
    if (kernel == "copyk") {
      return verify_loop([&](int i, int j, int k) {
        return equal(dst(i, j, k), src(i, j, k + 1));
      });
    }
    if (kernel == "avgi") {
      return verify_loop([&](int i, int j, int k) {
        return equal(dst(i, j, k), src(i - 1, j, k) + src(i + 1, j, k));
      });
    }
    if (kernel == "avgj") {
      return verify_loop([&](int i, int j, int k) {
        return equal(dst(i, j, k), src(i, j - 1, k) + src(i, j + 1, k));
      });
    }
    if (kernel == "avgk") {
      return verify_loop([&](int i, int j, int k) {
        return equal(dst(i, j, k), src(i, j, k - 1) + src(i, j, k + 1));
      });
    }
    if (kernel == "sumi") {
      return verify_loop([&](int i, int j, int k) {
        return equal(dst(i, j, k), src(i, j, k) + src(i + 1, j, k));
      });
    }
    if (kernel == "sumj") {
      return verify_loop([&](int i, int j, int k) {
        return equal(dst(i, j, k), src(i, j, k) + src(i, j + 1, k));
      });
    }
    if (kernel == "sumk") {
      return verify_loop([&](int i, int j, int k) {
        return equal(dst(i, j, k), src(i, j, k) + src(i, j, k + 1));
      });
    }
    if (kernel == "lapij") {
      return verify_loop([&](int i, int j, int k) {
        return equal(dst(i, j, k), src(i, j, k) + src(i - 1, j, k) +
                                       src(i + 1, j, k) + src(i, j - 1, k) +
                                       src(i, j + 1, k));
      });
    }
    throw ERROR("unknown stencil '" + kernel + "'");
  }

  std::size_t touched_bytes(const std::string& kernel) const override {
    return touched_elements(kernel) * sizeof(value_type);
  }

  std::vector<value_type, allocator> m_src_data, m_dst_data;
  value_type *m_src, *m_dst;
};

}  // platform
