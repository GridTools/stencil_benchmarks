#pragma once

#include <algorithm>
#include <iomanip>
#include <limits>
#include <ostream>

#include "table.h"

class result_array {
  using lim = std::numeric_limits<double>;

 public:
  double min() const {
    return std::accumulate(m_data.begin(), m_data.end(), lim::infinity(),
                           [](double a, double b) { return a < b ? a : b; });
  }

  double max() const {
    return std::accumulate(m_data.begin(), m_data.end(), -lim::infinity(),
                           [](double a, double b) { return a > b ? a : b; });
  }

  double avg() const {
    return std::accumulate(m_data.begin(), m_data.end(), 0.0) / m_data.size();
  }

 private:
  friend struct result;

  std::vector<double> m_data;
};

struct result {
  void push_back(double t, double gb) {
    time.m_data.push_back(t);
    bandwidth.m_data.push_back(gb / t);
  }

  result_array time, bandwidth;
};

template <class Char, class Traits>
std::basic_ostream<Char, Traits>& operator<<(
    std::basic_ostream<Char, Traits>& out, const result& r) {
  table<std::basic_ostream<Char, Traits>> t(out, 4);
  auto tdata = [&](const std::string& name, const result_array& a,
                   double mul = 1) {
    t << name << (a.avg() * mul) << (a.min() * mul) << (a.max() * mul);
  };

  t << ""
    << "Avg"
    << "Min"
    << "Max";
  tdata("T  [ms]", r.time, 1000);
  tdata("BW [GB/s]", r.bandwidth);

  return out;
}
