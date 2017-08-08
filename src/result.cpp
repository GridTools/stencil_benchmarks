#include <algorithm>
#include <limits>

#include "result.h"
#include "table.h"

double result_array::min() const {
    return std::accumulate(
        m_data.begin(), m_data.end(), lim::infinity(), [](double a, double b) { return a < b ? a : b; });
}

double result_array::max() const {
    return std::accumulate(
        m_data.begin(), m_data.end(), -lim::infinity(), [](double a, double b) { return a > b ? a : b; });
}

double result_array::avg() const { return std::accumulate(m_data.begin(), m_data.end(), 0.0) / m_data.size(); }

void result::push_back(double t, double gb) {
    time.m_data.push_back(t);
    bandwidth.m_data.push_back(gb / t);
}

std::ostream &operator<<(std::ostream &out, const result &r) {
    table t(5);
    auto tdata = [&](const std::string &name, const std::string &unit, const result_array &a, double mul = 1) {
        t << name << unit << (a.avg() * mul) << (a.min() * mul) << (a.max() * mul);
    };

    t << "Metric"
      << "Unit"
      << "Average"
      << "Minimum"
      << "Maximum";
    tdata("Time", "ms", r.time, 1000);
    tdata("Bandwidth", "GB/s", r.bandwidth);

    out << t;
    return out;
}
