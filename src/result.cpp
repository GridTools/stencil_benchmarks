#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <ostream>

#include "result.h"
#include "table.h"

result_array::result_array(const std::string &name, const std::string &unit) : m_name(name), m_unit(unit) {}

double result_array::min() const {
    return std::accumulate(
        m_data.begin(), m_data.end(), lim::infinity(), [](double a, double b) { return a < b ? a : b; });
}

double result_array::max() const {
    return std::accumulate(
        m_data.begin(), m_data.end(), -lim::infinity(), [](double a, double b) { return a > b ? a : b; });
}

double result_array::avg() const { return std::accumulate(m_data.begin(), m_data.end(), 0.0) / m_data.size(); }

double result_array::stdev() const {
    double mean = avg();
    double sum = 0;
    for (double d : m_data)
        sum += (d - mean) * (d - mean);
    return std::sqrt(sum / m_data.size());
}

result::result(const std::string &stencil) : m_stencil(stencil) {}

void result::add_data(const std::string &name, const std::string &unit) {
    for (const auto &d : m_data) {
        if (!d.m_data.empty())
            throw ERROR("can only add additional data to empty result object");
    }
    m_data.emplace_back(name, unit);
}

const result_array &result::operator[](const std::string &name) const {
    for (const auto &d : m_data) {
        if (d.name() == name)
            return d;
    }
    throw ERROR("could not find result data for '" + name + "'");
}

std::ostream &operator<<(std::ostream &out, const result &r) {
    out << "Result for stencil '" << r.stencil() << "':\n";

    table t(5);
    t << "Metric"
      << "Unit"
      << "Average"
      << "Minimum"
      << "Maximum";

    for (auto &ri : r)
        t << ri.name() << ri.unit() << ri.avg() << ri.min() << ri.max();

    out << t;
    return out;
}
