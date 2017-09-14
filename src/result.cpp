#include <algorithm>
#include <limits>
#include <numeric>
#include <ostream>

#include "result.h"
#include "table.h"

result_array::result_array(const std::vector<double> &data) : m_data(data) {}

result_array::result_array(std::vector<double> &&data) : m_data(data) {}

result_array::result_array() { m_data.reserve(20); }

double result_array::min() const {
    return std::accumulate(
        m_data.begin(), m_data.end(), lim::infinity(), [](double a, double b) { return a < b ? a : b; });
}

double result_array::max() const {
    return std::accumulate(
        m_data.begin(), m_data.end(), -lim::infinity(), [](double a, double b) { return a > b ? a : b; });
}

double result_array::avg() const { return std::accumulate(m_data.begin(), m_data.end(), 0.0) / m_data.size(); }

void result_array::push_back(double d) { m_data.push_back(d); }
