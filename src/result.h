#pragma once

#include <limits>
#include <string>
#include <vector>

class result_array {
    using lim = std::numeric_limits<double>;

  public:
    double min() const;
    double max() const;
    double avg() const;

  private:
    friend struct result;

    std::vector<double> m_data;
};

struct result {
    result() = default;
    explicit result(const std::string &stencil);

    void push_back(double t, double gb, double ctr, double ctr_imb);

    std::string stencil;
    result_array time, bandwidth, counter, counter_imbalance;
};

std::ostream &operator<<(std::ostream &out, const result &r);
