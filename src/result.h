#pragma once

#include <limits>
#include <string>
#include <vector>

class result_array {
    using lim = std::numeric_limits<double>;

  public:
    using value_type = std::vector<double>::value_type;

    result_array(const std::vector<double> &data);
    result_array(std::vector<double> &&data);
    result_array();

    double min() const;
    double max() const;
    double avg() const;

    void push_back(double d);

  private:
    friend struct result;

    std::vector<double> m_data;
};
