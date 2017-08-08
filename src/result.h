#pragma once

#include <limits>

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
    void push_back(double t, double gb);

    result_array time, bandwidth;
};

std::ostream &operator<<(std::ostream &out, const result &r);
