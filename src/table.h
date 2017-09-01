#pragma once

#include <iomanip>
#include <ostream>
#include <sstream>
#include <tuple>
#include <vector>

class table {
    enum class align { left, right };

  public:
    table(std::size_t cols, std::size_t prec = 6) : m_cols(cols), m_prec(prec) {}

    template <class Input>
    table &operator<<(const Input &i) {
        std::stringstream s;
        align a = align::left;

        if (std::is_scalar<Input>::value) {
            s << std::fixed << std::setprecision(m_prec);
            a = align::right;
        }
        s << i;
        m_out.emplace_back(a, s.str());
        return *this;
    }

    std::size_t cols() const { return m_cols; }

    friend std::ostream &operator<<(std::ostream &out, const table &t);

  private:
    std::size_t m_cols, m_prec;
    std::vector<std::tuple<align, std::string>> m_out;
};
