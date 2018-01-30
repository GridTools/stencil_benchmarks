#pragma once

#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "except.h"

class result_array {
    using lim = std::numeric_limits<double>;

  public:
    result_array() = default;
    result_array(const std::string &name, const std::string &unit);

    double min() const;
    double max() const;
    double avg() const;

    const std::string &name() const { return m_name; }
    const std::string &unit() const { return m_unit; }

  private:
    friend struct result;

    std::string m_name, m_unit;
    std::vector<double> m_data;
};

class result {
  public:
    using iterator = typename std::vector<result_array>::const_iterator;

    result() = default;
    explicit result(const std::string &stencil);

    void add_data(const std::string &name, const std::string &unit);

    template <class... Data>
    void push_back(const Data &... data) {
        if (sizeof...(Data) != m_data.size())
            throw ERROR("invalid number of arguments");
        push_back_impl(std::integral_constant<std::size_t, sizeof...(Data)>(), std::forward_as_tuple(data...));
    }

    const std::string &stencil() const { return m_stencil; }
    const result_array &operator[](const std::string &name) const;

    iterator begin() const { return m_data.begin(); }
    iterator end() const { return m_data.end(); }

    std::size_t size() const { return m_data.size(); }

  private:
    template <std::size_t I, class Data>
    void push_back_impl(std::integral_constant<std::size_t, I>, const Data &data) {
        push_back_impl(std::integral_constant<std::size_t, I - 1>(), data);
        m_data.at(I - 1).m_data.push_back(std::get<I - 1>(data));
    }

    template <class Data>
    void push_back_impl(std::integral_constant<std::size_t, 0>, const Data &data) {}

    std::string m_stencil;
    std::vector<result_array> m_data;
};

std::ostream &operator<<(std::ostream &out, const result &r);
