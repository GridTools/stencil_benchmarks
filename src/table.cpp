#include <cmath>

#include "table.h"

std::ostream &operator<<(std::ostream &out, const table &t) {
    std::vector< std::size_t > widths(t.m_cols);
    for (std::size_t i = 0; i < t.m_out.size(); ++i) {
        const int col = i % t.m_cols;
        widths[col] = std::max(widths[col], std::get< 1 >(t.m_out[i]).size());
    }
    for (std::size_t i = 0; i < t.m_out.size(); ++i) {
        const int col = i % t.m_cols;
        out << std::setw(widths[col]);
        if (std::get< 0 >(t.m_out[i]) == table::align::left)
            out << std::left;
        else
            out << std::right;
        out << std::get< 1 >(t.m_out[i]);

        if (col == t.m_cols - 1)
            out << std::endl;
        else
            out << "  ";
    }
    if (t.m_out.size() % t.m_cols != 0)
        out << std::endl;
    return out;
}
