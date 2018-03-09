#pragma once

namespace platform {

    template <typename ValueType, typename Allocator>
    struct data_field {
        data_field(unsigned int size) : m_data(size) {}

        std::vector<ValueType, Allocator> m_data;
    };
}
