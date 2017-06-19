#pragma once

inline unsigned int initial_offset(const unsigned int first_padding, const unsigned int halo, const unsigned jstride, const unsigned kstride) {
    return first_padding + halo + halo*jstride + halo*kstride;
}

inline unsigned int index(const unsigned int i, const unsigned int j, const unsigned jstride, const unsigned first_padding) {
    return i+j*jstride + first_padding;
}

