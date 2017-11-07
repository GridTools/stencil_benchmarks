#pragma once

#include <array>
#include <vector>

#include "except.h"

std::vector<int> prime_factors(int n);

template <int D>
std::array<int, D> threads_per_dim(int threads) {
    if (threads < 1)
        throw ERROR("not enough threads");

    std::array<int, D> ts;
    std::fill(ts.begin(), ts.end(), 1);
    std::vector<int> factors = prime_factors(threads);

    int idx = 0;
    for (int factor : factors) {
        ts[idx] *= factor;
        idx = (idx + 1) % D;
    }

    return ts;
}
