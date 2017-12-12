#include <cmath>
#include <iostream>

#include "blockgen.h"

std::vector<int> prime_factors(int n) {
    if (n < 2)
        return std::vector<int>();

    std::vector<int> primes;
    std::vector<bool> sieve(n / 2, true);

    for (int i = 2; i <= std::sqrt(n / 2); ++i) {
        if (sieve[i]) {
            primes.push_back(i);
            for (std::size_t j = i * i; j < sieve.size(); j += i)
                sieve[j] = false;
        }
    }

    std::vector<int> factors;

    for (int prime : primes) {
        std::cout << prime << std::endl;
        if (prime * prime > n)
            break;
        while (n % prime == 0) {
            factors.push_back(prime);
            n /= prime;
        }
    }
    if (n > 1)
        factors.push_back(n);

    return factors;
}
