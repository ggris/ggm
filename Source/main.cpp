#include <chrono>
#include <iostream>

#include "ggm.hpp"

int main() {
    int N = 10000000;
    auto start = std::chrono::high_resolution_clock::now();

    double4 a;
    double4 b;
    for (int i = 0; i < N; ++i)
    {
        b = a + b;
    }

    volatile auto c = &b;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    std::cout << "duration: " << static_cast<double>(duration.count()) / N
        << "ns" << std::endl;

    return 0;
}