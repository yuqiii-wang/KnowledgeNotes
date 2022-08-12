#include <iostream>
#include <chrono>
#include <stdlib.h>

#include <immintrin.h>  // AVX/AVX2
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2

#define NanoSec 1000000000.0

// compile by `g++ example_simd.cpp -mavx2`
int main()
{
    const int N = 64;

    int32_t A[N][N][N]; 
    int32_t B[N][N][N];
    int32_t AB[N][N][N];
    __m256i C[N][N][N/8];
    __m256i D[N][N][N/8];
    __m256i CD[N][N][N/8];
    int32_t* EE[N][N];
    

    // set rand seed
    srand (time(NULL));

    auto timestamp_nano_start = std::chrono::system_clock::now().time_since_epoch().count();
    for (int i1 = 0; i1 < N; i1++)
        for (int i2 = 0; i2 < N; i2++)
            for (int i3 = 0; i3 < N; i3++) 
            {
                A[i1][i2][i3] = 10;
                B[i1][i2][i3] = 10;
    }
    auto timestamp_nano_1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::cout << "Linear value assignment elapsed time (second): " << (double)(timestamp_nano_1-timestamp_nano_start)/NanoSec << std::endl;

    timestamp_nano_start = std::chrono::system_clock::now().time_since_epoch().count();
    for (int i1 = 0; i1 < N; i1++)
        for (int i2 = 0; i2 < N; i2++)
            for (int i3 = 0; i3 < N; i3++) 
            {
                AB[i1][i2][i3] = 
                A[i1][i2][i3] +
                B[i1][i2][i3];
    }
    timestamp_nano_1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::cout << "Linear sum elapsed time (second): " << (double)(timestamp_nano_1-timestamp_nano_start)/NanoSec << std::endl;

    auto timestamp_nano_start2 = std::chrono::system_clock::now().time_since_epoch().count();
    for (int i1 = 0; i1 < N; i1++)
        for (int i2 = 0; i2 < N; i2++)
            for (int i3 = 0; i3 < N/8; i3++) 
            {
                C[i1][i2][i3] = _mm256_set_epi32(10,10,
                                        10,10,
                                        10,10,
                                        10,10);
                D[i1][i2][i3] = _mm256_set_epi32(10,10,
                                        10,10,
                                        10,10,
                                        10,10);
    }
    auto timestamp_nano_2 = std::chrono::system_clock::now().time_since_epoch().count();
    std::cout << "SIMD value assignment elapsed time (second): " << (double)(timestamp_nano_2-timestamp_nano_start2)/NanoSec << std::endl;

    auto timestamp_nano_start3 = std::chrono::system_clock::now().time_since_epoch().count();
    for (int i1 = 0; i1 < N; i1++)
        for (int i2 = 0; i2 < N; i2++)
            for (int i3 = 0; i3 < N/8; i3++) 
            {
                CD[i1][i2][i3] = _mm256_add_epi32(C[i1][i2][i3], D[i1][i2][i3]);
    }
    auto timestamp_nano_3 = std::chrono::system_clock::now().time_since_epoch().count();
    std::cout << "SIMD sum elapsed time (second): " << (double)(timestamp_nano_3-timestamp_nano_start3)/NanoSec << std::endl;

    timestamp_nano_start3 = std::chrono::system_clock::now().time_since_epoch().count();
    for (int i1 = 0; i1 < N; i1++)
        for (int i2 = 0; i2 < N; i2++)
            {
                EE[i1][i2] = (int32_t*)CD[i1][i2];
    }
    timestamp_nano_3 = std::chrono::system_clock::now().time_since_epoch().count();
    std::cout << "SIMD conversion elapsed time (second): " << (double)(timestamp_nano_3-timestamp_nano_start3)/NanoSec << std::endl;


    // attest the last element is accessible, no seg fault
    std::cout << "EE[N-1][N-1][N-1] = " << EE[N-1][N-1][N-1] << std::endl;


    return 0;
    
}