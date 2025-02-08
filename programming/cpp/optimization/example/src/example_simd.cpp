#include <iostream>
#include <chrono>
#include <stdlib.h>

#include <benchmark/benchmark.h>

#include <immintrin.h>  // AVX/AVX2
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2

const int N = 16;

static void BM_NormalSum(benchmark::State& state) {

    int32_t A[N][N][N]; 
    int32_t B[N][N][N];
    int32_t AB[N][N][N];

    for (int i1 = 0; i1 < N; i1++)
        for (int i2 = 0; i2 < N; i2++)
            for (int i3 = 0; i3 < N; i3++) 
            {
                A[i1][i2][i3] = 10;
                B[i1][i2][i3] = 10;
    }

    for (auto _ : state)
    {
        for (int i1 = 0; i1 < N; i1++)
            for (int i2 = 0; i2 < N; i2++)
                for (int i3 = 0; i3 < N; i3++) 
                {
                    AB[i1][i2][i3] = 
                    A[i1][i2][i3] +
                    B[i1][i2][i3];
                }
    }

}
// Register the function as a benchmark
BENCHMARK(BM_NormalSum);



static void BM_SimdSum(benchmark::State& state) {

    int32_t A[N][N][N]; 
    int32_t B[N][N][N];
    int32_t AB[N][N][N];
    __m256i C[N][N][N/8];
    __m256i D[N][N][N/8];
    __m256i CD[N][N][N/8];
    int32_t* EE[N][N];

    for (int i1 = 0; i1 < N; i1++)
        for (int i2 = 0; i2 < N; i2++)
            for (int i3 = 0; i3 < N; i3++) 
            {
                A[i1][i2][i3] = 10;
                B[i1][i2][i3] = 10;
    }

    for (auto _ : state)
    {
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


        for (int i1 = 0; i1 < N; i1++)
            for (int i2 = 0; i2 < N; i2++)
                for (int i3 = 0; i3 < N/8; i3++) 
                {
                    CD[i1][i2][i3] = _mm256_add_epi32(C[i1][i2][i3], D[i1][i2][i3]);
        }

        for (int i1 = 0; i1 < N; i1++)
        for (int i2 = 0; i2 < N; i2++)
            {
                EE[i1][i2] = (int32_t*)CD[i1][i2];
    }
        
    }

}
// Register the function as a benchmark
BENCHMARK(BM_SimdSum);

BENCHMARK_MAIN();