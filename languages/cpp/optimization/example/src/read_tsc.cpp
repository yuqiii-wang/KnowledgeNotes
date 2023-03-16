#include <time.h>
#include <chrono>
#include <ctime>
#include <benchmark/benchmark.h>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif


static void BM_REALTIME(benchmark::State& state) {

    struct timespec ts;

    for (auto _ : state)
    {
        clock_gettime(CLOCK_REALTIME, &ts);
    }
    
}

BENCHMARK(BM_REALTIME);

static void BM_MONOTONIC(benchmark::State& state) {

    struct timespec ts;

    for (auto _ : state)
    {
        clock_gettime(CLOCK_MONOTONIC, &ts);
    }
    
}

BENCHMARK(BM_MONOTONIC);

static void BM_PROC_CPUTIME(benchmark::State& state) {

    struct timespec ts;

    for (auto _ : state)
    {
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    }
    
}

BENCHMARK(BM_PROC_CPUTIME);

static void BM_THR_CPUTIME(benchmark::State& state) {

    struct timespec ts;

    for (auto _ : state)
    {
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    }
    
}

BENCHMARK(BM_THR_CPUTIME);

static void BM_CHRONO_API(benchmark::State& state) {

    for (auto _ : state)
    {
        std::chrono::time_point<std::chrono::system_clock> now =
                                        std::chrono::system_clock::now();    
    }
    
}

BENCHMARK(BM_CHRONO_API);

static void BM_READ_TSC(benchmark::State& state) {

    for (auto _ : state)
    {
        unsigned long long i = __rdtsc();
    }
    
}

BENCHMARK(BM_READ_TSC);

BENCHMARK_MAIN();