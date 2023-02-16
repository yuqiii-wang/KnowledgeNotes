#include <benchmark/benchmark.h>
#include <random>

// `__builtin_expect` for branch prediction to speed up if condition execution
#define unlikely(expr) __builtin_expect(!!(expr), 0)
#define likely(expr) __builtin_expect(!!(expr), 1)

const int distMax = 100000;

static void BM_NoBranchingPrediction(benchmark::State& state) {

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0,distMax); // distribution in range [0, distMax]

  for (auto _ : state)
  {
    if ((dist(rng) % distMax) + 10 < distMax) 
    {
        for (int i = 0; i < 10; i++)
            std::string x("hello");
    }
  }
}
// Register the function as a benchmark
BENCHMARK(BM_NoBranchingPrediction);

// Define another benchmark
static void BM_HasBranchingPrediction(benchmark::State& state) {

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0,distMax); // distribution in range [0, distMax]

  for (auto _ : state)
  {
    if (likely((dist(rng) % distMax) + 10 < distMax) )
    {
        for (int i = 0; i < 10; i++)
            std::string x("hello");
    }
  }
}
BENCHMARK(BM_HasBranchingPrediction);

BENCHMARK_MAIN();