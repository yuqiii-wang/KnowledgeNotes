# Google Benchmark

## Quick Start

Define your function that you want to measure the performance/execution time.
```cpp
void strCompare(size_t size) [
    std::string s1(size, 'a');
    std::string s2(size, 'a');
    s1.compare(s2);
]
```

Define a google benchmark wrapper for the function. The for loop is required by google benchmark to run a sufficient number of executions until the result is regarded stable.
```cpp
static void BM_strCompare(benchmark::State& state) {
    for ( auto _ : state) {
        strCompare(99999);
    }
}
```

Register this benchmark wrapper and the `MAIN` starts running the benchmark wrapper
```cpp
BENCHMARK(BM_strCompare);
BENCHMARK_MAIN();
```