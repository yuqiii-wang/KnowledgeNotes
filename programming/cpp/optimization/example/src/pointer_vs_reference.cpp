#include <vector>
#include <unordered_map>
#include <benchmark/benchmark.h>
#include <random>


struct GibberishThings
{
    GibberishThings(){ setIntMemToOne(); }
    ~GibberishThings(){ setIntMemToZero(); }

    int a, b, c, d;
    void setIntMemToOne() {
        a = 1;
        b = 1;
        c = 1;
        d = 1;
    }

    void setRandomInt() {
        a = rand() % 10;
        b = rand() % 10;
        c = rand() % 10;
        d = rand() % 10;
    }


    void setIntMemToZero() {
        a = 0;
        b = 0;
        c = 0;
        d = 0;
    }
};

static void BM_POINTER(benchmark::State& state) {
    for (auto _ : state)
    {
        GibberishThings* g = new GibberishThings();
        g->setRandomInt();
        delete g;
    }
}

static void BM_REFERENCE(benchmark::State& state) {
    for (auto _ : state)
    {
        GibberishThings g;
        g.setRandomInt();
    }
}

BENCHMARK(BM_POINTER);
BENCHMARK(BM_REFERENCE);

BENCHMARK_MAIN();