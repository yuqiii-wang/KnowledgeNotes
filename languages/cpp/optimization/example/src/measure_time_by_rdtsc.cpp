#include <time.h>
#include <chrono>
#include <ctime>

#include <iostream>

// #include <linux/thread_info.h> 
// #include <linux/linkage.h> 
// #include <linux/list.h>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

const int runNums = 100000;
const int testFuncLoops = 1000;

const double cpuClockRate = 2.6e9; // get by `cat /proc/cpuinfo`

class TestFunc
{
public:
    int operator()(int a, int b)
    {
        for (int i = 0; i < testFuncLoops; i++) {
            b += a + i;
        }
        return b;
    }
};

long long measureElapsedTimeByRDTSC(TestFunc& func)
{

    unsigned cycles_low0, cycles_high0, cycles_low1, cycles_high1;

    asm volatile (
            "RDTSC\n\t"
            "mov %%edx, %0\n\t"
            "mov %%eax, %1\n\t": "=r" (cycles_high0), "=r" (cycles_low0)
    );

    func(10,10);

    asm volatile (
            "RDTSC\n\t"
            "mov %%edx, %0\n\t"
            "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)
    );

    auto start = ( ((uint64_t)cycles_high0 << 32) | cycles_low0 );
    auto end = ( ((uint64_t)cycles_high1 << 32) | cycles_low1 );

    return end - start;
}

double measureElapsedTimeByChrono(TestFunc& func)
{
    auto start = std::chrono::high_resolution_clock::now();
    func(10,10); 
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count();
}

int main()
{
    TestFunc func;
    long long elapsedTimeRDTSC = 0;
    double elapsedTimeChrono = 0;

    // unsigned long flags;
    // preempt_disable(); /*we disable preemption on our CPU*/
    // raw_local_irq_save(flags); /*we disable hard interrupts on our CPU*/
    // /*at this stage we exclusively own the CPU*/

    for (int i = 0; i < runNums; i++) {
        elapsedTimeRDTSC += measureElapsedTimeByRDTSC(func);
        elapsedTimeChrono += measureElapsedTimeByChrono(func);
    }

    // raw_local_irq_restore(flags); /*we enable hard interrupts on our CPU*/
    // preempt_enable();/*we enable preemption*/

    std::cout << "elapsedTimeRDTSC: " << elapsedTimeRDTSC / runNums / cpuClockRate << std::endl;
    std::cout << "elapsedTimeChrono: " << elapsedTimeChrono / runNums / 1000 << std::endl;

    return 0;
}