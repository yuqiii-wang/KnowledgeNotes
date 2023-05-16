# Common optimization practices

## Memory access by block 

```cpp
// multiple threads running a lambda
int a[N];
std::thread* threadRunsPtrs[n];
for (int i = 0; i < n; i++)
{
    threadRunsPtrs[i] = new thread(lambdaPerThread, i, N/n);
}

// each thread accessing an individual block of mem,
// good for parallel computation
auto lambdaPerThread = [&](int threadId; int blockRange)
{
    for (int i = blockRange*threadId; i < blockRange*(threadId+1); i++)
    {
        a[i];
    }
}

// scattered mem access, bad for parallel computation
// actually only one thread is running at a time
auto lambdaPerThread = [&](int threadId; int threadTotalNum)
{
    for (int i = threadId; i < N; i += threadTotalNum)
    {
        a[i];
    }
}
```

### Large Matrix Computation Optimization

When performing 

## Inline

`inline` function are faster in execution( compared to normal function) due to overhead saved by removal of
* function call
* pushing of function parameters on stack

However, it might reduce performance if misused, for increased cache misses and thrashing.

### `inline` Implementation

Inline expansion is similar to macro expansion as the compiler places a new copy of the function in each place it is called. 
Inlined functions run a little faster than the normal functions as function-calling-overheads are saved.

Inline expansion is used to eliminate the time overhead (excess time) when a function is called.

Without inline functions, the compiler decides which functions to inline.

Ordinarily, when a function is invoked, control is transferred to its definition by a branch or call instruction. With inlining, control drops through directly to the code for the function, without a branch or call instruction.

`inline` does not work for virtual function nor recursion.

### Implicit `inline`

When there is need to `sort` many `struct S`, the below code is inefficient for `compare` will not be considered `inline`.
```cpp
struct S {
    int a, b;
};
int n_items = 10000;
S arrS[n_items];

bool compare(const S& s1, const S& s2) {
    return s1.b < s2.b;
}

std::sort(arrS, arrS + n_items, compare);
```

Instead, define `struct Comparator`. The `operator()` is by default `inline`
```cpp
struct Comparator {
    bool operator()(const S& s1, const S& s2) const {
        return s1.b < s2.b;
    }
};

std::sort(arr, arr + n_items, Comparator());
```

### `inline` vs Macro

Inline expansion  occurs during compilation, without changing the source code (the text), while macro expansion occurs prior to compilation.

## `noexcept`

Compiler uses *flow graph* to optimize machine code generation. A flow graph consists of what are generally called "blocks" of the function (areas of code that have a single entrance and a single exit) and edges between the blocks to indicate where flow can jump to. `noexcept` alters the flow graph (simplifies flow graph not to cope with any error handling)

For example, code below using containers might throw `std::bad_alloc` error for lack of memory, adding complexity to flow graph. 
There are many errors a function can throw, and error handling code blocks can be many in a flow graph. By `noexcept`, flow graph is trimmed such that only `std::terminate()` is invoked when error throws. 
```cpp
double compute(double x) noexcept {
    std::string s = "Courtney and Anya";
    std::vector<double> tmp(1000);
    // ...
}
```

Another example is that, containers such as `std::vector` will move their elements if the elements' move constructor is `noexcept`, 
and copy otherwise (unless the copy constructor is not accessible, but a potentially throwing move constructor is, in which case the strong exception guarantee is waived).

### `noexcept` Best Practices

Use `noexcept` in below scenarios:
* move constructor
* move assignment
* destructor (since C++11, they are by default `noexcept`)

## Elapsed Time Measurement by Time Stamp Counter (TSC)

The Time Stamp Counter (TSC) is a 64-bit register present on all x86 processors since the Pentium. 
It counts the number of CPU cycles since its reset. The instruction `RDTSC` returns the `TSC` in `EDX:EAX`.

### The `clock_gettime`

Normal `std::chrono` implementations will use an OS-provided function like POSIX `clock_gettime()` that depending on config, for example, `CLOCK_PROCESS_CPUTIME_ID` and `CLOCK_THREAD_CPUTIME_ID` uses RDTSC, while some other config may trigger clock hardware interrupt.
Nevertheless, there is overhead associating with this method, typically $0.2$-$0.3$ micro secs.
As a result, for nano sec level time measurement, should use `RDTSC`, otherwise, for micro sec level or above, can just use `std::chrono`.

`clock_gettime` retrieves the time of the specified clock `clk_id`, and set to `tp`. 
```cpp
int clock_gettime(clockid_t clk_id, struct timespec *tp);

struct timespec {
        time_t   tv_sec;        /* seconds */
        long     tv_nsec;       /* nanoseconds */
};
```

There are four types of clock:

* `CLOCK_REALTIME`
    System-wide realtime clock. Setting this clock requires appropriate privileges. 
* `CLOCK_MONOTONIC`
    Clock that cannot be set and represents monotonic time since some unspecified starting point. 
* `CLOCK_PROCESS_CPUTIME_ID`
    High-resolution per-process timer from the CPU. 
* `CLOCK_THREAD_CPUTIME_ID`
    Thread-specific CPU-time clock.

Performance shown as below
```
----------------------------------------------------------
Benchmark                Time             CPU   Iterations
----------------------------------------------------------
BM_REALTIME           16.6 ns         16.6 ns     45458598
BM_MONOTONIC          17.2 ns         17.2 ns     40920728
BM_PROC_CPUTIME        672 ns          672 ns      1010758
BM_THR_CPUTIME         660 ns          660 ns      1019584
``` 

Fine-grained timing comes from a fixed-frequency counter that counts "reference cycles" regardless of turbo, power-saving, or clock-stopped idle, and this can be obtained from `rdtsc`, or `__rdtsc()` in C/C++.

### Practices

FIrst check if CPU is enabled RDTSC by `grep tsc /proc/cpuinfo`, where check
* constant_tsc
* nonstop_tsc

Then, just read TSC by `__rdtsc`.
```cpp
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

// optional wrapper if you don't want to just use __rdtsc() everywhere
inline
unsigned long long readTSC() {
    // _mm_lfence();  // optionally wait for earlier insns to retire before reading the clock
    return __rdtsc();
    // _mm_lfence();  // optionally block later instructions until rdtsc retires
}
```

The above read just needs 7 nano secs of processing time.
```
----------------------------------------------------------
Benchmark                Time             CPU   Iterations
----------------------------------------------------------
BM_READ_TSC           7.20 ns         7.20 ns     93764177
```

### x86 CPU Specifications

x86 since *Pentium* has introduced `TSC`:

* `rdtsc`: Read Time-Stamp Counter
* `rdtscp`: Read Time-Stamp Counter and Processor ID

Assembly implementation:

RDTSC instruction, once called, overwrites the `EAX` and `EDX` registers.
```cpp
unsigned cycles_low0, cycles_high0, cycles_low1, cycles_high1;

preempt_disable(); /*we disable preemption on our CPU*/
unsigned long flags;
raw_local_irq_save(flags); /*we disable hard interrupts on our CPU*/
/*at this stage we exclusively own the CPU*/

asm volatile (
          "RDTSC\n\t"
          "mov %%edx, %0\n\t"
          "mov %%eax, %1\n\t": "=r" (cycles_high0), "=r" (cycles_low0)
);

... // some code to run

asm volatile (
          "RDTSC\n\t"
          "mov %%edx, %0\n\t"
          "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)
);

raw_local_irq_restore(flags);
/*we enable hard interrupts on our CPU*/
preempt_enable();/*we enable preemption*/
```

The purely assembly implementation has almost the same time as that of by reading `__rdtsc();`.
```
----------------------------------------------------------
Benchmark                Time             CPU   Iterations
----------------------------------------------------------
BM_READ_ASM_TSC       6.97 ns         6.97 ns    104420355
```

Microsoft Visual C++ and Linux & gcc provides the API reading the TSC's value:

* Microsoft Visual C++:
```cpp
unsigned __int64 __rdtsc();
unsigned __int64 __rdtscp( unsigned int * AUX );
```

* Linux & gcc :
```cpp
extern __inline unsigned long long
__attribute__((__gnu_inline__, __always_inline__, __artificial__))
__rdtsc (void) {
  return __builtin_ia32_rdtsc ();
}

extern __inline unsigned long long
__attribute__((__gnu_inline__, __always_inline__, __artificial__))
__rdtscp (unsigned int *__A)
{
  return __builtin_ia32_rdtscp (__A);
}
```

Limitations (especially for old CPUs):

* Must be CPU-specific in testing (same CPU version): checking by `cpuid` if `CPUID.80000001H:EDX.RDTSCP[bit 27]` is `1`
* Must re-calibrate the counter if power-saving measures taken by the OS or BIOS or overclocking happens

Remediation to the downclocking/overclocking limitation:

Recent Intel processors include a constant rate TSC (identified by the `kern.timecounter.invariant_tsc` sysctl on FreeBSD or by the "constant_tsc" flag in Linux's `/proc/cpuinfo`).

With these processors, the TSC ticks at the processor's nominal frequency, regardless of the actual CPU clock frequency due to turbo or power saving states. Hence TSC ticks are counting the passage of time, not the number of CPU clock cycles elapsed.

ARM Specifications:

P.S., ARMv7 provides the *Cycle Counter Register* (`CCNT` instruction) to read and write the counter, but the instruction is privileged.