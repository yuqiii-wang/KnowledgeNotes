# Processor Affinity and CPU Isolation

Processor affinity, or CPU pinning or "cache affinity", enables the binding and unbinding of a process or a thread to a central processing unit (CPU) or a range of CPUs, so that the process or thread will execute only on the designated CPU or CPUs rather than any CPU. 

First, have a look at cores of a CPU by `lscpu` (alternatively, find each CPU core by `cat /proc/cpuinfo`) to make sure there are more than 1 CPU cores.
```
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Byte Order:                      Little Endian
Address sizes:                   39 bits physical, 48 bits virtual
CPU(s):                          12
On-line CPU(s) list:             0-11
Thread(s) per core:              2
Core(s) per socket:              6
Socket(s):                       1
NUMA node(s):                    1
...
```

## Thread Native Handle

Thread native handle is implementation-specific information storage of a thread.

For example, the code below proves that `t.native_handle()` returns the exact result as that of `std::this_thread::get_id()`,
indicating that they refer to the same underlying thread.
```cpp
td::mutex iomutex;
std::thread t = std::thread([&iomutex] {
{
    std::lock_guard<std::mutex> iolock(iomutex);
    std::cout << "Thread: my id = " << std::this_thread::get_id() << "\n"
            << "        my pthread id = " << pthread_self() << "\n";
}
});

{
std::lock_guard<std::mutex> iolock(iomutex);
std::cout << "Launched t: id = " << t.get_id() << "\n"
            << "            native_handle = " << t.native_handle() << "\n";
}
```
prints
```
Launched t: id = 139665344911104
            native_handle = 139665344911104
Thread: my id = 139665344911104
        my pthread id = 139665344911104
```

* On Linux `native_handle` is stored at `thread._M_id(of type id)._M_thread`.
* On Windows `native_handle` is stored at `thread._Thr(of type _Thrd_t, not of type id)._Hnd`.
* On MacOS `native_handle` is stored at `thread.__t_`.

## Thread Affinity

Affinity means that instead of being free to run the thread on any CPU it feels like, the OS scheduler is asked to only schedule a given thread to a single CPU or a pre-defined set of CPUs. 

By default, the affinity covers all logical CPUs in the system, so the OS can pick any of them for any thread.

Thread has builtin support of setting CPUs, first set CPU config
```cpp
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
CPU_SET(cpuid, &cpuset);
```
where `CPU_ZERO` is used to clear `cpuset`, so that it contains no CPUs; 
`CPU_SET` adds CPU cpu to `cpuset`.

Then launch a thread by `pthread_setaffinity_np`,
which set the CPU affinity mask of the thread `thread` to the CPU set pointed to by `cpuset`.
```cpp
#define _GNU_SOURCE             /* See feature_test_macros(7) */
#include <pthread.h>

int pthread_setaffinity_np(pthread_t thread, size_t cpusetsize,
                            const cpu_set_t *cpuset);
int pthread_getaffinity_np(pthread_t thread, size_t cpusetsize,
                            cpu_set_t *cpuset);
```

For example, below code force threads running on 0, 1, 2 and 3 CPU.
```cpp
constexpr unsigned num_threads = 4;
std::mutex iomutex;
std::vector<std::thread> threads(num_threads);
for (unsigned i = 0; i < num_threads; ++i) {
  threads[i] = std::thread([&iomutex, i] {...});

  // Create a cpu_set_t object representing a set of CPUs. Clear it and mark
  // only CPU i as set.
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(i, &cpuset);
  int rc = pthread_setaffinity_np(threads[i].native_handle(),
                                  sizeof(cpu_set_t), &cpuset);
}
```

## CPU Isolation

Symmetrical multiprocessing (SMP) denotes a multiprocessor architecture in which no CPU is selected as the Master CPU, but rather all of them cooperate on an equal basis, hence the name "symmetrical."

CPU isolation means removing one or more CPU cores from Linux process SMP scheduler.

### The `isolcpus` Command

`isolcpus` isolates CPUs from the kernel scheduler.

```
isolcpus=<cpu_number>
```

`isolcpus` removes the specified CPUs defined by the `cpu_number` values, from the general kernel SMP balancing and scheduler algorithms. 
The only way to move a process onto or off an "isolated" CPU is via the CPU affinity syscalls. 

For example, `isolcpus=0` means Linux should not allocates tasks on CPU 0.

## I/O (IRQs) Affinity

When dealing with I/O work, hardware interrupt handlers are usually scheduled by Linux SMP.
If user code has user space thread affinity without specifying I/O affinity, there will be data across threads.

Starting with the 2.4 kernel, Linux has gained the ability to assign certain
IRQs to specific processors (or groups of processors).  
This is known as SMP IRQ affinity, and it allows user-level control of hardware event handling by which CPU core.

### Practices

In `/proc/irq/` list a number of IRQs as folders corresponding to different interrupt events.
In each of these directories is the `smp_affinity` file that maps an IRQ to CPU cores (see the mapping table by `cat /proc/interrupts`).

