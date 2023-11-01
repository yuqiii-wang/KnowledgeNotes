# Cache False Sharing

## MESI (modified, Exlusive, Shared, Invalid)

MESI (Modified, Exlusive, Shared, Invalid) protocol is an Invalidate-based cache coherence protocol (how to manage cache sharing among diff processors).

Cache sharing unit is based on **one cache line** (typically 64 bytes)

* Modified: The cache line is present only in the current cache, and is dirty (diff rom main memory), should write back to main memory before further read
* Exclusive: The cache line is present only in the current cache, but is clean - it matches main memory. It can be shared.
* Shared: This cache line may be stored in other caches of the machine and is clean - it matches the main memory.
* Invalid: The cache is not used.

## False Cache Sharing

`#pragma omp parallel` is an OpenMP lib macro that does MESI.

However, MESI does not guarantee the below scenario preventing false sharing.

```cpp
double sum = 0.0, sumLocal[10];
#pragma omp paralle num_threads(10)
{
    int thdId = omp_get_thread_num();
    sumLocal[thdId] = 0.0;

    // multi-threading loop
    #prgama omp for
    for (i = 0; i < 1000; i++) {
        sumLocal[thdId] += 1; // this line causes trouble
    }

    #pragma omp atomic
    sum += sumLocal[thdId];
}
```

Given the example shown as above, there is a potential for false sharing on array `sumLocal`, that is small enough to fit in a single cache line.
During execution, `sumLocal` is copied to multiple cache in diff processors, and each processor updates its own `sumLocal[thdId] += 1;` that leaves other elements empty/null in individual cache.
As a result, in `sum += sumLocal[thdId];`, the `sumLocal` arrays from diff caches are merged in main memory and the `sum` is likely wrong.

## Remediation

use compiler directives to force individual aligned to one cache line size.

```cpp
__declspec (align(64)) int thread_shared_var;
```