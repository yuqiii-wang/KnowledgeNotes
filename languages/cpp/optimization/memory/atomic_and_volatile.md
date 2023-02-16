# Atomic and Volatile

## Atomicity

Mandatory requirements:

* Atomicity for read/write: before the finish of the current execution, no other thread can read/write the variables

* Consistency: on multi-threading on different CPU cores with different L1/L2/L3 caches, data should be consistent across caches.

* `std::atomic`

`std::atomic` works with trivial copyables (such as C-compatible POD types, e.g., int, bool, char) to guarantee thread safety (defined behavior when one thread read and another thread write) by trying squeezing one operation into one cpu cycle (one instruction). 

Only some POD types are by default atomic (placeable inside one register), such as `char` and `int16_t` (both 2 bytes), dependent on register config, other POD types might not be atomic.

It is **NOT** allowed to apply atomic to an array such as
```cpp
std::atomic<std::array<int,10>> myArray;
```
in which `myArray`'s element is not readable/writable.



## Volatile

Compiler guarantees that

* Compiler does not remove `volatile`-declared variables, nor changes the order of execution, no matter what optimization flag is set

* Compiler does not optimize `volatile`-declared variables into CPU registers, but makes sure every read/write happens in memory.

However, compiler does NOT guarantee that

* atomicity of read/write

* `volatile`-declared variables not necessarily on RAM, could be on L1/L2/L3 cache or hardware driver/peripherals

* CPU hardware has lower level pipelining, might affect the order of `volatile`-declared variable execution

### Typical Use of `volatile`

Use the `volatile` qualifier to provide access to memory locations that are used by asynchronous processes such as interrupt handlers.

For example, `extern_int == 100` will be compiled as `true` by compiler optimization. If `extern_int` changes by callback of an interrupt handler, this `while(extern_int == 100)` no longer works.
```cpp
int extern_int = 100;

while(extern_int == 100)
{
   //your code
}
```

Instead, use `volatile` to prevent compiler optimization to resolve the aforementioned issue.
```cpp
volatile int extern_int = 100;

while(extern_int == 100)
{
   //your code
}
```

### `volatile` Implementation for `++i`

When used `volatile i; ++i;`, there are four actions: Load、Increment、Store、Memory Barriers.

Memory barrier is thread safe that demands data must be retrieved from the main memory. However, for CPU loading, executing and storing between ALU and cache before implementing memory barrier, `volatile` cannot guarantee thread safety.

## `std::atomic`

## Memory barrier

Memory barrier is used to sync memory access order. It forces that read/write of a variable must be from/to the main memory, and order of execution must be not changed.