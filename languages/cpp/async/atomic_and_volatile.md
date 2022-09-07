# Atomic and Volatile

## Atomicity

Mandatory requirements:

* Atomicity for read/write: before the finish of the current execution, no other thread can read/write the variables

* Consistency: on multi-threading on different CPU cores with different L1/L2/L3 caches, data should be consistent across caches.

## Volatile

Compiler guarantees that

* Compiler does not remove `volatile`-declared variables, nor changes the order of execution, no matter what optimization flag is set

* Compiler does not optimize `volatile`-declared variables into CPU registers, but makes sure every read/write happens in memory.

However, compiler does NOT guarantee that

* atomicity of read/write

* `volatile`-declared variables not necessarily on RAM, could be on L1/L2/L3 cache or hardware driver/peripherals

* CPU hardware has lower level pipelining, might affect the order of `volatile`-declared variable execution

## Memory barrier

Memory barrier is used to sync memory access order. It forces that read/write of a variable must be from/to the main memory, and order of execution must be not changed.