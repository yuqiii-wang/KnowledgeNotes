# Atomicity

Atomicity refers to, in multi-threading, how variables can be read/write by what order they are visible to other threads that must wait till the variables become available/unlocked for next read/write.

Mandatory requirements:

* Atomicity for read/write: before the finish of the current execution, no other thread can read/write the variables. In `std::atomic`, it refers to *trivial copyables*

* Consistency/coherence: on multi-threading on different CPU cores with different L1/L2/L3 caches, data should be consistent across caches.

## Volatile

`volatile` prevents compiler optimization on variables and forces CPU re-read from memory.
However, it does NOT consider cache in multiple CPU cores, and this can be a trouble of data consistency.

`volatile` is more often used in embedded device programmed in C code, for such devices' CPUs have no cache but often receive many hardware high-priority interrupts (the interrupt-influenced variables CANNOT be optimized).

Compiler guarantees that

* Compiler does not remove `volatile`-declared variables, nor changes the order of execution, no matter what optimization flag is set

* Compiler does not optimize `volatile`-declared variables into CPU registers, but makes sure every read/write happens in memory.

However, compiler does NOT guarantee that

* atomicity of read/write

* `volatile`-declared variables not necessarily on RAM, could be on L1/L2/L3 cache or hardware driver/peripherals

* CPU hardware has lower level pipelining, might affect the order of `volatile`-declared variable execution

### Typical Use of `volatile`

Use the `volatile` qualifier to provide access to memory locations that are used by asynchronous processes such as interrupt handlers.

For example, `while (extern_int == 100)` will be compiled as `while (true)` by compiler optimization if compiler does NOT see dependent changes.
If `extern_int` changes by, e.g, another process through shared memory on `extern_int`, this `while(extern_int == 100)` no longer works.

```cpp
int extern_int = 100;

... // compiler does NOT see any update to extern_int

// promoted to `while (true)` by compiler optimization
while(extern_int == 100) {
   ...
}
```

Instead, use `volatile` to prevent compiler optimization to resolve the aforementioned issue.

```cpp
volatile int extern_int = 100;

... // compiler does NOT optimize extern_int regardless 
    // if `extern_int` might ot might not receive any update

// no optimization
while (extern_int == 100) {
   ...
}
```

### `volatile` Implementation for `++i`

When used `volatile i; ++i;`, there are four actions: Load、Increment、Store、Memory Barriers.

Memory barrier is thread safe that demands data must be retrieved from the main memory. However, for CPU loading, executing and storing between ALU and cache before implementing memory barrier, `volatile` cannot guarantee thread safety.

## `std::atomic`

`std::atomic` wraps trivial copyables and provides memory barriers by what order the copyable variables can be read/write.

`std::atomic` works with trivial copyables (such as C-compatible POD types, e.g., int, bool, char) to guarantee thread safety (defined behavior when one thread read and another thread write) by trying squeezing one operation into one cpu cycle (one instruction). 

Only some POD types are by default atomic (placeable inside one register), such as `char` and `int16_t` (both 2 bytes), dependent on register config, other POD types might not be atomic.

It is **NOT** allowed to apply atomic to an array such as

```cpp
std::atomic<std::array<int,10>> myArray;
```
in which `myArray`'s element is not readable/writable.

### `std::atomic` in Assembly

The above requirement mandating POD variable per one atomic operation derives from the fact that one CPU clock cycle can only run one assembly instruction, e.g., CPUs with 64-bit registers can only update up to an `uint64_t` at once.

For example, here updates two variables `int x = 0; x += 1;` and `int y = 1; y += 2;`.

For `std::memory_order_relaxed` atomic operation,

```cpp
std::atomic_uint64_t x{0};
std::atomic_uint64_t y{1};
x.fetch_add(1, std::memory_order_relaxed);
y += 2;
```

clang++ generates the following assembly:

```asm
lock   addl $0x1,0x2009f5(%rip)        # 0x601040 <x>
movl   $0x3,0x2009e7(%rip)             # 0x60103c <y>
```

where compiler is free to optimize `y = 1; y += 2;` into one instruction directly moving `0x3` to the register that holds the variable `y`, and the `movl` is re-ordered after `x` taking arithmetic add.

For `memory_order_seq_cst` atomic operation,

```cpp
...
x.fetch_add(1, std::memory_order_seq_cst);
y += 2;
```

clang++ enforces the order of code execution, no optimization is performed.

```asm
movl   $0x1,0x2009f2(%rip)        # 0x60103c <y>
lock   addl $0x1,0x2009eb(%rip)   # 0x601040 <x>
addl   $0x2,0x2009e0(%rip)        # 0x60103c <y>
```

P.S. `lock` is a prefix on read-modify-write operaation on memory (`INC`, `XCHG`, `CMPXCHG` etc.)

The `lock` prefix ensures that the CPU has exclusive ownership of the appropriate cache line for the duration of the operation. This may be achieved by asserting a bus lock.

## Typical Atomic Instructions

### Load and Store

Atomic load and store use strong memory order `std::memory_order_seq_cst` to guarantee read atomicity.

* `std::atomic<T>::load`: atomically loads and returns the current value of the atomic variable.

```cpp
T load( std::memory_order order
            = std::memory_order_seq_cst ) const noexcept;
```

* `std::atomic<T>::store`: Atomically replaces the current value with desired.

```cpp
void store( T desired, std::memory_order order =
                           std::memory_order_seq_cst ) noexcept;
```

### Fetch and Arithmetics

* `fetch_add`: atomically replaces the current value with the result of arithmetic addition of the value and `arg`.

```cpp
T fetch_add( T arg, std::memory_order order =
                        std::memory_order_seq_cst ) noexcept;
```

### Set A Value

* `std::atomic_flag::test_and_set`: atomically changes the state of a `std::atomic_flag` to set (`true`) and returns the value it held before.

```cpp
bool test_and_set( std::memory_order order =
                       std::memory_order_seq_cst ) noexcept;
```

The `std::atomic_flag` is an atomic boolean type guaranteed to be lock-free.
It provides NO load/store operation, but simply a boolean state.

It can be used as an atomic lock to avoid expensive `mutex`.
Below is an example of spinlock.

```cpp
// default to `false`
std::atomic_flag lock = ATOMIC_FLAG_INIT;

// `f` will be run in multi-threading mode
void f(int n)
{
    for (int cnt = 0; cnt < 40; ++cnt)
    {
        while (lock.test_and_set(std::memory_order_acquire)) // acquire lock
        {
            // Since C++20, it is possible to update atomic_flag's
            // value only when there is a chance to acquire the lock.
            // See also: https://stackoverflow.com/questions/62318642
        #if defined(__cpp_lib_atomic_flag_test)
            while (lock.test(std::memory_order_relaxed)) // test lock
        #endif
                ; // spin
        }
        static int out{};
        std::cout << n << ((++out % 40) == 0 ? '\n' : ' ');
        lock.clear(std::memory_order_release); // release lock
    }
}
```

* `std::atomic<T>::exchange`: atomically replaces the underlying value with `desired` (a read-modify-write operation).

```cpp
T exchange( T desired, std::memory_order order =
                           std::memory_order_seq_cst ) noexcept;
```

### Compare And Swap (CAS)

Compare and Swap (CAS) is a `lock` x86 assembly instruction to replace `if(a==b) a=c;` so that CAS operation can be put into one CPU clock cycle.

Assembly equivalent is

```cpp
//  Perform atomic 'compare and swap' operation on the pointer.
//  The pointer is compared to 'cmp' argument and if they are
//  equal, its value is set to 'val'. Old value of the pointer is returned.
inline T *cas (T *cmp_, T *val_)
{
    T *old;
    __asm__ volatile (
        "lock; cmpxchg %2, %3"
        : "=a" (old), "=m" (ptr)
        : "r" (val_), "m" (ptr), "0" (cmp_)
        : "cc");
    return old;
}
```

* `compare_exchange_strong` vs `compare_exchange_weak`

The key diff is that `compare_exchange_weak` may fail spuriously (for ABA problem, acts as if `desired != expected` even if they are equal), while `compare_exchange_strong` guarantees success.


```cpp
// since c++11
bool compare_exchange_strong( T& expected, T desired,
                              std::memory_order order =
                                  std::memory_order_seq_cst ) noexcept;
// since c++11
bool compare_exchange_weak( T& expected, T desired,
                            std::memory_order order =
                                std::memory_order_seq_cst ) noexcept;
```

In multithreaded computing, the ABA problem occurs during synchronization, when a memory location is read twice and both reads see no change of the variable value, this thread can conclude no update in the interim; however, another thread may update this variable then soon change the value back.

In CAS, when `desired != expected` really happens, it can be said that the exchange operation is done.
However, `compare_exchange_weak` might return `true` even for `desired == expected` due to the parasitic ABA problem.

`compare_exchange_strong` needs extra overhead to retry in the case of failure, while `compare_exchange_weak` immediately returns.

To make sure the success of CAS, `compare_exchange_weak` can be put in a loop such that `while (!value.compare_exchange_weak(expected, desired))` that is equivalent to `if (value.compare_exchange_strong(expected, desired))`.

```cpp
#include <iostream>
#include <atomic>
#include <thread>

// Shared atomic variable
std::atomic<int> value(0);

// Function using compare_exchange_weak
void weak_cas() {
    int expected = 0;
    int desired = 1;
    // Attempt to set value to 1 if it is currently 0
    while (!value.compare_exchange_weak(expected, desired)) {
        // Spurious failure may occur, so retry
        expected = 0;
    }
    std::cout << "Weak CAS successful, value: " << value.load() << std::endl;
}

// Function using compare_exchange_strong
void strong_cas() {
    int expected = 0;
    int desired = 2;
    // Attempt to set value to 2 if it is currently 0
    if (value.compare_exchange_strong(expected, desired)) {
        std::cout << "Strong CAS successful, value: " << value.load() << std::endl;
    } else {
        std::cout << "Strong CAS failed, expected: " << expected << ", actual: " << value.load() << std::endl;
    }
}

int main() {
    std::thread t1(weak_cas);
    std::thread t2(strong_cas);

    t1.join();
    t2.join();

    return 0;
}
```

## Use Cases

* A `SpinLock` is often used in place of `mutex` to avoid expensive cost of context switch. However, it is computation intensive as all waiting threads are pending on `while (flag_.exchange(true, std::memory_order_acquire));` running on a loop attempting to acquire the lock.

```cpp
class SpinLock {
private:
    std::atomic<bool> flag_ = {false};
public:
    void Lock() {
        while (flag_.exchange(true, std::memory_order_acquire)); 
    }

    void Unlock() {
        flag_.store(false, std::memory_order_release); 
    }
};
```

* Lock Free Stack uses CAS to implement `push` and `pop`.

Declare `node<T>* next;` to meet the requirement that atomic operations must be on trivial copyables (a pointer is always trivial copyable).

A stack has its `push` and `pop` both on the head.
Only the stack's head pointer is of `std::atomic<node<T>*> _head;`.

The `push(const T& val)` sets up the `val` as a node and gets it pointed by the stack head such that  `new_node->next = _head.load();`, where `_head.load()` guarantees atomicity preventing other threads' operation on the head.

Below is an example of a lock-free stack `class LFS`.

```cpp
#include <iostream>
#include <atomic>

template<class T>
class LFS {
public:
    struct node
    {
        T data;
        node<T>* next;
        node(const T& data) : data(data), next(nullptr) {}
    };

    void push(const T& val) {
        node<T>* new_node = new node<T>(val);
        new_node->next = _head.load();
        while(!_head.compare_exchange_weak(new_node->next, new_node));
    }

    bool pop() {
        node<T>* got = _head.load();
        node<T>* nextn = nullptr;
        do {
            if(got == nullptr) {
                return false;
            }
            nextn = got->next;
        } while(!_head.compare_exchange_weak(got, nextn));
        delete got;
        return true;
    }
private:
    std::atomic<node<T>*> _head;
};
```