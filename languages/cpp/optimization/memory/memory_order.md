# Memory Order

## Atomic Load and Store

A typical operation on a variable is load-read/write-store. 
Atomic load and store specify in which order data should be retrieved as set in `std::memory_order order`.

`std::atomic<T>::load` atomically loads and returns the current value of the atomic variable. 
```cpp
T load( std::memory_order order = std::memory_order_seq_cst ) const noexcept;
```

`std::atomic<T>::store` atomically replaces the current value with desired.
```cpp
void store( T desired, std::memory_order order = std::memory_order_seq_cst ) noexcept;
```

Given the `std::atomic<int> foo (0)` that is `set` and `get` by two threads, atomic load and store can guarantee the value consistency.
If simply `int foo = 0` and with normal set such as `void set_foo(int x) { foo = x; }`, there is no consistency guarantee in another thread `int get() { return foo; }`.
```cpp
// atomic::load/store example
#include <iostream>       // std::cout
#include <atomic>         // std::atomic, std::memory_order_relaxed
#include <thread>         // std::thread

std::atomic<int> foo (0);

void set_foo(int x) {
  foo.store(x,std::memory_order_relaxed);     // set value atomically
}

void print_foo() {
  int x;
  do {
    x = foo.load(std::memory_order_relaxed);  // get value atomically
  } while (x==0);
  std::cout << "foo: " << x << '\n';
}

int main ()
{
  std::thread first (print_foo);
  std::thread second (set_foo,10);
  first.join();
  second.join();
  return 0;
}
```

## Memory Order Types

### Memory Ordering Modes

* Relaxed ordering

Atomic operations tagged `memory_order_relaxed` are not synchronization operations; they do not impose an order among concurrent memory accesses. 
They only guarantee atomicity and modification order consistency.

For example, the code below produces true for `r1 == r2 == 99 ` if thread 1 and thread 2 run synchronously.
The side-effect of event $D$ on `y` could be visible to the load $A$ in thread 1, while the side effect of $B$ on `x` could be visible to the load $C$ in thread 2.
```cpp
// Both defined in thread 1 and thread 2
std::atomic<int> x (0);
std::atomic<int> y (0);
int r1, r2;

// Suppose both thread 1 and thread 2 simultaneously run into theses instructions
// Thread 1:
r1 = y.load(std::memory_order_relaxed); // A
x.store(r1, std::memory_order_relaxed); // B
// Thread 2:
r2 = x.load(std::memory_order_relaxed); // C 
y.store(99, std::memory_order_relaxed); // D
```

* Release-Acquire ordering

If an atomic store in thread 1 is tagged `memory_order_release`, an atomic load in thread 2 from the same variable is tagged `memory_order_acquire`, 
and the load in thread 2 reads a value written by the store in thread 1, then the store in thread 1 synchronizes-with the load in thread 2.
Together, they can be tagged `memory_order_acq_rel`.

That is, once the atomic load is completed in thread 1, thread 2 is guaranteed to see everything thread 1 wrote to memory.

Mutual exclusion locks, such as `std::mutex` or `atomic spinlock`, are an example of release-acquire synchronization.

* Release-Consume ordering

If an atomic store in thread 1 is tagged `memory_order_release`, an atomic load in thread 2 from the same variable is tagged `memory_order_consume`, 
and the load in thread 2 reads a value written by the store in thread 1, then the store in thread 1 is dependency-ordered before the load in thread 2.

* Sequentially-consistent ordering

Atomic operations tagged `memory_order_seq_cst` not only order memory the same way as release/acquire ordering, but also establish a single total modification order of all atomic operations that are so tagged.

The example code below shows true for `r1 == 1 && r2 == 3 && r3 == 0` when thread 1, 2 and 3 run synchronously.
A modification order has to be first loading and modifying, then storing. As a result, when synchronously running and tagged `memory_order_seq_cst`, event $F$ precedes $A$, event $C$ precedes $E$.
The overall execution order would be $C$-$E$-$F$-$A$.
```cpp
// Both defined in thread 1 and thread 2
std::atomic<int> x (0);
std::atomic<int> y (0);
int r1, r2, r3;

// run synchronously
// Thread 1:
x.store(1, std::memory_order_seq_cst); // A
y.store(1, std::memory_order_release); // B
// Thread 2:
r1 = y.fetch_add(1, std::memory_order_seq_cst); // C
r2 = y.load(std::memory_order_relaxed); // D
// Thread 3:
y.store(3, std::memory_order_seq_cst); // E
r3 = x.load(std::memory_order_seq_cst); // F
```

### `std::memory_order` Specifications

`std::memory_order` specifies how memory accesses, including regular, non-atomic memory accesses, are to be ordered around an atomic operation.

```cpp
// since C++11, until C++20
typedef enum memory_order {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;
```
