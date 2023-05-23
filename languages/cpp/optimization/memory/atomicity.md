# Atomicity

Mandatory requirements:

* Atomicity for read/write: before the finish of the current execution, no other thread can read/write the variables

* Consistency: on multi-threading on different CPU cores with different L1/L2/L3 caches, data should be consistent across caches.

### `std::atomic`

`std::atomic` works with trivial copyables (such as C-compatible POD types, e.g., int, bool, char) to guarantee thread safety (defined behavior when one thread read and another thread write) by trying squeezing one operation into one cpu cycle (one instruction). 

Only some POD types are by default atomic (placeable inside one register), such as `char` and `int16_t` (both 2 bytes), dependent on register config, other POD types might not be atomic.

It is **NOT** allowed to apply atomic to an array such as
```cpp
std::atomic<std::array<int,10>> myArray;
```
in which `myArray`'s element is not readable/writable.

### `std::atomic` in Assembly



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


## Memory Compare and Exchange

Compare-and-exchange atomically compares the object representation (until C++20)/value representation (since C++20) of `*this` with that of `expected`, and if those are bitwise-equal, replaces the former with `desired` (performs read-modify-write operation). Otherwise, loads the actual value stored in `*this` into `expected` (performs load operation).

Strong and weak versions refer to performance and safety, that `strong` guarantees successful compare-and-exchange, but slow in execution, while `weak` has the opposite performance.

`order` is about the memory synchronization ordering for both operations.

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

### Example

Compare-and-exchange operations are often used as basic building blocks of *lockfree* data structures.

Suppose there is a list of nodes linked by `next`. There might be multiple threads setting nodes as new heads to the list.
`head.compare_exchange_weak` is used here to guarantee atomicity of setting the current node as a new head at the time when compare-and-exchange happens.

```cpp
#include <atomic>
template<typename T>
struct node
{
    T data;
    node* next;
    node(const T& data) : data(data), next(nullptr) {}
};

template<typename T>
class stack
{
    std::atomic<node<T>*> head;
 public:
    void push(const T& data)
    {
      node<T>* new_node = new node<T>(data);
 
      // put the current value of head into new_node->next
      new_node->next = head.load(std::memory_order_relaxed);
 
      // now make new_node the new head, but if the head
      // is no longer what's stored in new_node->next
      // (some other thread must have inserted a node just now)
      // then put that new head into new_node->next and try again
      while(!head.compare_exchange_weak(new_node->next, new_node,
                                        std::memory_order_release,
                                        std::memory_order_relaxed))
          ; // the body of the loop is empty
 
// Note: the above use is not thread-safe in at least 
// GCC prior to 4.8.3 (bug 60272), clang prior to 2014-05-05 (bug 18899)
// MSVC prior to 2014-03-17 (bug 819819). The following is a workaround:
//      node<T>* old_head = head.load(std::memory_order_relaxed);
//      do {
//          new_node->next = old_head;
//       } while(!head.compare_exchange_weak(old_head, new_node,
//                                           std::memory_order_release,
//                                           std::memory_order_relaxed));
    }
};
int main()
{
    stack<int> s;
    s.push(1);
    s.push(2);
    s.push(3);
}
```
