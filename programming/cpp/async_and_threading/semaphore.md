# Semaphore

Generally speaking, a semaphore is a counter that is used to control access to a resource (allows a specified number of threads to access a resource concurrently.), and usually heavier in implementation than mutex.

By `C++20` standard, a typical counting semaphore can be defined as

```cpp
std::counting_semaphore<size_t> c_sem(int);
```

where the templated `size_t` specifies the max number of threads (max counter value) allowed for resource access; the argument `int` is the initial counter (the number of threads can start execution immediately until `acquire()` decrements the semaphore counter to 0).

Semaphore has two methods:

* `release`: increments the internal counter and unblocks acquirers
* `acquire`: decrements the internal counter or blocks until it can

When semaphore counter is 0, no thread can execute the protected critical resource.

Rule of thumb:

* Use mutex + condition variable to specify which thread to wake up
* Use semaphore to wake up anyone waiting thread
* Use mutex + condition variable for "small" operation (frequent lock acquisition and release), e.g., formula computation in critical section
* Use semaphore for long-time waiting operation, e.g., I/O operation

## Case Study: Zero Even Odd

Design three threads, `zero`, `even`, `odd`, that take turns to process (print) a number in the range of 0 to n.
Assume the `printNumber` function contains critical section (number printing is atomic), so that the critical section should be protected.

* `zero` will print 0.
* `even` will print even numbers.
* `odd` will print odd numbers.

See below for the semaphore vs mutex version implementation.

### The Semaphore Solution

First, set up three semaphores with counter max to `1`, where only the `zero` thread is set up to `1` at initialization.

```cpp
counting_semaphore<1> zero_sem{1}; // init to 1
counting_semaphore<1> even_sem{0}; // init to 0
counting_semaphore<1> odd_sem{0};  // init to 0
```

For `zero_sem` init to `1`, a thread can acquire the semaphore and execute the critical section (decremented the semaphore counter to 0, hence no other thread can run the `zero` critical section code).
`even_sem`, `odd_sem` are initialized to `0` so that `even` and `odd` threads will be blocked on start, and will be woken up with semaphore release from the `zero` thread (depending on the value of `i % 2`).

Having done running `printNumber(0);`, the `zero` thread by `release()` increments either `even`'s or `odd`'s semaphore counter, thereby allowing the start of another thread.

Next, e.g., when `even` thread is woken up and has printed its number, it returns the control to the `zero` thread to decide next semaphore (`i % 2` happens in the `zero` thread).

The sequence runs until all `n` numbers are exhausted.

```cpp
#include <iostream>
#include <thread>
#include <semaphore>
#include <functional>

using namespace std;

class ZeroEvenOdd {
private:
    int n;
    int current = 1;
    counting_semaphore<1> zero_sem{1}; // init to 1
    counting_semaphore<1> even_sem{0}; // init to 0
    counting_semaphore<1> odd_sem{0};  // init to 0

public:
    ZeroEvenOdd(int n) : n(n) {}

    void zero(function<void(int)> printNumber) {
        for (int i = 1; i <= n; ++i) {
            zero_sem.acquire();      // wait for zero semaphore
            printNumber(0);
            if (i % 2 == 0) {
                even_sem.release();   // release even semaphore
            } else {
                odd_sem.release();    // release odd semaphore
            }
        }
    }

    void even(function<void(int)> printNumber) {
        for (int i = 2; i <= n; i += 2) {
            even_sem.acquire();       // wait for even semaphore
            printNumber(i);
            zero_sem.release();       // release zero semaphore
        }
    }

    void odd(function<void(int)> printNumber) {
        for (int i = 1; i <= n; i += 2) {
            odd_sem.acquire();        // wait for odd semaphore
            printNumber(i);
            zero_sem.release();       // release zero semaphore
        }
    }
};
```

### Mutex + Condition Variable Solution

In contrast to the above semaphore that uses `i % 2` to decide semaphore to send to which thread, mutex + condition variable directly instructs which thread to wake up.

`cv.wait(...)` explicitly puts thread to sleep on condition, then the wake up check happens when seen `notify_all`/`notify_one`.

```cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

using namespace std;

class ZeroEvenOdd {
private:
    int n;
    int current = 1;
    mutex mtx;
    condition_variable cv;
    bool is_zero = true;
    bool is_odd = true;

public:
    ZeroEvenOdd(int n) : n(n) {}

    void zero(function<void(int)> printNumber) {
        for (int i = 0; i < n; ++i) {
            unique_lock<mutex> lock(mtx);
            cv.wait(lock, [this] { return is_zero; }); // wait for is_zero to be true
            printNumber(0);
            is_zero = false;
            cv.notify_all(); // wake up even or odd
        }
    }

    void even(function<void(int)> printNumber) {
        for (int i = 2; i <= n; i += 2) {
            unique_lock<mutex> lock(mtx);
            cv.wait(lock, [this] { return !is_zero && !is_odd; }); // wait for non-zero and non-odd
            printNumber(i);
            is_zero = true;
            is_odd = true;
            cv.notify_one(); // wake up zero
        }
    }

    void odd(function<void(int)> printNumber) {
        for (int i = 1; i <= n; i += 2) {
            unique_lock<mutex> lock(mtx);
            cv.wait(lock, [this] { return !is_zero && is_odd; }); // wait for non-zero and is_odd
            printNumber(i);
            is_zero = true;
            is_odd = false;
            cv.notify_one(); // wake up zero
        }
    }
};
```

## Semaphore Underlying Implementation

When put to sleep, semaphore relying on OS, e.g., Linux `futex` to put thread in a thread queue to be picked up awaiting a semaphore permission (non-zero sempahore counter to a thread).

`futex` (Fast User-space Mutex) is a linux kernel efficient synchronization mechanism that features

* User-Space First: detected frequent switches of mutex-like scenario, keep the lock and threads in user space
* Kernel-Space Delegation: severe lock contention that a thread has not yet obtained a lock for a long time, such threads are put in queue in kernel space

### Comparison Semaphore vs Mutex

#### Key Insight Why Mutex Is Fast In Wake Up

Mutex puts thread sleep in user space; semaphore puts thread sleep in kernel space.

If put in kernel space, the physical threads are different in terms of the declared threads used in user space code, and CPU caches are lost as a result of context switch.

#### Semaphore vs Mutex Comparison Table

||Mutex|Semaphore|
|-|-|-|
|CPU Utilization|High (keep spinning checking CPU availability)|Low (sleep, released CPU)|
|Context Switch|No|Yes|
|Latency|Low (for no context switch)|High (for possible context switch)|
|Use Case Scenarios|Short-time wait, e.g., spin lock|Long-time wait, e.g., I/O operation, process synchronization|
