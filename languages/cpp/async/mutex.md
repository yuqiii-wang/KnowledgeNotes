# Mutex Under the Hood

## Mutex 

### Test-and-set

The *test-and-set* instruction is an instruction used to write (set) 1 to a memory location and return its old value as a single atomic (i.e., non-interruptible) operation. 

If multiple processes may access the same memory location, and if a process is currently performing a test-and-set, no other process may begin another test-and-set until the first process's test-and-set is finished.

Atomic mutex operation is built on the concept of test-and-set: the mutex memory is zero prior to any process holding this mutex, and this mutex is set to one once acquired by a process. When a process releases this mutex, it sets the memory back to zero.

If a process uses `spin_lock`, this process periodically checks this mutex, otherwise, this process is put to sleep until received a wake-up signal.

### Assembly Implementation

Given a shared variable `shared_val` between multiple threads, once one thread reaches to `CMP [shared_val],BL` that tests if the `shared_val` is free: not freed, goto `OutLoop2`, otherwise it is free, use `LOCK CMPXCHG [shared_val],BL` to set the lock.

```x86asm
; BL is the mutex id
; shared_val, a memory address

CMP [shared_val],BL ; Perhaps it is locked to us anyway
JZ .OutLoop2
.Loop1:
CMP [shared_val],0xFF ; Free
JZ .OutLoop1 ; Yes
pause ; equal to rep nop.
JMP .Loop1 ; Else, retry

.OutLoop1:

; Lock is free, grab it
MOV AL,0xFF
LOCK CMPXCHG [shared_val],BL
JNZ .Loop1 ; Write failed

.OutLoop2: ; Lock Acquired
```

where `CMPXCHG` performs comparison between the value in the AL, AX, EAX, or RAX register and the first operand (destination operand). If the two values are equal, the second operand (source operand) is loaded into the destination operand. Otherwise, the destination operand is loaded into the AL, AX, EAX or RAX register.

`LOCK` assembly is an instruction prefix, which applies to read-modify-write instructions such as `INC`, `XCHG`, `CMPXCHG`. 

The `LOCK` prefix ensures that the CPU has exclusive ownership of the appropriate cache line for the duration of the operation, and provides certain additional ordering guarantees.
This may be achieved by asserting a bus lock, but the CPU will avoid this where possible. 
If the bus is locked then it is only for the duration of the locked instruction.

### `mutex` and `spin_lock`

`Spinlock` is a lock which causes a thread trying to acquire it to simply wait in the loop and repeatedly check for its availability. 
In contrast, a `mutex` is a program object that is created so that multiple processes can take turns sharing the same resource. 

* Mutex: when critical resource was used by other threads, it goes to sleep, release CPU to other threads
* Spinlock: when critical resource was used by other threads, it just wait, will NOT give CPU to other threads

A thread reached `mutex` immediately goes to sleep, until waken by `mutex.unlock()`; while for `spinlock`, a thread periodically checks it.

### Mutex `lock()` vs `try_lock()`

`lock()` blocks other threads, and once other threads shave reached the `lock()`, OS puts them into sleep.
When the `lock()`'s owned thread exits/reached `unlock()`, OS wakes up one of the other asleep threads, and this awaken thread then owns the `lock()`.
If `std::condition_variable` is used, thread wake up is triggered on the specified condition.

`try_lock()` immediately returns, released to do other work, usually used with `if` statement.

```cpp
#include <iostream>
#include <thread>
#include <mutex>

std::mutex mtx;

void test_lock(int id) {
    mtx.lock(); // Lock the mutex
    std::cout << "Thread " << id << std::endl;
    mtx.unlock(); // Unlock the mutex
}

void test_try_lock(int id) {
    if (mtx.try_lock()) { // Attempt to lock the mutex
        std::cout << "Thread " << id << " acquired the lock." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Simulate work
        mtx.unlock(); // Unlock the mutex
    } else {
        std::cout << "Thread " << id << " could not acquire the lock." << std::endl;
    }
}

int main() {
    std::thread t1(test_lock, 1);
    std::thread t2(test_lock, 2);
    t1.join();
    t2.join();

    std::thread t3(test_try_lock, 3);
    std::thread t4(test_try_lock, 4);
    t3.join();
    t4.join();

    return 0;
}
```

## `lock_guard`

```cpp
template< class Mutex >
class lock_guard;
```

The class `lock_guard` is a mutex wrapper that provides a convenient RAII-style mechanism for owning a mutex for the duration of a scoped block.

When a lock_guard object is created, it attempts to take ownership of the mutex it is given. When control leaves the scope in which the lock_guard object was created, the lock_guard is destructed and the mutex is released.

```cpp
#include <thread>
#include <mutex>
#include <iostream>
 
int g_i = 0;
std::mutex g_i_mutex;  // protects g_i
 
void safe_increment()
{
    const std::lock_guard<std::mutex> lock(g_i_mutex);
    ++g_i;
 
    std::cout << "g_i: " << g_i << "; in thread #"
              << std::this_thread::get_id() << '\n';
 
    // g_i_mutex is automatically released when lock
    // goes out of scope
}
 
int main()
{
    std::cout << "g_i: " << g_i << "; in main()\n";
 
    std::thread t1(safe_increment);
    std::thread t2(safe_increment);
 
    t1.join();
    t2.join();
 
    std::cout << "g_i: " << g_i << "; in main()\n";
}
```

### `std::condition_variable`

The `condition_variable` class is a synchronization primitive that can be used to block a thread, or multiple threads at the same time, until another thread both modifies a shared variable (the condition), and notifies the `condition_variable`.

The thread that intends to modify the shared variable has to

1. acquire a `std::mutex` (typically via `std::lock_guard`); If the mutex is unlocked, there might be error for wake-up signal, since thread has been running or even has exited since there is no mutex lock.
2. perform the modification while the lock is held
3. execute `notify_one` or `notify_all` on the `std::condition_variable` (the lock does not need to be held for notification)

### Practice Details

* `std::lock_guard` releases its own lock when gone out of scope
* `notify_one` wakes up one thread (user cannot choose which thread to wake up), whereas `notify_all` wakes up all threads
* `std::condition_variable::wait` is equivalent to

```cpp
while (!stop_waiting()) {
    wait(lock);
}
```

Note that lock must be acquired before entering this method, and it is reacquired after wait(lock) exits, which means that lock can be used to guard access to stop_waiting().

### Example

The code below works like this:

* `std::condition_variable cv;` and `std::mutex m;` are shared among the main and worker thread for notification and flow control purposes.

```cpp
#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>

std::mutex m;
std::condition_variable cv;
std::string data;
bool ready = false;
bool processed = false;
 
void worker_thread()
{
    // Wait until main() sends data
    std::unique_lock lk(m);
    cv.wait(lk, []{return ready;});
 
    // after the wait, we own the lock.
    std::cout << "Worker thread is processing data\n";
    data += " after processing";
 
    // Send data back to main()
    processed = true;
    std::cout << "Worker thread signals data processing completed\n";
 
    // Manual unlocking is done before notifying, to avoid waking up
    // the waiting thread only to block again (see notify_one for details)
    lk.unlock();
    cv.notify_one();
}
 
int main()
{
    std::thread worker(worker_thread);
 
    data = "Example data";
    // send data to the worker thread
    {
        std::lock_guard lk(m);
        ready = true;
        std::cout << "main() signals data ready for processing\n";

        // lk is released as it goes out of scope
    }
    cv.notify_one();
 
    // wait for the worker
    {
        std::unique_lock lk(m);
        cv.wait(lk, []{return processed;});

        // lk is released as it goes out of scope
    }
    std::cout << "Back in main(), data = " << data << '\n';
 
    worker.join();
}
```

## Deadlock Debug

Given an example of two threads having deadlocks to each other shown as below,
that thread1 acquires `mutex1` then `mutex2`; thread2 acquires `mutex2` then `mutex1`.
Thread1 wants to lock `mutex2` then releases `mutex1`, 
while thread2 wants to lock `mutex1` then releases `mutex2`, hence reached a deadlock.

```cpp

pthread_mutex_t mutex1;
pthread_mutex_t mutex2;


void *ThreadWork1(void *arg)
{
  int *p = (int*)arg;
  pthread_mutex_lock(&mutex1);
  
  sleep(2);
  
  pthread_mutex_lock(&mutex2);
  pthread_mutex_unlock(&mutex2);
  pthread_mutex_unlock(&mutex1);
  return NULL;
}

void *ThreadWork2(void *arg)
{
  int *p = (int*)arg;
  pthread_mutex_lock(&mutex2);
  
  sleep(2);
  
  pthread_mutex_lock(&mutex1);
  pthread_mutex_unlock(&mutex1);
  pthread_mutex_unlock(&mutex2);
  return NULL;
}
```

GDB can help find the deadlock:

1. `ps -elf | grep <your_program_name>`
2. `sudo gdb attach <pid>` (might need `sudo`)
3. inside gdb: `thread apply all bt` to check running threads
4. inside gdb: `t <thread_num>` goto inside this tread
5. inside gdb one thread: `f <stack_num>` goto a particular stack
6. inside gdb one thread one stack: should see mutex reaching a dead lock

Below is a result from `thread apply all bt`, that both thread 2 and thread 2 are in a lock state.

<div style="display: flex; justify-content: center;">
      <img src="imgs/deadlock_debug_all_threads.png" width="40%" height="40%" alt="deadlock_debug_all_threads" />
</div>
</br>

## Read/Write Deadlock

Often for many threads accessing one shared variable, they want to just read the variable, not writing it. 
Concurrent read should be safe.

Example shown as below that,
`get()` should allow concurrent read, while `increment()` and `reset()` should have exclusive ownership of `_mutex`.

When `increment()` or `reset()` runs, the shared lock becomes an exclusive lock, hence preventing read operation `get()`.

```cpp
class ThreadSafeCounter {
 public:
  ThreadSafeCounter() = default;
 
  // Multiple threads/readers can read the counter's value at the same time.
  unsigned int get() const {
    std::shared_lock lock(mutex_);
    return value_;
  }
 
  // Only one thread/writer can increment/write the counter's value.
  void increment() {
    std::unique_lock lock(mutex_);
    ++value_;
  }
 
  // Only one thread/writer can reset/write the counter's value.
  void reset() {
    std::unique_lock lock(mutex_);
    value_ = 0;
  }
 
 private:
  mutable std::shared_mutex mutex_;
  unsigned int value_ = 0;
};
```

### Mutex Example on `i++`

`i++` (assumed that `i` is a static variable) can be compiled to the below assembly code.

```asm
MOV [idx], %eax
INC %eax
MOV %eax, [idx]
```

Without CPU pipelining, to get the right result, there should be
|Thread 1|Thread 2|
|-|-|
|`MOV [idx], %eax`||
|`INC %eax`||
|`MOV %eax, [idx]`||
||`MOV [idx], %eax`|
||`INC %eax`|
||`MOV %eax, [idx]`|

However, likely there might be 
|Thread 1|Thread 2|
|-|-|
|`MOV [idx], %eax`||
||`MOV [idx], %eax`|
||`INC %eax`|
||`MOV %eax, [idx]`|
|`INC %eax`||
|`MOV %eax, [idx]`||

where `[idx]` is a global variable while `%eax` is CPU/thread specific.

## Lock Expense and Practices

Depending on hardware, one thread's `mutex.unlock()` triggering another thread's `mutex.lock()` usually costs ranged from $10^{-6}$ up to $10^{-4}$ seconds, during which time OS needs to do context switch for the other thread prepared for `mutex.lock()`.

### Read/Write Lock

Intuitively speaking, read lock is required when reading, write lock is required when writing.

```cpp
static int i;
pthread_rwlock_t rwlock;

// below code in multi-threading
{
    pthread_rwlock_wrlock(&rwlock);
    i++;
    pthread_rwlock_unlock(&rwlock);   
}
```

### Spinlock

Spinlock can be used as below

```cpp
static int i;
pthread_spinlock_t spinlock;

// below code in multi-threading
{
    pthread_spin_lock(&spinlock);
    i++;
    pthread_spin_unlock(&spinlock);   
}
```
