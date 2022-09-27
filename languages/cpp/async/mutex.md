# Mutex Under the hood

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

The `LOCK` prefix ensures that the CPU has exclusive ownership of the appropriate cache line for the duration of the operation, and provides certain additional ordering guarantees. This may be achieved by asserting a bus lock, but the CPU will avoid this where possible. If the bus is locked then it is only for the duration of the locked instruction.

## `mutex` and `spin_lock`

`Spinlock` is a lock which causes a thread trying to acquire it to simply wait in the loop and repeatedly check for its availability. In contrast, a `mutex` is a program object that is created so that multiple processes can take turns sharing the same resource. 

A thread reached `mutex` immediately goes to sleep, until waken by `mutex.unlock()`; while for `spinlock`, a thread periodically checks it.

## `lock_guard`

```cpp
template< class Mutex >
class lock_guard;
```

The class lock_guard is a mutex wrapper that provides a convenient RAII-style mechanism for owning a mutex for the duration of a scoped block.

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

## `std::condition_variable`

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
* 


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