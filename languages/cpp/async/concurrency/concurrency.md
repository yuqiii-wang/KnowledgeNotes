# Concurrency 

## Prefer task-based programming to thread-based

* Thread-based:
```cpp
int doAsyncWork();
std::thread t(doAsyncWork);
```

* Task-based
```cpp
// "fut" for "future"
auto fut = std::async(doAsyncWork);
```

### Advantages of using `std::async`

* Avoid the complications of dealing with threading:
1. Hardware threads: CPU cores: threads are a limited resource, and exception might be thrown for lack of thread available.
2. Software threads (also known as OS threads or system threads), managing message passing between processes: there are context switches when too many threads are running simultaneously
3. `std::threads` are objects in a C++ process that act as handles to underlying
software threads: 

### Use Example of `async`

```cpp
template<typename F, typename... Ts>
inline std::future<typename std::result_of<F(Ts...)>::type>
reallyAsync(F&& f, Ts&&... params)
{
    // for asynchronous
    return std::async(std::launch::async, // call to f(params...)
    std::forward<F>(f),
    std::forward<Ts>(params)...);
}

// run f asynchronously;
// throw if std::async
// would throw
auto fut = reallyAsync(f);
```

### Policy: Single- or Multi- Threading

Multithreading programming is all about concurrent execution of different functions. Async programming is about non-blocking execution between functions, and we can apply async with single-threaded or multithreaded programming.

`std::async` can either use a single or separate thread for async operations:

* `std::launch::async`

the task is executed on a different thread, potentially by launching a new thread
```cpp
// Calls barPrint("hello"); with async policy
// prints "43" concurrently
auto a3 = std::async(std::launch::async, barPrint, "hello");
```

* `std::launch::deferred`

the task is executed on the calling thread the first time its result is requested (lazy evaluation). The main thread might pick sleep/free time to handle the async operation, rather than launching a new thread.
```cpp
// Calls barPrint("world!") with deferred policy
// prints "world!" when a.get() or a.wait() is called
auto a = std::async(std::launch::deferred, barPrint, "world!");
sleep(1);
a.wait();
```
