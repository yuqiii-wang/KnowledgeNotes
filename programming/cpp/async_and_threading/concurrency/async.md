# Async

## Promise and Future

Future and Promise are the two separate sides of an asynchronous operation.

`std::promise` is used by the "producer/writer" of the asynchronous operation.

`std::future` is used by the "consumer/reader" of the asynchronous operation.

### Only use `set_value(value)` and `get()` once

The following part of code will fail for `set_value/set_exception` multiple invocations raise exceptions.
```cpp
std::promise<int> send_value;
std::future<int> receive_value = send_value.get_future();

std::thread t1 = std::thread([&]() {
    while (!exit_flag) {
        int value = my_custom_function_1();
        send_value.set_value(value);
    }
});

std::thread t2 = std::thread([&]() {
    while (!exit_flag) {
        int value = receive_value.get();
        my_custom_function_2(value);
    }
});
```

### `std::async`

The function template `async` runs the function f asynchronously and returns a `std::future` that will eventually hold the result of that function call.

## Thread

### `std::thread::detach`

Separates the thread of execution from the thread object, allowing execution to continue independently. Any allocated resources will be freed once the thread exits. 

Otherwise, `std::terminate` would have killed the thread at `}` (thread out of scope).

### thread with a class member function

Threading with non-static member function by `&` and `this`, together they provide invocation to an object's method rather than a non-static class member function. 
```cpp
std::thread th1(&ClassName::classMethod, this, arg1, arg2);
```

### `std::atomic`

If `a` is accessed and modifiable by multiple threads, `atomic` (preserve atomicity to POD data types) is required.

```cpp
mutable std::atomic<unsigned> a{ 0 };
```

## `std::async`

```cpp
template< class Function, class... Args >
[[nodiscard]]
std::future<std::invoke_result_t<std::decay_t<Function>, std::decay_t<Args>...>>
    async( Function&& f, Args&&... args );
```

The function template async runs the function `f` asynchronously (potentially in a separate thread which might be a part of a thread pool) and returns a std::future that will eventually hold the result of that function call.

### C++ Attribute: `[[nodiscard]]`

If a function declared `nodiscard` or a function returning an enumeration or class declared `nodiscard` by value is called from a discarded-value expression other than a cast to `void`, the compiler is encouraged to issue a warning.

### `std::result_of` and `std::invoke_result`

They are used to deduces the return type of an *INVOKE expression* (a.k.a *Callable*) at compile time.

A Callable type is a type for which the INVOKE operation (used by, e.g., `std::function`, `std::bind`, and `std::thread::thread`) is applicable.

```cpp
template< class F, class... ArgTypes >
class result_of<F(ArgTypes...)>;

template< class F, class... ArgTypes >
class invoke_result;
```
where `F` must be a callable type.

For example,
```cpp
struct S
{
    double operator()(char, int&);
    float operator()(int) { return 1.0;}
};

void main()
{
    // the result of invoking S with char and int& arguments is double
    std::result_of<S(char, int&)>::type d = 3.14; // d has type double
    static_assert(std::is_same<decltype(d), double>::value, "");
}
```

### `std::decay`

Applies lvalue-to-rvalue, array-to-pointer, and function-to-pointer implicit conversions to the type `T`.

```cpp
template< class T >
struct decay;
```

### `std::launch::async` vs `std::launch::deferred`

policy	-	bitmask value, where individual bits control the allowed methods of execution
|Bit|	Explanation|
|-|-|
|`std::launch::async`|	enable asynchronous evaluation|
|`std::launch::deferred`|	enable lazy evaluation|

* If the `async` flag (default) is set (i.e. (policy & std::launch::async) != 0), then `async` executes the callable object `f` **on a new thread of execution** (with all thread-locals initialized) as if spawned by `std::thread(std::forward<F>(f), std::forward<Args>(args)...)`, except that if the function `f` returns a value or throws an exception, it is stored in the shared state accessible through the `std::future` that `async` returns to the caller.

* If the `deferred` flag is set (i.e. (`policy & std::launch::deferred) != 0`), then `async` converts `f` and `args...` the same way as by `std::thread` constructor, but does not spawn a new thread of execution. 
Instead, lazy evaluation is performed: the first call to a non-timed wait function on the `std::future` that async returned to the caller will cause the copy of `f` to be invoked (as an rvalue) with the copies of `args...` (also passed as rvalues) in the current thread (which does not have to be the thread that originally called `std::async`). 
The result or exception is placed in the shared state associated with the future and only then it is made ready. All further accesses to the same `std::future` will return the result immediately.