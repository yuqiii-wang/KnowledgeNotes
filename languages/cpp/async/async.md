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