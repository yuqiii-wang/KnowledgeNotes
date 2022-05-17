# STL 

## Containers

vector、deque、stack/queue、map/set、unordered_map/unordered_set

### `std::map`

`std::map` uses red-black tree, that renders high computation cost when having too many insertion and deletion operation (to balance the tree). Read/write operations on the map is $O\big(log(n)\big)$.

### `std::string`



### `std::unordered_map`

`std::unordered_map` does not have order hence the read/write operations on the map is always $O(1)$

### `std::vector`

A vector grows exponentially, such as by $2^n$: $2$, $4$, $8$, $16$ ..., and every time it grows, there need copy operations that assigns elements from old memory to new memory addresses.

### `std::list`

`insert()`/`erase()` in a list need $O(n)$ since it iterates over the whole list to determine where to insert/delete an element.

`push_back()` only needs $O(1)$.

### `std::array`

`std::array` is a container that encapsulates fixed size arrays, init by such as 
```cpp
std::array<int, 3> a {1, 2, 3};
```

### `std::deque`

`std::deque` (double-ended queue) is an indexed sequence container that allows fast insertion and deletion at both its beginning and its end.

Deque implements pointer to the queue's first and end elements, while elements inside are chopped into chunks, each chunk is a vector, linked through a map:

![deque](imgs/deque.png "deque")


## Tools

### `std::sort()`

The algorithm used by `sort()` is *IntroSort*. Introsort being a hybrid sorting algorithm uses three sorting algorithm to minimize the running time, *Quicksort*, *Heapsort* and *Insertion Sort*. 

## `condition_variable`

`condition_variable` is a synchronization primitive that can be used to block a thread, or multiple threads at the same time, until another thread both modifies a shared variable (the condition), and notifies the condition_variable. 

A typical usage is force return an executing function when timeout:
1. Pass a bond function `f` to a timeout service function
2. Set a mutex lock
3. Branch a thread that runs the func `f`, notifies the main thread when finished
4. Conditional Var throws exception either on timeout or unlocked mutex
5. If Conditional Var does not throw exception, the function return with success
```cpp
template<typename T_ret, typename... T_inputs>
T_ret SpClient::functionWrapperReturnWhenTimeout(std::function<T_ret(T_inputs ...)>& f, T_inputs inputs...) {

    std::mutex mutex_FunctionWrapperReturnWhenTimeout;
    std::condition_variable condVar_FunctionWrapperReturnWhenTimeout;
    T_ret result = 0;

    std::unique_lock<std::mutex> mutexLock_FunctionWrapperReturnWhenTimeout(mutex_FunctionWrapperReturnWhenTimeout);

    std::thread thread_functionWrapperReturnWhenTimeout([&]() {
        result = f(arg1, arg2, arg3);
        condVar_FunctionWrapperReturnWhenTimeout.notify_one();
    });
    thread_functionWrapperReturnWhenTimeout.detach();

    if (condVar_FunctionWrapperReturnWhenTimeout.wait_for(
        mutexLock_FunctionWrapperReturnWhenTimeout, std::chrono::seconds(20)) == 
            std::cv_status::timeout)  {
        throw std::runtime_error("Timeout");
    }

    return result; 
}
```