# Coroutine

A *coroutine* is a special function that can suspend its execution and resume later at the exact point where execution was suspended. When suspending, the function may return (yield) a value. When coroutine ends execution it also may return a value. 

When a coroutine is suspended, its state is copied into an allocated object that represents the state of the coroutine.

`range` in the code below is a coroutine function, that it returns `co_yield v`, then suspend operation, and resume after `co_yield v` is retrieved.
```cpp
template<typename T> 
std::unique_generator<T> range(T fromInclusive, T toExclusive) { 
    for (T v = fromInclusive; v < toExclusive; ++v) { 
        co_yield v; 
    }
}

int main() { 
    for (auto val : range(1, 10)) { 
        std::cout << val << '\n'; 
    } 
} 
```