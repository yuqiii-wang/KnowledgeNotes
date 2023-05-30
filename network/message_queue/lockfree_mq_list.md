# Lock-Free Message Queue (List)

Lock-free message queue usually refers to using *Compare And Swap* (CAS) `compare_exchange_weak(...)` to guarantee thread safety avoid lock-introduced, kernel space/user space switch, and thread-switch related costs.

```cpp
// since c++11
bool compare_exchange_weak( T& expected, T desired,
                            std::memory_order order =
                                std::memory_order_seq_cst ) noexcept;
```