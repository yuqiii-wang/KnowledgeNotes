# Some C++ Advanced Knowledge

## Run-Time Type Information (RTTI)

*Run-time type information* or *run-time type identification* (RTTI) is a feature that exposes information about an object's data type at runtime. 

The `typeid` keyword is used to determine the class of an object at run time. 
It returns a reference to `std::type_info` object (`typeid` operator returns `std::type_info`), which exists until the end of the program.

Below code shows that inside `Person* ptr = &employee;`, the type of `ptr` is unknown until in run-time env, since it is the dereference of a pointer to a polymorphic class.

For `Person* ptr = &employee;`, the pointer type is a `Person`; the pointed obj is an `Employee`. For `Person& ref = employee;`, the reference result is a `Person`.

```cpp
#include <iostream>
#include <typeinfo>

class Person {
public:
    virtual ~Person() = default;
};

class Employee : public Person {};

int main() {
    Person person;
    Employee employee;
    Person* ptr = &employee;
    Person& ref = employee;
    
    // The string returned by typeid::name is implementation-defined.
    std::cout << typeid(person).name()
              << std::endl;  // Person (statically known at compile-time).
    std::cout << typeid(employee).name()
              << std::endl;  // Employee (statically known at compile-time).
    std::cout << typeid(ptr).name()
              << std::endl;  // Person* (statically known at compile-time).
    std::cout << typeid(*ptr).name()
              << std::endl;  // Employee (looked up dynamically at run-time
                             //           because it is the dereference of a
                             //           pointer to a polymorphic class).
    std::cout << typeid(ref).name()
              << std::endl;  // Employee (references can also be polymorphic)
    return 0;
}
```

### `std::type_info`

`std::type_info::name` returns an implementation-defined type name (*name mangling*).

Name mangling (also called name decoration) is a technique used to resolve unique name representation, providing a way of encoding additional information in the name of a function, structure, class or another datatype in order to pass more semantic information from the compiler to the linker.

So that, the type defined by `static_cast<new-type> obj` returns `new-type` at compile time, while `dynamic_cast<new-type> obj` checks what the `obj` actually is by name mangling representation. As a result,  inside `Person* ptr = &employee;`, `ptr` is `Person*` while `ptr*` refers to `Employee`.

## Cache Warmup

For some barely executed code (called *critical code*), cache is unlikely maintaining the data.
Once executed, CPU needs to fetch data from the main memory (both instructions and data).

Solution is by warming up the code by periodically simulating calling the critical code.

For example, in the code below, in most business scenarios, `buy_stock();` would not happen until reaching a good stock price.
The `buy_stock();` can be said critical code and would not be kept in cache by CPU.
```cpp
void run() {
    while (true) {
        auto stock_price = get_stock_price();
        bool should_buy_stock = compute_to_judge_action(stock_price);
        if (!should_buy_stock) {
            update_stock_price_history(stock_price);
        }
        else {
            buy_stock();
        }
    }
}
```

Instead, define `buy_stock(bool isSimulation=true);` to simulate some behavior of `buy_stock();`, except for the last actual action sending order to exchange server.

```cpp

Order ord;

buy_stock(bool isSimulation=true) {
    if (isSimulation) {
        
    }
}
```

In implementation, set a counter `atomic<int> count;` then by `if (count % simulationInterval = 0)` to trigger once simulation buying stock `buy_stock(true);`.
The `const int simulationInterval = 20;` should be carefully tested so that it does not run too frequently as overhead, nor of low frequency to prevent cache miss.


```cpp
const int simulationInterval = 20;
void run() {
    atomic<int> count{0};
    while (true) {
        auto stock_price = get_stock_price();
        bool should_buy_stock = compute_to_judge_action(stock_price);
        count++;
        if (!should_buy_stock) {
            update_stock_price_history(stock_price);
            if (count % simulationInterval = 0) {
                buy_stock(true);
            }
        }
        else {
            buy_stock(false);
        }
    }
}
```

For testing cache warmup, use TSC for non-simulation `buy_stock(false);`.
For a very large interval such as `const int simulationInterval = 10000;`, should see large `auto elapsed_cnt = end - start;`; otherwise, should see small `elapsed_cnt` for small interval `const int simulationInterval = 10;`.

```cpp
unsigned cycles_low0, cycles_high0, cycles_low1, cycles_high1;

asm volatile (
        "RDTSC\n\t"
        "mov %%edx, %0\n\t"
        "mov %%eax, %1\n\t": "=r" (cycles_high0), "=r" (cycles_low0)
);

buy_stock(false);

asm volatile (
        "RDTSC\n\t"
        "mov %%edx, %0\n\t"
        "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)
);

auto start = ( ((uint64_t)cycles_high0 << 32) | cycles_low0 );
auto end = ( ((uint64_t)cycles_high1 << 32) | cycles_low1 );

auto elapsed_cnt = end - start;
```