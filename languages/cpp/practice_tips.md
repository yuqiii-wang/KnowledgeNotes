# Some Practice Tips

* `std::endl` must be appened to `std::cout`

The reason for this is that typically endl flushes the contents of the stream.

You must implement at least one `std::endl` before exit of a program.

* `std::queue`, `std::deque` and `std::stack`

`deque`: Double ended queue, insert and remove from both ends

`queue`: insert only in one end and remove from the other (first in first out)

`stack`: LIFO context (last-in first-out)

* i++ vs ++i

++i will increment the value of i, and then return the incremented value.
```cpp
int i = 1;
int j = ++i;
// (i is 2, j is 2)
```
i++ will increment the value of i, but return the original value that i held before being incremented.
```cpp
int i = 1;
int j = i++;
// (i is 2, j is 1)
```

* Private, Protected and Friend

* GDB Common Debugging Practices

* std::vector<bool>

`std::vector<bool>` contains boolean values in compressed form using only one bit for value (and not 8 how bool[] arrays do). It is not possible to return a reference to a bit in c++, 

* template

cpp template functions must be in `#include` with their implementation, which means, for example, in header files, template should be materialized with definitions rather than a pure declaration.

Example.hpp
```cpp
class Example {
    template<typename T>
    T method_empty(T& t);

    template<typename T>
    T method_realized(T& t){
        return t;
    }
};
```
Example.cpp
```cpp
#include "Example.cpp"
T Example::method_empty(T& t){
    return t;
}
```
main.cpp
```cpp
#include "Example.cpp"

int main(){
    Example example;
    int i = 1;
    example.method_empty(i); // linker err, definition must be in Example.hpp
    example.method_realized(i); // ok
    return 0;
}
```

* `constexpr`

The constexpr specifier declares that it is possible to evaluate the value of the function or variable at compile time. It is used for compiler optimization and input params as `const`

For example:
```cpp
constexpr int multiply (int x, int y) return x * y;
extern const int val = multiply(10,10);
```
would be compiled into
```as
push    rbp
mov     rbp, rsp
mov     esi, 100 //substituted in as an immediate
```
since `multiply` is constexpr.

While, 
```cpp
const int multiply (int x, int y) return x * y;
const int val = multiply(10,10);
```
would be compile into 
```as
multiply(int, int):
        push    rbp
        mov     rbp, rsp
        mov     DWORD PTR [rbp-4], edi
        mov     DWORD PTR [rbp-8], esi
        mov     eax, DWORD PTR [rbp-4]
        imul    eax, DWORD PTR [rbp-8]
        pop     rbp
        ret
...
__static_initialization_and_destruction_0(int, int):
...
        call    multiply(int, int)
```

* lambda

example
```cpp
float x[5] = {5,4,3,2,1};
std::sort(x, x + n,
    [](float a, float b) {
        return (std::abs(a) < std::abs(b));
    }
);
```

`[]` is called capture clause, that `[=]` means by value capture while `[&]` is by reference capture

* string types

`std::string` is allocates memory in **a single block** (as needed, and sometimes preemptively), and best practice suggests pre-computing the string size the filling it.

`std::stringstream`, `std::istringstream` and `std::ostringstream` **1)** better express the intent to appending strings by `<<` and `>>` respectively. **2)** A stringstream writes into a stringbuffer, which usually means **a linked list of buffers**. Memory is not continuous, and it requires no reallocations as the buffer grows.


They are interchangeable via the following
```cpp
const std::string tmp = stringstream.str();
const char* cstr = tmp.c_str();
```

* `std::atomic`

`std::atomic` works with trivial copyables (such as C-compatible POD types, e.g., int, bool, char) to guarantee thread safety (defined behavior when one thread read and another thread write) by trying squeezing one operation into one cpu cycle (one instruction).

It is **NOT** allowed to apply atomic to an array such as
```cpp
std::atomic<std::array<int,10>> myArray;
```
in which `myArray`'s element is not readable/writable.

* constructors

default
```cpp
// class construct
construct(){}
// main()
construct c;
```

parameterized
```cpp
// class construct
construct(int a){}
// main()
int a = 1;
construct c(a);
```

copy construct (only shallow copy)
using `=` copy assignment without defined copy constructor in class is undefined behavior.
```cpp
// class construct
construct(){}
construct(const construct&){} // copy construct
// main()
construct c1;
construct c2 = c1; // copy construct
construct c3(c1); // also a copy construct
```

* `new` vs `malloc`

`new` allocates memory and calls constructor for object initialization. But `malloc()` allocates memory and does not call constructor.

Return type of `new` is exact data type while `malloc()` returns `void*`.

* exceptions

`try`/`catch` cannot catch all exceptions, some typical are

1) divided by zero

2) segmentation fault/access out of range