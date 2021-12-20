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

The constexpr specifier declares that it is possible to evaluate the value of the function or variable at compile time. 