# Template

## Template and Instantiation Separation

It is required that user/programmer should explicitly define instantiation of template if there are more than one exact match templates.

### Example

Given a template and its instantiation as below
```cpp
// foo.h

template <typename T>
void foo(T value);
```

```cpp
// foo.cpp

#include "foo.h"
#include <iostream>

template <typename T>
void foo(T value) {
    std::cout << value << std::endl;
}
```

The below code throw Linker Error for not defined `foo<int>(int)`
```cpp
// main.cpp

#include "foo.h"

int main() {
    foo(42);
    return 0;
} 
```

Explicit Instantiation is required in `foo.cpp`
```cpp
// foo.cpp

#include "foo.h"
#include <iostream>

template <typename T>
void foo(T value) {
    std::cout << value << std::endl;
}

template void foo<int>(int);  // Explicit instantiation
```

## `auto` Deduction

If source var is a lvalue, `auto&&` returns a reference; if source var is a rvalue, `auto&&` returns a universal reference.
```cpp
auto x = 27;           // (x is neither a pointer nor a reference), x's type is int
const auto cx = x;     // (cx is neither a pointer nor a reference), cs's type is const int
const auto& rx = x;    // (rx is a non-universal reference), rx's type is a reference to a const int

auto&& uref1 = x;      // x is int and lvalue, so uref1's type is int&
auto&& uref2 = cx;     // cx is const int and lvalue, so uref2's type is const int &
auto&& uref3 = 27;     // 27 is an int and rvalue, so uref3's type is int&&
```

Use of `std::initializer_list<T>`
```cpp
auto x1 = 27;          // type is int, value is 27
auto x2(27);           // type is int, value is 27
auto x3 = { 27 };      // type is std::initializer_list<int>, value is { 27 }
auto x4{ 27 };         // type is std::initializer_list<int>, value is { 27 }
                       // in some compilers type may be deduced as an int with a 
                       // value of 27. See remarks for more information.
auto x5 = { 1, 2.0 }   // error! can't deduce T for std::initializer_list<t>
```

`auto` can work on function return.
```cpp
// f returns int:
auto f() { return 42; }
// g returns void:
auto g() { std::cout << "hello, world!\n"; }
```

`decltype(auto)` deduces a type using the type deduction rules of decltype rather than those of `auto`
```cpp
int* p = new int(42);
auto x = *p;           // x has type int
decltype(auto) y = *p; // y is a reference to *p
```

## Variadic Template