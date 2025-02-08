# Initialization

Different ways of initialization:
```cpp
int x(0); // initializer is in parentheses
int y = 0; // initializer follows "="
int z{ 0 }; // initializer is in braces
int zz = { 0 }; // initializer uses "=" and braces, same as above
```

There are distinct constructor invocations given using diff ways of init, that
```cpp
std::vector<int> v1(10, 20); // use non-std::initializer_list
// ctor: create 10-element
// std::vector, all elements have
// value of 20
std::vector<int> v2{10, 20}; // use std::initializer_list ctor:
// create 2-element std::vector,
// element values are 10 and 20
```

Use of `std::initializer_list<T>` should be preferred when using `{}` ctor.

List initialization `{}` does not allow narrowing that adds safety in initialization.

```cpp
void fun(double val, int val2) {

    int x2 = val;    // if val == 7.9, x2 becomes 7 (bad)

    char c2 = val2;  // if val2 == 1025, c2 becomes 1 (bad)

    int x3 {val};    // error: possible truncation (good)

    char c3 {val2};  // error: possible narrowing (good)

    char c4 {24};    // OK: 24 can be represented exactly as a char (good)

    char c5 {264};   // error (assuming 8-bit chars): 264 cannot be 
                     // represented as a char (good)

    int x4 {2.0};    // error: no double to int value conversion (good)

}
```

## When to must use *intialization list* rather than *assignment*

* const var 

* Base class constructor