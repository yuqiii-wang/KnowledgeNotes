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

## When to must use *intialization list* rather than *assignment*

* const var 

* Base class constructor