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

## Variadic Template