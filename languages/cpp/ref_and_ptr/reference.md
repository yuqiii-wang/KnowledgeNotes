# Reference

## Reference vs Pointer

Rule of thumb: Use references when you can, and pointers when you have to.

### Diffs

* A pointer can be re-assigned, while a reference is not (a pointer can have modifiable address, while reference is const).

* Init check, that a pointer can be assigned with a `nullptr` whereas reference is forbidden.

* Scope management, that a pointer offers more layers of indirection, accessed/passed from/to external scopes

## Reference Tools

* reference_wrapper

A `reference_wrapper<Ty>` is a copy constructible and copy assignable wrapper around a reference to an object or a function of type `Ty`, and holds a pointer that points to an object of that type.

`std::ref` and `std::cref` (for const reference) can help create a reference_wrapper. Please be aware of out of scope err when using `reference_wrapper<Ty>` since it is a wrapper/pointer to an variable.

Below is an example that shows by `std::ref(x)` there's no need of explicit declaring reference type.
```cpp
template<typename N>
void change(N n) {
 //if n is std::reference_wrapper<int>, 
 // it implicitly converts to int& here.
 n += 1; 
}

void foo() {
 int x = 10; 

 int& xref = x;
 change(xref); //Passed by value 
 //x is still 10
 std::cout << x << "\n"; //10

 //Be explicit to pass by reference
 change<int&>(x);
 //x is 11 now
 std::cout << x << "\n"; //11

 //Or use std::ref
 change(std::ref(x)); //Passed by reference
 //x is 12 now
 std::cout << x << "\n"; //12
}
```

* bind

The function template `bind` generates a forwarding call wrapper for f (returns a function object based on f). Calling this wrapper is equivalent to invoking f with some of its arguments bound to args. 

For example:
```cpp
#include <functional>

int foo(int x1, int x2) {
    std::max(x1, x2);
}

int main(){
    using namespace std::placeholders;

    auto f1 = std::bind(foo, _1, _2);
    int max_val = f1(1, 2);
    return 0;
}
```

* weak reference
