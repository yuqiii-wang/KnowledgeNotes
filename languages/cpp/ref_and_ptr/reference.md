# Reference

## Reference vs Pointer

Rule of thumb: Use references when you can, and pointers when you have to.

### Diffs

* A pointer can be re-assigned, while a reference is not (a pointer can have modifiable address, while reference is const).

* Init check, that a pointer can be assigned with a `nullptr` whereas reference is forbidden. References must be initialized before use, and this adds stability in code that allows compiler to easily detect uninitialized object.

* Scope management, that a pointer offers more layers of indirection, accessed/passed from/to external scopes

* resource by pointer might outlive local scope if not handled by `delete`, whereas reference-based resource will die when local scope ends.

## Universal References vs Rvalue References

`T&&` earns it name by working as either rvalue reference or lvalue reference, binding to const or non-const objects, and to volatile or non-volatile objects.

* If a function template parameter has type `T&&` for a deduced type `T`, or if an
object is declared using `auto&&`, the parameter or object is a universal reference.
```cpp
// param is a universal reference
template<typename MyTemplateType>
void someFunc(MyTemplateType&& param);
```
* If the form of the type declaration isn’t precisely `type&&`, or if type deduction
does not occur, `type&&` denotes an rvalue reference.
```cpp
// param is an rvalue reference
template<typename T>
void f(std::vector<T>&& param);
```
* Universal references correspond to rvalue references if they’re initialized with
rvalues. They correspond to lvalue references if they’re initialized with lvalues.
```cpp
// param is a universal reference
template<typename T>
void f(T&& param);
Widget w;
// lvalue passed to f; param's type is
// Widget& (i.e., an lvalue reference)
f(w);
// rvalue passed to f; param's type is
// Widget&& (i.e., an rvalue reference)
f(std::move(w));
```

## Reference Deduction and Collapsing

* When an lvalue is passed as an argument, T is deduced to be an lvalue reference. 
* When an rvalue is passed, T is deduced to be a
non-reference.
```cpp
template<typename T>
void func(T&& param);

Widget widgetFactory(); // function returning rvalue
Widget w; // a variable (an lvalue)

func(w); // call func with lvalue; T deduced
         // to be Widget&
func(widgetFactory()); // call func with rvalue; T deduced
                       // to be Widget
```

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

## `std::remove_reference`

If the type `T` is a reference type, provides the member `typedef` type which is the type referred to by `T`. Otherwise type is `T`.

```cpp
template< class T >
struct remove_reference;
```

For example, the code below prints out `true` since `int&&` is removed from being a reference to a normal `int`
```cpp
std::cout << std::is_same<int, std::remove_reference<int&&>::type>::value << std::endl;
```