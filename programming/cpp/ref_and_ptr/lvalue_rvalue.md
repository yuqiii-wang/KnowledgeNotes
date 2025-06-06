# Rvalue vs Lvalue

Put simply, an *lvalue* is an object reference and an *rvalue* is a value. An lvalue refers to an object that persists beyond a single expression.
An rvalue is a temporary value that does not persist beyond the expression that uses it.

```cpp
int get(){
    return 0;
}
int main(){
    int val = get(); // get() returns rvalue
    return 0;
}
```

Reference takes lvalue, so that

```cpp
void setVal(int& val){}
int main(){
    // this line of code below will fail compiler for that 10 is an immediate val (a temp rvalue)
    setVal(10);

    // however, if we assign an addr, compiler will pass it
    int val = 10;
    setVal(val);

    return 0;
}

```

for `lvalue` and `rvalue` as arguments, it is recommended to add overloads, rather than by `const&`

```cpp
void foo(X& x) {}
void foo(X&& x) { foo(x); }

// although the following func takes both, it is not of good practice
void foo(const X& x){}
```

## Examples

### lvalue reference must either be a const or take an addr

```cpp
int a = 5;
int &ref_a = a; 
int &ref_a = 5; // Compilation error, lvalue reference must be const

const int &ref_a = 5;  // good to pass compilation, 
                       // but no semantic meaning, 
                       // since ref_a refers to invalid addr
```

### `std::vector::push_back` takes lvalue

```cpp
std::vector<int> vec;

vec.push_back(1); // compilation error

int a = 1;
vec.push_back(a); // works, a is lvalue, but not elegant
```

### `std::vector::emplace_back` takes `T&&`

```cpp
std::vector<std::string> vec;

std::string str1{"asdfg"};
std::string str2{"asdfg"};

vec.push_back(str1); // copy
vec.push_back(std::move(str1)); // by move, str1 is now empty
vec.emplace_back(str2); // same as by move, str2 is now empty
vec.emplace_back("axcsddcas"); // valid, emplace_back can take rvalue
```

### Operation Overloading Example

Consider this,

```cpp
Object A, B, C, D, E;
A = (B + (C + (D + E)));
```

However, `operator+` such as on `(D + E)` whose result is `+` to `C`, creates multiple temp objects and soon useless thus deleted, given the following Object definition.

```cpp
Object Object::operator+ (const Object& rhs) {
    Object temp (*this);
    // logic for adding
    return temp;
}
```

This can be addressed by

```cpp
Object& Object::operator+ (Object&& rhs) {
    // logic to modify rhs directly
    return rhs;
}
```

This is known as `move` semantics, in which `const Object& rhs` is a lvalve reference and `Object&& rhs` changes to a modifiable rvalue reference.

### & vs && in var declaration

`auto& x` is simply a var reference.

`auto&& x` treats x as a temp rvalue without making it as `const&`.

### Example

```cpp
class Widget {
public:


void doWork() &; // this version of doWork applies
                 // only when *this is an lvalue
void doWork() &&; // this version of doWork applies
                  // only when *this is an rvalue
};

Widget makeWidget(); // factory function (returns rvalue)
Widget w; // normal object (an lvalue)

w.doWork(); // calls Widget::doWork for lvalues
// (i.e., Widget::doWork &)
makeWidget().doWork(); // calls Widget::doWork for rvalues
// (i.e., Widget::doWork &&)
```

## Move and Forward

### Summary

Both `move` and `forward` are introduced to deal with expensive "copy" operation when passing params; `move` is used to force using *rvalue reference* without copy, while `forward` is of added compatibility handling *rvalue/lvalue reference* as well as `const reference`.

### Further explained

Consider this statement:

```cpp
std::string src_str = "hello";
std::string dest_str = src_str;
```

inside, `=` is actually a copy assignment operator. If `src_str` is no longer used, it can actually be assigned to the addr of `src_str` to `dest_str`, such as below

```cpp
std::string dest_str = std::move(src_str);
// The above statement is same as
std::string dest_str((std::string&&)src_str);
```

so that `src_str` becomes a temporary (an rvalue).

In contrast to `std::move` that treats an object as a temp rvalue, `std::forward` has a single use case: to cast a templated function parameter (inside the function) to the value category (lvalue or rvalue) the caller used to pass it.
This allows rvalue arguments to be passed on as `rvalue`s, and `lvalue`s to be passed on as `lvalue`s, a scheme called "perfect forwarding".

In the example below, `wrap(T&& t)` has a param not deduced until invocation.

```cpp
struct S{};

void foo(S& s){};
void foo(S&& s){};

template<typename T>
void wrap(T&& t){
    foo(std::forward<T>(t));
}

int main(){
    S s;
    wrap(s); // lvalue
    wrap(S()); // rvalue
    return 0;
}
```

### Use Cases

In C++, all non-pointer lvalue objects are RAII managed that when out of scope, they are demised.
In nature, `std::move` just forces changing a lvalue object to rvalue so that such object can be movable/extended lifecycle to greater/another scope.

The implication of using `std::move` is that user is happy to transfer the ownership of an object to another scope.

```cpp
A a;
fun(a); // When a is small, usually primitive data types
fun(&a); // similar to passing by pointer, both the current and the `fun` scopes share the ownership of `a`
fun(std::move(a)); // the current scope no longer wants a, but transferred to `fun`
```

#### The rvalue reference

By default, `a2 = a1;` implements copy constructor `A& operator=(const A&)`.

Rvalue reference by `std::move` is a temp reference object that can pass reference data type validation, and it has implications user do not want to use the object in the old scope.

Reference is semantically useful such as in `void foo(A&& other)` where the reference can be passed to be de-referenced as an address to a pointer.

```cpp
#include <iostream>

class A {
public:
    A& operator=(const A&) {
        std::cout << "copy" << std::endl;
        return *this;
    }
    A& operator=(A&&) {
        std::cout << "move" << std::endl;
        return *this;
    }
};

void foo(A&& other) {
    A* p = &other;
}

int main() {
    A a1, a2;
    a2 = a1;                // print "copy"
    a2 = std::move(a1);     // print "move"
    foo(std::move(a2));
}
```

#### `std::unique_ptr` construction

`std::unique_ptr` construction is efficient as it implements move constructor.

### Example: Forward Test

```cpp
#include <iostream>

using namespace std;

int main()
{
    auto print = [](auto& x) {
        cout << x << endl;
    }; 
    
    auto forwarder1 = [&](const auto& e) {
      print(std::forward<decltype(e)>(e));
    };
    auto forwarder2 = [&]( auto&& e) {
      print(std::forward<decltype(e)>(e));
    };
    
    int e = 1;
    
    forwarder1(std::move(e)); // working because legal to bind const lvalue reference to rvalue

    // forwarder2(std::move(e)); 
    // above NOT working because deduced type is rvalue of type int 
    // in print(), cannot bind non-const lvalue reference of type ‘int&’ to an rvalue of type ‘int’

    return 0;
}
```

### Example: `std::string`

In the code below, `str2(std::move(str));` makes the original `str` invalidated to become a null string, while `std::move(str)` without other move constructor invocation does not make the original `str` null.

```cpp
std::string str("hello");
std::cout << "before: " << str << std::endl;
std::string str2(std::move(str));
std::cout << "after: " << str << std::endl; // null str

std::string str("hello");
std::cout << "before: " << str << std::endl;
std::cout << "middle: " << std::move(str) << std::endl; // "hello"
std::cout << "after: " << str << std::endl; // "hello"
```

Explained: `std::move` is just a right value casting and nothing more.
The casting operation itself does not change anything.
However, in `str2(std::move(str));`, the default `std::string` move constructor replaced the data pointer from `str`'s to `str2`'s, hence the previous `str`'s data pointer is a `nullptr`.

### `auto&&` vs `const auto&`

```cpp
auto forwarder1 = [&](const auto& e) {
     std::cout << std::forward<decltype(e)>(e) << std::endl;
};
auto forwarder2 = [&]( auto&& e) {
     std::cout << std::forward<decltype(e)>(e) << std::endl;
};

int e = 1;
    
forwarder1(std::move(e)); // working because legal to bind const lvalue reference to rvalue
```
