# Some C++ Knowledge


### Throw exception vs return error code

### Floating point values

```bash
double a = 1/3; 
a=? 
```

Answer should be `0` since both `1` and `3` are `int`s

### Data Alignment

Compiler does not change the order in which vars are declared. Given a minimum number of bytes read cycle (such as 64 bits: 8 bytes), since a `char` takes one byte, an `int` four bytes, together they can be placed inside one read cycle.

For example, the below two `struct`s have diff sizes: 
```bash
sizeof(structc_t) = 24
sizeof(structd_t) = 16
```

```cpp
typedef struct structc_tag
{
   char        c;
   double      d;
   int         s;
} structc_t;
 
typedef struct structd_tag
{
   double      d;
   int         s;
   char        c;
} structd_t;
```

### placement new

As it allows to construct an object on memory that is already allocated, it is required for optimizations as it is faster not to re-allocate all the time. It is useful for object been re-constructed multiple times.

```cpp
int main() {
    // buffer on stack, init with 2 elems
    unsigned char buf[sizeof(int)*2] ;
  
    // placement new in buf
    int *pInt = new (buf) int(3);
    int *qInt = new (buf + sizeof (int)) int(5);

    // pBuf and pBuf are addrs of buf and buf+1 respectively, with init int values. 
    int *pBuf = (int*)(buf+0) ;
    int *pBuf = (int*) (buf + sizeof(int) * 1);

    return 0;
}
```

### Scopes

* Public — Any other object or module can access this class member.
* Protected — Only members of this class or subclasses of this class can access this member.
* Private — Only this class can access this member; subclass members cannot.

### `noexcept`

`noexcept` is used to forbid exception `throw`, that
```cpp
// the function f() does not throw
void f() noexcept; 
```

* `noexcept`, `noexcept(true)` do not allow throwing exception 
* `noexcept(expression)` in which when `expression` is evaluated to be true, do not throw exception, when false, allow throwing exception.

### `typedef`, `decltype`, `typename`, `#define`

* `#define` simply replaces var name during pre-processing

* `decltype` is used to set up var type, such as
```cpp
struct A { double x; };
const A* a;

// type of y is double (declared type)
decltype(a->x) y;
// type of z is const double& (lvalue expression)
decltype((a->x)) z = y;
```

* `typedef` is used to define a new type based on existing basic data type
```cpp
// simple typedef
typedef unsigned long ulong;
// the following two objects have the same type
unsigned long l1;
```

* `typename` has two use cases:

As a template argument keyword (instead of class)

A typename keyword tells the compiler that an identifier is a type (rather than a static member variable)

Without `typename`, `ptr` is considered a static member, such as
```cpp
template <class T>
Class MyClass
{
  typename T::SubType * ptr;
  ...
};
```

### `volatile`

`volatile` prevents compiler optimization that treats a variable as const, but dictates fetching variable value every time from its memory. Depending on multi-threading conditions, compiler may apply lock on the variable address each time when it fetches a volatile variable.

### `explicit`

Compiler has implicit conversion for arguments being passed to a function, while
the `explicit`-keyword can be used to enforce a function to be called *explicitly*.

```cpp
//  a class with one int arg constructor
class Foo {
private:
  int m_foo;
public:
  Foo (int foo) : m_foo (foo) {}
  int GetFoo () { return m_foo; }
};

// a function implicitly takes a Foo obj
void DoBar (Foo foo) {
  int i = foo.GetFoo ();
}

// Implicit call is available
int main () {
  DoBar (42);
}
```

However, if `Foo`'s constructor is defined `explicit`, the above invocation is forbidden, only `DoBar (Foo (42))` is allowed.

### `static`

`static` is basically a global variable.

It is init **before** entering `main()`.

### `struct` vs `union`

A Structure `struct` does not have a shared location for all of its members. It makes the size of a Structure to be greater than or equal to the sum of the size of its data members (disregarding memory alignment).

A Union `union` does not have a separate location for every member in it. It makes its size equal to the size of the largest member among all the data members.

## C++ 11

* Lambda

* `auto` and `decltype`

`decltype` gives the declared type of the expression that is passed to it. 
`auto` does the same thing as template type deduction.

for example, given a function returns a reference, `auto` will still be a value (need `auto&` to get a reference), but `decltype` will be exactly the type of the return value.

```cpp
int a = 1;
int& foo() {
  return a;
}

decltype(foo()) a = foo(); //a is an `int&`
auto b = foo(); //b is an `int`
```

* Deleted and Defaulted Functions

When a function is declared `default`, compiler generates the function; When a function is declared `delete`, compiler does not generate the function.

For example, automatic generated constructors, destructors and assignment operators are forbidden in use when declared `delete`.

* `nullptr`

`nullptr` is useful avoiding ambiguous use of `NULL` and `0`.

* rvalue reference

use of `std::move`.

* `promise` and `future`, `async()`

* smart pointer: `shared_ptr` and `unique_ptr`

* Regular expression library

## C++17

* `std::optional`

`std::optional` manages an optional contained value, i.e. a value that may or may not be present.

For example, the below code replaces `{}` with `"empty"` for the `std::optional` set to false.
```cpp
// optional can be used as the return type of a factory that may fail
std::optional<std::string> create(bool b) {
    if (b) return "Godzilla";
    return {};
}
 
int main() {
    std::cout << "create(false) returned "
              << create(false).value_or("empty") << '\n';
    return 0;
}
```

## c++20

* module

```cpp
export module myModule;

namespace myModule
{
  export int add (int a, int b) 
    return a + b;

  export class Foo {
    public:
      int add(int a, int b) 
        return myModule::add(a, b);
  };
}
```

* Co-routine

* Concepts

`concept` is used to contain a template.
For example, below code dictates the input argument must be a signed integer.
 
```cpp
concept SignedIntegralT = std::is_integral_v<T> && std::is_signed_v<T>;

template <SignedIntegralT T>
T get(T& t) {return t;}
```