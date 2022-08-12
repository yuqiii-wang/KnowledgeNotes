# Some C++ Knowledge

### STL Container Thread-Safe Access and Modification

STL containers are made thread safety to each element modification, such as `std::vector<Element>`, that
* simultaneous reads of the same object/element are OK
* simultaneous read/writes of different objects/elements of a container are OK

`std::vector<bool>` has each element occupying one bit of space, not necessary in contiguous space. 

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

### `const`

`const` to var is subject to type checking and is non-modifiable.

`const` to class member function makes this function *read-only* to its object, which means forbidden invocation to other non-`const` member function, forbidden modification to class members.

`const` to an expression means that the expression can be evaluated at compiling time.

`void function (T const&);`: a reference is a const pointer. `int * const a = &b;` is the same as `int& a = b;`.

## C++ 11

* Lambda

* `auto` and `decltype`

* Deleted and Defaulted Functions

When a function is declared `default`, compiler generates the function; When a function is declared `delete`, compiler does not generate the function.

For example, automatic generated constructors, destructors and assignment operators are forbidden in use when declared `delete`.

* `nullptr`

`nullptr` is useful avoiding ambiguous use of `NULL` and `0`.

* rvalue reference

use of `std::move`.

* `promise` and `future`, `async()`

* smart pointer: `shared_ptr` and `unique_ptr`
