# Some Practice Tips

* `std::endl` must be appended to `std::cout`

The reason for this is that typically endl flushes the contents of the stream.

You must implement at least one `std::endl` before exit of a program.

* `std::queue`, `std::deque` and `std::stack`

`deque`: Double ended queue, insert and remove from both ends

`queue`: insert only in one end and remove from the other (first in first out)

`stack`: LIFO context (last-in first-out)

* `i++` vs `++i`

`++i` increments the number before the current expression is evaluated, whereas `i++` increments the number after the expression is evaluated.

There is no diff for them being placed in a loop condition (However, `++i` should be preferred as old version compiler might generate more machine code for `i++` than that of `++i`)
```cpp
for (int i = 0; i < 10; i++){;}
```
or
```cpp
for (int i = 0; i < 10; ++i){;}
```

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

The constexpr specifier declares that it is possible to evaluate the value of the function or variable at compile time. It is used for compiler optimization and input params as `const`

For example:
```cpp
constexpr int multiply (int x, int y) return x * y;
extern const int val = multiply(10,10);
```
would be compiled into
```as
push    rbp
mov     rbp, rsp
mov     esi, 100 //substituted in as an immediate
```
since `multiply` is constexpr.

While, 
```cpp
const int multiply (int x, int y) return x * y;
const int val = multiply(10,10);
```
would be compile into 
```as
multiply(int, int):
        push    rbp
        mov     rbp, rsp
        mov     DWORD PTR [rbp-4], edi
        mov     DWORD PTR [rbp-8], esi
        mov     eax, DWORD PTR [rbp-4]
        imul    eax, DWORD PTR [rbp-8]
        pop     rbp
        ret
...
__static_initialization_and_destruction_0(int, int):
...
        call    multiply(int, int)
```

* string types

`std::string` is allocates memory in **a single block** (as needed, and sometimes preemptively), and best practice suggests pre-computing the string size the filling it.

`std::stringstream`, `std::istringstream` and `std::ostringstream` **1)** better express the intent to appending strings by `<<` and `>>` respectively. **2)** A stringstream writes into a stringbuffer, which usually means **a linked list of buffers**. Memory is not continuous, and it requires no reallocations as the buffer grows.


They are interchangeable via the following
```cpp
const std::string tmp = stringstream.str();
const char* cstr = tmp.c_str();
```

* `std::atomic`

`std::atomic` works with trivial copyables (such as C-compatible POD types, e.g., int, bool, char) to guarantee thread safety (defined behavior when one thread read and another thread write) by trying squeezing one operation into one cpu cycle (one instruction). 

Only some POD types are by default atomic (placeable inside one register), such as `char` and `int16_t` (both 2 bytes), dependent on register config, other POD types might not be atomic.

It is **NOT** allowed to apply atomic to an array such as
```cpp
std::atomic<std::array<int,10>> myArray;
```
in which `myArray`'s element is not readable/writable.

* constructors

default
```cpp
// class construct
construct(){}
// main()
construct c;
```

parameterized
```cpp
// class construct
construct(int a){}
// main()
int a = 1;
construct c(a);
```

copy construct (only shallow copy)
using `=` copy assignment without defined copy constructor in class is undefined behavior.
```cpp
// class construct
construct(){}
construct(const construct&){} // copy construct
// main()
construct c1;
construct c2 = c1; // copy construct
construct c3(c1); // also a copy construct
```

* `new` vs `malloc`

`new` allocates memory and calls constructor for object initialization. But `malloc()` allocates memory and does not call constructor.

Return type of `new` is exact data type while `malloc()` returns `void*`.

Use `delete` to deallocates a block of memory. Use `delete` for non-`new` allocated memory rendering undefined behavior.

* exceptions

`try`/`catch` cannot catch all exceptions, some typical are

1) divided by zero

2) segmentation fault/access out of range

* POD

A POD type is a type that is compatible with C 

* Some code tricks
Show the print results:
```cpp
using namespace std;
int  a=4;
int  &f(int  x)
{
    a = a + x;
    return  a;
}

int main()
{
    int t = 5;
    cout<<f(t)<<endl;  //a = 9
    f(t) = 20;           //a = 20
    cout<<f(t)<<endl;  //t = 5,a = 25
    t = f(t);            //a = 30 t = 30
    cout<<f(t)<<endl;  //t = 60
    return 0;
}
```

* `class` vs `struct`

Diffs: 
1) when inheritance, struct's members are default public, while class'es are private.
2) when accessed as object, struct object members are default public, while class'es are private.

* Tricks: show the result

`||` returns when met the first true statement, so the `++y` is not run. `true` is implicitly converted to `int` 1.

```bash
t = 1;
x = 3;
y = 2;
```

```cpp
int x = 2, y = 2, t = 0;
t = x++ || ++y;
```

* `const` of diff forms

`const` applies to the thing left of it. If there is nothing on the left then it applies to the thing right of it.

Always remember things preceding on the left hand side of `*` are the pointer pointed type, right hand side only allows `const` to say if the pointer is a const (do not allow re-pointing to a new object)

`int const*` is equivalent to `const int*`, pointer to const int.

`int *const` is a constant pointer to integer

`const int* const` is a constant pointer to constant integer

* `final` vs `override`

* `NULL` vs `nullptr`

Better use `std::nullptr_t` implicitly converts to all raw pointer types and prevents ambiguity of integral type.

```cpp
// three overloads of f
void f(int);
void f(bool);
void f(void*);

f(0); // calls f(int), not f(void*)
f(NULL); // might not compile, but typically calls
         // f(int). Never calls f(void*)
```

* Use scoped `enum`

```cpp
enum class Color { black, white, red };
// are scoped to Color

auto white = false;
// fine, no other

Color c = Color::white; 
// fine
```

* `delete` use to default functions

There are compiler generated functions such as copy constructor but you do not want user to invoke, you can hence:
```cpp
basic_ios(const basic_ios& ) = delete;
basic_ios& operator=(const basic_ios&) = delete;
```


* order of execution

```cpp
int (*((*ptr(int, int)))) (int); 
```

Explain:
```cpp
// function return to a pointer
*ptr(int, int)

// take the return pointer as an arg
(*ptr(int, int))

// extra parentheses does not make any difference
((*ptr(int, int)))

// function pointer to pointer
*((*ptr(int, int)))

// function pointer to int pointer
int (*((*ptr(int, int))))
```