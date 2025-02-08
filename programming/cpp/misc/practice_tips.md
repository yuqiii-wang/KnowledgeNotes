# Some Practice Tips

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

* g++ compiler compiles `i[a]` to `*(i+a)`

For example, for `a` and `b` of `int*`, given the code below, compiler compiles `i[a]` to `*(i+a)` and `a[i]` to `*(a+i)`. Since `*(a+i)` and `*(i+a)` points to the same addr. `i[a]` is equal to `a[i]`.
```cpp
for (int i = 1; i <= 5; ++i)
{
    cout << i[a] << endl;
    cout << i[b] << endl;
    cout << i[a][b] << endl;
}
```

* `extern`

Usually, global vars are declared in `.h` referenced by `#include`; 
the `extern` can be used ysed to reference global vars in another `.c` file.

`extern` vars in compilation does not immediately has definition.


* `extern "C"`

C++ uses `extern "C"` to prevent mangling.

For example, by a C compiler, the below code compilation fails for duplicate function name `printMe`, whereas a C++ compiler succeeds.
This is because a C++ compiler implicitly renames (*mangles*) functions based on their parameters to provide function overloading.
```cpp
#include <stdio.h>
    
// Two functions are defined with the same name
// but have different parameters

void printMe(int a) {
  printf("int: %i\n", a);
}

void printMe(char a) {
  printf("char: %c\n", a);
}
    
int main() {
  printMe('a');
  printMe(1);
  return 0;
}
```

However, there are scenarios when programmer does not want the c++ compiler mangle function names. 
For instance, in a large project, mangling may cause function to wrongly link function definition. 
As a result, it might be convenient to provide a keyword `extern "C"` which tells the C++ compiler not to mangle function names such as by `extern "C" void printMe(int a);`.