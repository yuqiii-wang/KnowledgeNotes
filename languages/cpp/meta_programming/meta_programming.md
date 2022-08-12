# Meta Programming

*Template metaprogramming* (TMP) is a metaprogramming technique in which templates are used by a compiler to generate temporary source code, which is merged by the compiler with the rest of the source code and then compiled. 

* Meta Programming vs Macro

A macro is a piece of code that executes at compile time and either performs textual manipulation of code to-be compiled (e.g. C++ macros) or manipulates the abstract syntax tree being produced by the compiler (e.g. Rust or Lisp macros).

Template describes the generic form of the generated source code, and the instantiation causes a specific set of source code to be generated from the generic form in the template.

In conclusion, macro is about pre-processing, template is about compilation-time.

## Template Deduction

Give some c++ code below,
```cpp
template<typename T>
void f(ParamType param);

// call f with some expression
f(expr);
```
compilers use expr to deduce two types: one for `T` and one for `ParamType`.

* When `T` is a pointer or a reference, the deduction is normal (var pass by pointer/reference)
* When it is neither a pointer nor a reference, it is pass by value, even if a var is declared a reference
* When it is an array, it is pass by pointer (array declaration is treated as a pointer declaration).

```cpp
template<typename T>
void f(T param);
// param is now passed by value

const int& rx = x;
f(rx); // T's and param's types are still both int

const char name[] = "J. P. Briggs";
f(name); // name is array, but T deduced as const char*
```

* `decltype`

```cpp
int x = 0;
```
`decltype((x))` is `int&` while `decltype(x)` is `int` (Being a
name, `x` is an lvalue, and C++ defines the expression `(x)` to be an lvalue)

This has significance when using `decltype(auto)`
```cpp
decltype(auto) f1()
{
int x = 0;
return x;
// decltype(x) is int, so f1 returns int
}
decltype(auto) f2()
{
int x = 0;
return (x);
// decltype((x)) is int&, so f2 returns int&
}
```

For debugging, you can use (it yields `std::type_info` object, and `std::type_info` has a member function, `name`)
```cpp
std::cout << typeid(x).name() << '\n';
```
However, it might not be accurate since `typeid(x)` passes var by value.

## `auto` with lambda

`auto` takes code's initialized value as the type, might be prone to error if data type is changeable.

`std::function` is a template pointer and might allocate memory in heap for code in closure.

```cpp
std::function<bool(const std::unique_ptr<T>&,
                const std::unique_ptr<T>&)> func;
```

## Use of `union`

`union` declaration of a variable with alternative data type occupying the same memory location.

For example given below, only 4 bytes are used for this union. Same bits fo data on the four byte memory, interpreted as either `int` or `float` depending on compiler and runtime execution.
```cpp
union{int a; float b;};
```