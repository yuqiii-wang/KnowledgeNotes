# Const

## `const`

`const` to var is subject to type checking and is non-modifiable.

* `const` vs `constexpr`: `const` is set to read-only by compiler, mutable by `const_cast` (such as below); while `constexpr` means that the expression can be evaluated at compiling time.

```cpp
struct type {
    int i;
 
    type(): i(3) {}
 
    void f(int v) const {
        // this->i = v;                 // compile error: this is a pointer to const
        const_cast<type*>(this)->i = v; // OK as long as the type object isn't const
    }
};
```

* `const` to a reference

`void function (T const&);`: a reference is a const pointer. `int * const a = &b;` is the same as `int& a = b;`.

* `const member function`

`const` to class member function makes this function *read-only* to its object, which means forbidden invocation to other non-`const` member function, forbidden modification to class members.

The below code throw an error for `error: passing 'const xy_stl::Count' as 'this' argument discards qualifiers [-fpermissive]`.

```cpp
int add(const Count& count){
        return c + count.getCount();
}
```

The error derives from `count.getCount()` not declared as `const`. The remediation is declaring a trailing `const`.
```cpp
int Count::getCount() const
```

## `consteval` (supported since c++20)

## `constexpr`

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

## `constinit`
