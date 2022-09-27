# Const

* `consteval` (supported since c++20)

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

* `constinit`