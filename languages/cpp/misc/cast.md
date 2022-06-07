# Four types of cast

## `std::move` and `std::forward`

`std::move` and `std::forward` are merely functions (actually function templates)
that perform casts. `std::move` unconditionally casts its argument to an rvalue, while
std::forward performs this cast only if a particular condition is fulfilled.

### `std::move`

Benefits:
* Moving containers is now as cheap as copying pointers

```cpp
template<typename T>
typename remove_reference<T>::type&&
move(T&& param)
{
    using ReturnType = typename remove_reference<T>::type&&;
    return static_cast<ReturnType>(param);
}
```

The `&&` part of the function’s return type implies that std::move returns an rvalue
reference.

### `std::forward`

`std::forward` performs casting on certain conditions.
The most common scenario is a function template taking a universal reference parameter.

```cpp
void process(const Widget& lvalArg);
void process(Widget&& rvalArg);

template<typename T>
void logAndProcess(T&& param)
{
    auto now = std::chrono::system_clock::now(); 
    makeLogEntry("Calling 'process'", now);
    process(std::forward<T>(param));
}
```

## Four Types of Cast

### `static_cast<new-type>(expression)` 

`static_cast<new-type>(expression)` returns a value of type `new-type`. 
It performs conversions between compatible types, and throws error during compilation time if the conversion types mismatch. This is an advantage over C-style conversion that throws error ay run time for conversion types mismatch.
```cpp
char c = 10;       // 1 byte
int *p = (int*)&c; // 4 bytes

*p = 5; // run-time error: stack corruption

int *q = static_cast<int*>(&c); // compile-time error
```

Remember, it is unsafe if it is used to downcast an object
```cpp
class B {};
class D : public B {};

B* pb;
D* pd = static_cast<D*>(pb); // unsafe, downcast to its base
```

### `dynamic_cast<new-type>(expression)` 

`dynamic_cast<new-type>(expression)` returns a value of type `new-type`. The type of expression must be a pointer if `new-type` is a pointer, or an l-value if `new-type` is a reference.

`dynamic_cast` is useful when object type is unknown. It returns a null pointer if the object referred to doesn't contain the type casted to as a base class (when you cast to a reference, a bad_cast exception is thrown in that case).

```cpp
if (A1 *a1 = dynamic_cast<A1*>(&obj)) {
  ...
} else if (A2 *a2 = dynamic_cast<A2*>(&obj)) {
  ...
} else if (A3 *a3 = dynamic_cast<A3*>(&obj)) {
  ...
}
```

### `reinterpret_cast<new-type>(expression)`

`reinterpret_cast<new-type>(expression)` performs conversions between types by reinterpreting the underlying bit pattern.

`reinterpret_cast` can force pointer type change, and makes type unsafe.

For example,
```cpp
union U { int a; double b; } u = {0};
int arr[2];

// value of p3 is "pointer to u.a":
// u.a and u are pointer-interconvertible
int* p = reinterpret_cast<int*>(&u);

// value of p2 is "pointer to u.b": u.a and
// u.b are pointer-interconvertible because
// both are pointer-interconvertible with u
double* p2 = reinterpret_cast<double*>(p); 

// value of p3 is unchanged by reinterpret_cast
// and is "pointer to arr"
int* p3 = reinterpret_cast<int*>(&arr); 
```

### `const_cast<new-type>(expression)`

`const_cast<new-type>(expression)` may be used to cast away (remove) constness or volatility, such as
```cpp
int i = 3;                 // i is not declared const
const int& rci = i; 
const_cast<int&>(rci) = 4; // OK: modifies i
```