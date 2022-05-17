# Pointers

## Pointer Knowledge

### Array vs pointer

Pointer is more versatile that it can point to many things, while array can be only created either on stack or globally.

### Pointer to const

`int const *` means that the int is constant, while `int * const` would mean that the pointer is constant.

## Four types of cast

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