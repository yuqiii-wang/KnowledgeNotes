# Cast and conversion

## Implicit type conversions

When no explicit casts are present, the compiler uses built-in standard conversions to convert one of the operands so that the types match. 

The type match search follows the below order:

1. Widening conversions (promotion)

A value in a smaller variable is assigned to a larger variable with no loss of data. So that it is safe.

|From|To|
|-|-|
|`__int64`, `long long `|`double`|
|`bool`, `char`|Any other built-in type|
|`short`|`int`|

2. Narrowing conversions (coercion)

The compiler performs narrowing conversions implicitly, but it warns you about potential data loss. 

```cpp
int i = INT_MAX + 1; //warning C4307:'+':integral constant overflow
wchar_t wch = 'A'; //OK
char c = wch; // warning C4244:'initializing':conversion from 'wchar_t'
              // to 'char', possible loss of data
unsigned char c2 = 0xfffe; //warning C4305:'initializing':truncation from
                           // 'int' to 'unsigned char'
int j = 1.9f; // warning C4244:'initializing':conversion from 'float' to
              // 'int', possible loss of data
int k = 7.7; // warning C4244:'initializing':conversion from 'double' to
             // 'int', possible loss of data
```

3. Pointer conversion

First, the "Help" string constant literal is converted to a char* that points to the first element of the array; that pointer is then incremented by three elements so that it now points to the last element 'p'.

```cpp
char* s = "Help" + 3;
```

## Downcast vs Upcast

* Upcast

*Upcast* is the process to create the derived class's pointer or reference from the base class's pointer or reference.
```cpp
Base *ptr = &derived_obj;
```

* Downcast

The *Downcast* is an opposite process to upcast, which converts the base class's pointer or reference to the derived class's pointer or reference. 
Downcast is dangerous since the derived class may have members not defined in base class.
```cpp
Derived *ptr = &base_obj;
```

## Four types of cast 

### Summary

* For strict "value casting", use `static_cast<new-type>`. 
* For run-time polymorphic casting of pointers, use `dynamic_cast<new-type>`. 
* Direct bit representation, use `reintrepret_cast<new-type>`. 
* To just throw const out the window there is `const_cast<new-type>`.

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


`static_cast` happens at compile time (Type info at compile time). As a result, no safety check is performed during runtime to check if the object being converted is in fact a full object of the destination type. 

### `dynamic_cast<new-type>(expression)` 

`dynamic_cast<new-type>(expression)` returns a value of type `new-type`. The type of expression must be a pointer if `new-type` is a pointer, or an l-value if `new-type` is a reference.

`dynamic_cast` only works on polymorphic types, i.e., types that declare at least one virtual function, hence, used only with pointers and references to objects.

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


Another example: `Animal` as the base class and `Dog` and `Cat` are the derived
```cpp
class Animal { /* Some virtual members */ };
class Dog: public Animal {};
class Cat: public Animal {};

Dog     dog;
Cat     cat;
Animal& AnimalRefDog = dog;  // Notice no cast required. (Dogs and cats are animals).
Animal& AnimalRefCat = cat;
Animal* AnimalPtrDog = &dog;
Animal* AnimalPtrCat = &cat;

Cat&    catRef1 = dynamic_cast<Cat&>(AnimalRefDog);  // Throws an exception  AnimalRefDog is a dog
Cat*    catPtr1 = dynamic_cast<Cat*>(AnimalPtrDog);  // Returns NULL         AnimalPtrDog is a dog
Cat&    catRef2 = dynamic_cast<Cat&>(AnimalRefCat);  // Works
Cat*    catPtr2 = dynamic_cast<Cat*>(AnimalPtrCat);  // Works

// This on the other hand makes no sense
// An animal object is not a cat. Therefore it can not be treated like a Cat.
Animal  a;
Cat&    catRef1 = dynamic_cast<Cat&>(a);    // Throws an exception  Its not a CAT
Cat*    catPtr1 = dynamic_cast<Cat*>(&a);   // Returns NULL         Its not a CAT.
```

`dynamic_cast` raises exception when casting a base object to a derived. `static_cast` would not see this error.

```cpp
class CBase { };
class CDerived: public CBase { };

CBase b; CBase* pb;
CDerived d; CDerived* pd;

pb = dynamic_cast<CBase*>(&d);     // ok: derived-to-base
pd = dynamic_cast<CDerived*>(&b);  // wrong: base-to-derived
```

`dynamic_cast` happens at run time.

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

* `reinterpret_cast` vs `static_cast`

In `static_cast`, for example, to convert between `int(12)` to `unsigned float (12.0f)`, needs to compute the underlying bit representation of the two type patterns, and make bit changes.

In `reinterpret_cast`, CPU does not invoke any calculations. 
It just treats a set of bits in the memory like if it had another type. 

* `reinterpret_cast` Use Cases

It can be used to source raw bit data then reinterpret it to a new data type directly by the underlying bit format.

```cpp
char c = 'a';
int val = *reinterpret_cast<int*>(&c); // val = 97 or 0110 0001
```

### `const_cast<new-type>(expression)`

`const_cast<new-type>(expression)` may be used to cast away (remove) constness or volatility, such as
```cpp
int i = 3;                 // i is not declared const
const int& rci = i; 
const_cast<int&>(rci) = 4; // OK: modifies i
```

Remember, `const` variables are simply labelled as "read-only" by compiler, not something hard coded in assembly.
Instead, `constexpr` makes code to assembly hence not modifiable.