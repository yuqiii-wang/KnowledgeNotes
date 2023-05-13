# Pointers

## Array vs pointer

Pointer is more versatile that it can point to many things, while array can be only created either on stack or globally.

## Pointer to const

`int const *` means that the int is constant, while `int * const` would mean that the pointer is constant.

## Opaque pointer

an opaque pointer is a special case of an opaque data type, a data type declared to be a pointer to a record or data structure of some unspecified type.

Opaque pointers are a way to hide the implementation details of an interface from ordinary clients, so that the implementation may be changed without the need to recompile the modules using it. 

This technique is described in Design Patterns as the *Bridge pattern*. It is sometimes referred to as "handle classes", the "Pimpl idiom" (for "pointer to implementation idiom"), "Compiler firewall idiom", "d-pointer" or "Cheshire Cat".

Code below shows `private: std::unique_ptr<CheshireCat> d_ptr_;` as a hidden pointer whose actual implementation is unknown.
```cpp
/* PublicClass.h */

#include <memory>

class PublicClass {
 public:
  PublicClass();                               // Constructor
  PublicClass(const PublicClass&);             // Copy constructor
  PublicClass(PublicClass&&);                  // Move constructor
  PublicClass& operator=(const PublicClass&);  // Copy assignment operator
  PublicClass& operator=(PublicClass&&);       // Move assignment operator
  ~PublicClass();                              // Destructor

  // Other operations...

 private:
  struct CheshireCat;                   // Not defined here
  std::unique_ptr<CheshireCat> d_ptr_;  // Opaque pointer
};
/* PublicClass.cpp */

#include "PublicClass.h"

struct PublicClass::CheshireCat {
  int a;
  int b;
};

PublicClass::PublicClass()
    : d_ptr_(std::make_unique<CheshireCat>()) {
  // Do nothing.
}

PublicClass::PublicClass(const PublicClass& other)
    : d_ptr_(std::make_unique<CheshireCat>(*other.d_ptr_)) {
  // Do nothing.
}

PublicClass::PublicClass(PublicClass&& other) = default;

PublicClass& PublicClass::operator=(const PublicClass &other) {
  *d_ptr_ = *other.d_ptr_;
  return *this;
}

PublicClass& PublicClass::operator=(PublicClass&&) = default;

PublicClass::~PublicClass() = default;
```

## `delete` vs `free`

|`delete`|`free`|
|-|-|
|It de-allocates the memory dynamically.	|It destroys the memory at the runtime.|
|It should only be used either for the pointers pointing to the memory allocated using the new operator or for a `NULL` pointer.	|It should only be used either for the pointers pointing to the memory allocated using `malloc()` or for a `NULL` pointer.|
|This operator calls the destructor after it destroys the allocated memory. 	|This function only frees the memory from the heap. It does not call the destructor.|

## Pointer Optimization Difficulty

Compared to reference, pointer is difficult for compiler to perform optimization,
for compiler does not know what object/memory a pointer actually points to, that compiler has to forfeit many optimization tricks.

For example, given two pointers `int *p` and `int *q` to perform their sum after increment operations,
```cpp
int f(int *p, int *q) { 
  *p += 1; 
  *q += 1; 
  return *p + *q; 
} 
```
compiler can allocate two registers `r1` and `r2` then just add them up
```x86asm
INCR r1
INCR r2
ADD r1, r2
```

However, if `p` and `q` are referring to the same memory address, the above assembly is wrong.
Instead, should load first then perform arithmetic operations.

```x86asm
LDR r1, p    ; load p pointed int to r1
INCR r1
LDR r2, q    ; load q pointed int to r2
INCR r2
ADD r1, r2
```

A solution is to add `restrict` that indicates within the scope restrict pointers cannot point to the same address.
```cpp
int f(int *restrict p, int *restrict q);
```

## Raw Pointer Management and RAII

Raw pointer is often hard to manage for its lifecycle and content access management can be complex, especially in big project where there might be many threads accessing the pointer simultaneously.

### Base Class and Derived Class

In common business logic, once base class is demised, derived class's allocated pointers should be freed as well.
In base class destructor 

### Pointer Vector

### Mutex for Raw Pointer