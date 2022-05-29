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