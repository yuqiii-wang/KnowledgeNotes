# Resource Acquisition Is Initialization (RAII)

Core concept: a resource must be acquired before use to the lifetime of an object, and must be freed after use.

Some typical:

* Constructors must throw exception when acquiring recourses fails

* Check and delete resources after the resources go out of scope (such as using `delete` to free resources and disconnecting a DB)

* Must apply lock for resources being modified by multiple threads (such as vars and files)

* Use smart pointers to manage resources that are used in different scopes

## A Typical Obj Lifecycle

```cpp
#include <memory>


template <typename T>
void life_of_an_object
{
    std::allocator<T> alloc;

    // 1. allocate/malloc 
    T * p = alloc.allocate(1);

    // 2. placement new run constructor
    new (p) T(); 

    // 3. to destroy the obk, run destructor
    p->~T();

    // 4. deallocate/free
    alloc.deallocate(p, 1);
}
```

## Temp object destruction

Compiler auto invokes destructor of a temp obj once its execution finishes.

The content of `p1` is undefined behavior, that `substr(1)` returns a temporary object which is soon destroyed automatically once this line of expression finishes running.
```cpp
string s1 = string("string1");
const char* p1 = s1.substr(1).data();
```

The correction would be this below.
```cpp
string s1 = string("string1");
string sTmp = s1.substr(1);
const char* p1 = sTmp.data();
```

## Examples

### Container Cautions

When using containers such as `std::vector<T>`, if `T` has sub objects with allocated memory, must first free `T` before let `std::vector<T>` run out of scope. Smart pointer cannot detect if sub object memory is freed.