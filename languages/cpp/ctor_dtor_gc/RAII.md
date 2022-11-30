# Resource Acquisition Is Initialization (RAII)

RAII is a practice philosophy: a resource must be acquired before use to the lifetime of an object, and must be freed after use.

Some typical:

* Constructors must throw exception when acquiring recourses fails

* Check and delete resources after the resources go out of scope (such as using `delete` to free resources and disconnecting a DB)

* Must apply lock for resources being modified by multiple threads (such as vars and files)

* Use smart pointers to manage resources that are used in different scopes

## Practice Example

The below code is bad for `std::mutex m` might never be freed.

```cpp
std::mutex m;
 
void bad() 
{
    m.lock();                    
    f();                         // if f() throws exception, mutex m will never be freed
    if(!everything_ok()) return; // return early, mutex m will never be freed
    m.unlock();                  
}
```

The problem can be solved by applying `lock_guard` that automatically checks ownership and releases resources when out of scope.

The class `lock_guard` is a mutex wrapper that provides a convenient RAII-style mechanism for owning a mutex for the duration of a scoped block.

```cpp
void good()
{
    std::lock_guard<std::mutex> lk(m); // RAII: apply resource management when the obj is init.
    f();                               
    if(!everything_ok()) return;       
}
```

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

## Container Cautions

When using containers such as `std::vector<T>`, if `T` has sub objects with allocated memory, must first free `T` before let `std::vector<T>` run out of scope. Smart pointer cannot detect if sub object memory is freed.