# Some C++ Advanced Knowledge

## `restrict`

`restrict` tells the compiler that a pointer is not *aliased*, that is, not referenced by any other pointers. This allows the compiler to perform additional optimizations. It is the opposite of `volatile`.

For example, 
```cpp
// ManyMemberStruct has many members
struct ManyMemberStruct {
    int a = 0;
    int b = 0;
    // ...
    int z = 0;
};

// manyMemberStructPtr is a restrict pointer
ManyMemberStruct* restrict manyMemberStructPtr = new ManyMemberStruct();

// Assume there are many operations on the pointer manyMemberStructPtr.
// Code below might be optimized by compiler.
// since the memory of manyMemberStructPtr is only pointed by manyMemberStructPtr,
// no other pointer points to the memory.
// Compiler might facilitate operations without 
// worrying about such as concurrency read/write by other pointers
manyMemberStructPtr.a = 1;
manyMemberStructPtr.b = 2;
// ...
manyMemberStructPtr.z = 26;

delete manyMemberStructPtr;
```

## Run-Time Type Information (RTTI)

*Run-time type information* or *run-time type identification* (RTTI) is a feature that exposes information about an object's data type at runtime. 

The `typeid` keyword is used to determine the class of an object at run time. 
It returns a reference to `std::type_info` object (`typeid` operator returns `std::type_info`), which exists until the end of the program.

Below code shows that inside `Person* ptr = &employee;`, the type of `ptr` is unknown until in run-time env, since it is the dereference of a pointer to a polymorphic class.

For `Person* ptr = &employee;`, the pointer type is a `Person`; the pointed obj is an `Employee`. For `Person& ref = employee;`, the reference result is a `Person`.

```cpp
#include <iostream>
#include <typeinfo>

class Person {
public:
    virtual ~Person() = default;
};

class Employee : public Person {};

int main() {
    Person person;
    Employee employee;
    Person* ptr = &employee;
    Person& ref = employee;
    
    // The string returned by typeid::name is implementation-defined.
    std::cout << typeid(person).name()
              << std::endl;  // Person (statically known at compile-time).
    std::cout << typeid(employee).name()
              << std::endl;  // Employee (statically known at compile-time).
    std::cout << typeid(ptr).name()
              << std::endl;  // Person* (statically known at compile-time).
    std::cout << typeid(*ptr).name()
              << std::endl;  // Employee (looked up dynamically at run-time
                             //           because it is the dereference of a
                             //           pointer to a polymorphic class).
    std::cout << typeid(ref).name()
              << std::endl;  // Employee (references can also be polymorphic)
    return 0;
}
```

### `std::type_info`

`std::type_info::name` returns an implementation-defined type name (*name mangling*).

Name mangling (also called name decoration) is a technique used to resolve unique name representation, providing a way of encoding additional information in the name of a function, structure, class or another datatype in order to pass more semantic information from the compiler to the linker.

So that, the type defined by `static_cast<new-type> obj` returns `new-type` at compile time, while `dynamic_cast<new-type> obj` checks what the `obj` actually is by name mangling representation. As a result,  inside `Person* ptr = &employee;`, `ptr` is `Person*` while `ptr*` refers to `Employee`.