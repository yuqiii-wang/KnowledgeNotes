# Virtual Method Performance

Why does employing virtual method cause performance issues:
1. Virtual table lookup cost
2. Cache miss
3. Compiler cannot optimize virtual functions (such as no `inline`)

## C++ RTTI (Run-Time Type Information) 

For C++ polymorphism often there is `Base *b = new Derived;`.
```cpp
Base *b = new Derived;
Base &b1 = *b;
```

To obtain `b`'s type, RTTI comes into rescue at runtime by the following
* `typeid` operator and `type_info` class
* `dynamic_cast` operator

The above two implementations need dynamic lookup at runtime for type resolution, hence introducing additional table lookup cost.

### `typeid` and `type_info`

`typeid` operator result is `type_info`. 
The class `type_info` holds implementation-specific information about a type, including name and equality.

For equality, `operator==` and `operator!=` (removed in C++20)
checks whether the objects refer to the same type.

For name, `const char* name() const noexcept;` (since C++11) returns an implementation defined null-terminated character string containing the name of the type. 

Some implementations (such as MSVC, IBM, Oracle) produce a human-readable type name. Others, most notably gcc and clang, return the mangled name, which is specified by the *Itanium C++ ABI* (a rule of resolving unique variable names). 
The mangled name can be converted to human-readable form using implementation-specific API such as `abi::__cxa_demangle`.

Itanium C++ ABI RTTI specifications are shown as below.
```c++
abi::__fundamental_type_info
abi::__array_type_info
abi::__function_type_info
abi::__enum_type_info
abi::__class_type_info
abi::__si_class_type_info
abi::__vmi_class_type_info
abi::__pbase_type_info
abi::__pointer_type_info
abi::__pointer_to_member_type_info
```

### `typeid` and Virtual Methods

Since `typeid()` returns a `const std::type_info&` that stores type info of an object.
If an object/class contains one or many virtual functions, `typeid` can only find the type at runtime, opposed to compile time if the object/class has no virtual functions.

Look at this snippet of code:
`Base` has a virtual member function `fun`, 
and `Derived` inherited from `Base` defines `fun`.
Finally in `main`, do this test `typeid(Base *b)`.

```cpp
#include <iostream>
#include <typeinfo>

class Base{
public:
     virtual void fun() {}
};

class Derived : public Base{
public:
     void fun() {}
};

void fun(Base *b) {
  const std::type_info &info = typeid(b);
}

int main() {
  Base *b = new Derived;
  fun(b);
  
  return 0;
}
```

The ASM code is shown as below, where `.quad` describes a pointer entry.

The offset `OFFSET FLAT` in `fun(Base*):` points to `typeinfo for Base*` (`Base*` is a variable, not a fixed pointer). If `Derived*` is passed to `fun(Base*):`, the `Base*` pointer points to `typeinfo for Derived` that then points to `typeinfo name for Derived`.

```x86asm
fun(Base*):
        push    rbp
        mov     rbp, rsp
        mov     QWORD PTR [rbp-24], rdi
        mov     QWORD PTR [rbp-8], OFFSET FLAT:typeinfo for Base*
        pop     rbp
        ret
vtable for Derived:
        .quad   0
        .quad   typeinfo for Derived
        .quad   Derived::fun()
vtable for Base:
        .quad   0
        .quad   typeinfo for Base
        .quad   Base::fun()
typeinfo name for Base*:
        .string "P4Base"
typeinfo for Base*:
        .quad   vtable for __cxxabiv1::__pointer_type_info+16
        .quad   typeinfo name for Base*
        .long   0
        .zero   4
        .quad   typeinfo for Base
typeinfo name for Derived:
        .string "7Derived"
typeinfo for Derived:
        .quad   vtable for __cxxabiv1::__si_class_type_info+16
        .quad   typeinfo name for Derived
        .quad   typeinfo for Base
typeinfo name for Base:
        .string "4Base"
typeinfo for Base:
        .quad   vtable for __cxxabiv1::__class_type_info+16
        .quad   typeinfo name for Base
```

If there is no virtual method in a class such as
```cpp
#include <iostream>
#include <string>
#include <typeinfo>

class MyClss {};

int main() {
  MyClss s;
  const std::type_info &info = typeid(s);
  
  return 0;
}
```

the compiled ASM would be below that `OFFSET FLAT:typeinfo for MyClss` directly points to `MyClass`.
```x86asm
main:
        push    rbp
        mov     rbp, rsp
        mov     QWORD PTR [rbp-8], OFFSET FLAT:typeinfo for MyClss
        mov     eax, 0
        pop     rbp
        ret
typeinfo name for MyClss:
        .string "6MyClss"
typeinfo for MyClss:
        .quad   vtable for __cxxabiv1::__class_type_info+16
        .quad   typeinfo name for MyClss
```

### `dynamic_cast`

Generally speaking, `dynamic_cast` finds out inheritance relationships of the `src` vs `dst` class, 
then maps the object memory content reinterpreted by the desired target class.

`dynamic_cast` has the below implementation.
1. `vtable` just de-references `src_ptr`
2. `vtable_prefix *prefix` addr of derived object virtual table
3. `*whole_ptr` the most derived class from `src_ptr`
4. `whole_type` the type of most derived class from `src_ptr`, such as `class`, `si_class` (single inheritance), `vmi_class` (virtual/multiple inheritance)
5. `whole_vtable` the virtual table of the most derived class
```cpp
__dynamic_cast (const void *src_ptr,    // object started from
                const __class_type_info *src_type, // type of the starting object
                const __class_type_info *dst_type, // desired target type
                ptrdiff_t src2dst) // how src and dst are related
{
  const void *vtable = *static_cast <const void *const *> (src_ptr);
  const vtable_prefix *prefix =
      adjust_pointer <vtable_prefix> (vtable,
              -offsetof (vtable_prefix, origin));
  const void *whole_ptr =
      adjust_pointer <void> (src_ptr, prefix->whole_object);
  const __class_type_info *whole_type = prefix->whole_type;
  __class_type_info::__dyncast_result result;

  // If the whole object vptr doesn't refer to the whole object type, we're
  // in the middle of constructing a primary base, and src is a separate
  // base.  This has undefined behavior and we can't find anything outside
  // of the base we're actually constructing, so fail now rather than
  // segfault later trying to use a vbase offset that doesn't exist.
  const void *whole_vtable = *static_cast <const void *const *> (whole_ptr);
  const vtable_prefix *whole_prefix =
    adjust_pointer <vtable_prefix> (whole_vtable,
            -offsetof (vtable_prefix, origin));
  const void *whole_vtable = *static_cast <const void *const *> (whole_ptr);
  const vtable_prefix *whole_prefix =
    (adjust_pointer <vtable_prefix>
     (whole_vtable, -ptrdiff_t (offsetof (vtable_prefix, origin))));
  if (whole_prefix->whole_type != whole_type)
    return NULL;

  // Avoid virtual function call in the simple success case.
  if (src2dst >= 0
      && src2dst == -prefix->whole_object
      && *whole_type == *dst_type)
    return const_cast <void *> (whole_ptr);

  whole_type->__do_dyncast (src2dst, __class_type_info::__contained_public,
                            dst_type, whole_ptr, src_type, src_ptr, result);
}

// Adjust a pointer by an offset.
const void*
adjust_pointer(const void* p, std::ptrdiff_t off)
{
 // FIXME: should we align pointer after adjustment?
 const char *cp = reinterpret_cast<const char*>(p) + off;
 return reinterpret_cast<const void*>(cp);
}
``` 
where `ptrdiff_t src2dst` is the offset from `dst` to `src`; it has special meanings when value is one of the below
* -1: no hint
* -2: src is not a public base of dst (such as `Base1 *bc1 = dynamic_cast<Derived*>(b2);`)
* -3: src is a multiple public base type but never a virtual base type (no diamond pattern inheritance)

### `dynamic_cast` Example

This snippet of code below describes two base class inheritances `class Derived : public Base1, public Base2` where both `Base1` and `Base2` have virtual methods,

and how compiler would handle different forms of casting.

```cpp
#include <iostream>
#include <typeinfo>

class Base1 {
public:
  void f0() {}
  virtual void f1() {}
  int a;
};

class Base2 {
public:
  virtual void f2() {}
  int b;
};

class Derived : public Base1, public Base2 {
public:
  void d() {}
  void f2() {}  // override Base2::f2()
  int c;
};

int main() {
  Derived *d = new Derived;
  Base1 *b1 = new Derived;
  Base2 *b2 = new Derived;

  Base2 *bc2 = dynamic_cast<Base2*>(d); // upcasting 
  Derived *dc1 = dynamic_cast<Derived*>(b1); // downcasting 
  Base1 *bc1 = dynamic_cast<Derived*>(b2);
  
  return 0;
}
```

* Up Casting

`Base2 *b2 = dynamic_cast<Base2*>(d);` is identical to `static_cast` that casting finishes at compile time.

* Down Casting

By the existing virtual table for the derived class, `dynamic_cast` uses *depth first search* to traverse inheritance tree and finds the base class.

## Cache Hit Performance Issues in Employing Virtual Methods

Virtual table dynamic lookup can cause performance issues for cache miss.

In the three examples below, out-of-order virtual method repeated invocations see significantly decrease in performance.


For `class Derived : public Base`
```c++
class Base {
  public:
  virtual void foo() {}
};

class Derived : public Base {
  public:
  int a;
  int c;
  virtual void virCall() {
    a = 10;
  }
  void nonVirCall() {
    c = 10;
  }
};
```
* Non-virtual method repeated invocations

```cpp
Derived d;
for(int i=0; i<100000; ++i){
    d.nonVirCall();
}
```
* Virtual method repeated invocations

```cpp
Derived d;
for(int i=0; i<100000; ++i){
    d.virCall();
}
```

* Out-of-order virtual method repeated invocations

```cpp
std::vector<Derived> dArr;
dArr.resize(100000);
for(int i=0; i<100000; ++i){
    Derived d;
    dArr[i] = d;
}
std::random_shuffle(dArr.begin(), dArr.end());
for(auto& item: dArr){
    item->virCall();
}
```