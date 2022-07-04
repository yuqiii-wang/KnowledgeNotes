# Some C++ Advanced Knowledge

### `std::unordered_map` thread safety

When multi-threading running on `std::unordered_map` performing simultaneous read and write on elements, it is NOT thread-safe.

When inserting a new element, there is rehashing, so that iterator is not valid. However, reference remains valid (element memory unchanged).

### `inline` performance issues

An inline function is one for which the compiler copies the code from the function definition directly into the code of the calling function rather than creating a separate set of instructions in memory, hence improving performance.

However, it might reduce performance if misused, for increased cache misses and thrashing.


### Curiously recurring template pattern (CRTP)

CRTP is an idiom in C++ in which a class X derives from a class template instantiation using X itself as a template argument.

```cpp
// The Curiously Recurring Template Pattern (CRTP)
template <class T>
class Base
{
    // methods within Base can use template to access members of Derived
};
class Derived : public Base<Derived> {};
```

* Use example: object counter

```cpp
template <typename T>
struct counter
{
    static inline int objects_created = 0;
    static inline int objects_alive = 0;

    counter()
    {
        ++objects_created;
        ++objects_alive;
    }
    
    counter(const counter&)
    {
        ++objects_created;
        ++objects_alive;
    }
protected:
    ~counter() // objects should never be removed through pointers of this type
    {
        --objects_alive;
    }
};

class X : counter<X>
{
    // ...
};

class Y : counter<Y>
{
    // ...
};
```

### Substitution failure is not an error (SFINAE) 

An invalid substitution of template parameters is not in itself an error.

```cpp
struct Test {
  typedef int foo;
};

template <typename T>
void f(typename T::foo) {}  // Definition #1

template <typename T>
void f(T) {}  // Definition #2

int main() {
  f<Test>(10);  // Call #1.
  f<int>(10);   // Call #2. Without error (even though there is no int::foo)
                // thanks to SFINAE.
}
```

### `noexcept`

Compiler uses *flow graph* to optimize machine code generation. A flow graph consists of what are generally called "blocks" of the function (areas of code that have a single entrance and a single exit) and edges between the blocks to indicate where flow can jump to. `Noexcept` alters the flow graph (simplifies flow graph not to cope with any error handling)

For example, code below using containers might throw `std::bad_alloc` error for lack of memory, and compiler needs attaching `std::terminate()` when error was thrown, hence adding complexity to flow graph. Remember, there are many errors a function can throw, and error handling code blocks can be many in a flow graph. By `noexcept`, flow graph is trimmed. 
```cpp
double compute(double x) noexcept {
    std::string s = "Courtney and Anya";
    std::vector<double> tmp(1000);
    // ...
}
```

### `restrict`

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
