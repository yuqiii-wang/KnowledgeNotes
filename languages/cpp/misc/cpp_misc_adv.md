# Some C++ Advanced Knowledge

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
