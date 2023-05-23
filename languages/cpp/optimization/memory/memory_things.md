# Something About Memory

## Scope of Local Variables (A Compiler Perspective)

## Stack vs Heap

| | Stack | Heap | Explained |
|-|-|-|-|
| Cost of Usage | Low | High | for heap, need additional system call such as `malloc` for memory allocation |
| Deallocation of variable | Not necessary | Clear deallocation is important | Must use system call `free` to deallocate memory |
| Access time | Quick | Slow as compared to stack | Need additional memory reference to locate variables |
| Order for allocation of memory | Allocated as a contiguous block | Allocated in a random order | Stack memory is managed by stack pointer register to move up/down |

### Overflow Considerations

Overflow is invalid access to not allocated memory. If vars are declared on stack, it might trigger stack overflow; on heap, then heap overflow.

Examples:

* Invalid `memset`: allocated only 32 bytes, operated 64 bytes, HEAP overflow
```cpp
void* heap = (void*) malloc(32);
memset(heap, 'A', 64);
```

* requesting for too much stack memory, STACK overflow

```cpp
int a[99999999];
```

* recursively invoking a functions for too many times, it is a STACK overflow
```cpp
void foo() {
    foo();
}
```


## Memory Alignment

Given the `struct A`, for a 32-bit CPU, each time 4 contiguous memory bytes are loaded.
```cpp
struct A {
    char c;
    int i;
};
```

If in not-aligned layout (same as above `struct A`), to load `char c` and `int i`, require 3 loading operations: 1 for `char c` and 2 for `int i` (for it exists across two load instructions).

```cpp
struct A {
    int i;
    char c;
};
```
If in aligned layout such as first declaring `int i` then `char c`, require 2 loading operations: 1 for `char c` and 1 for `int i`.

<div style="display: flex; justify-content: center;">
      <img src="imgs/mem_alignment.png" width="30%" height="30%" alt="mem_alignment">
</div>
</br>

Another alternative is `alignas` (since c++11) that automatically aligns declared variables
```cpp
struct alignas(16) A {
  char c;
  int i;
};
```

## POD vs Trivial

## Memory Pool

Memory pool basically is a list that links many memory blocks. When this pool has too few blocks, it asks OS for more memory and appends new memory blocks to the list.

A typical implementation is `std::allocator`.

## `memmove()` vs `memcopy()`

`void *memcpy (void * restrict dst ,const void * src ,size_t n);` copies `n` bytes from memory from `src` location to memory pointed to `dst`.  
If `dst` and `src` area overlaps then behavior is undefined. 

Implementation shows as below as simply copying bytes by moving pointer `*(dst++) = *(src++);`.
It could be a problem if `dst` and `src` point to the same addr, resulting in undefined behavior.
```cpp
while(n) //till n
{
    //Copy byte by byte
    *(dst++) = *(src++);
    --n;
}
```

`void memmove(void *dst, const void *src, size_t n);` copies `n` bytes from `src` area to `dst` area.
These two areas might overlap; the copy process always done in a non-destructive manner.

Implementation shows as below that a temporary pointer is introduced, that takes care of the scenario when `src` and `dst` point to the same addr.
```cpp
char *tmp  = (char *)malloc(sizeof(char ) * n);
for(i =0; i < n ; ++i)
{
    *(tmp + i) = *(src + i);
}
//copy tmp to dest
for(i =0 ; i < n ; ++i)
{
    *(dst + i) = *(tmp + i);
}
free(tmp);
```