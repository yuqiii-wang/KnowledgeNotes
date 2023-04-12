# Allocator

`std::allocator` by default calls `malloc` for memory allocation.

The `std::allocator` class template is the default Allocator used by all standard library containers.

```cpp
// default allocator for ints
std::allocator<int> alloc;

// demonstrating the few directly usable members
static_assert(std::is_same_v<int, decltype(alloc)::value_type>);
int* p = alloc.allocate(1);  // space for one int
alloc.deallocate(p, 1);      // and it is gone
```

### Allocator Implementation

* `address` (until C++20)

(public member function) obtains the address of an object, even if operator& is overloaded

* `allocate`
 
(public member function) Allocates `n * sizeof(T)` bytes of uninitialized storage by calling `::operator new(std::size_t)` or `::operator new(std::size_t, std::align_val_t)` (since C++17)

Typical use case in a custom allocator `MyLib::MyAlloc<T>`:
```cpp
std::vector<int,MyLib::MyAlloc<int> > v;
```
where `v` grows in size, if `v.size()` is near to `std::numeric_limits<std::size_t>::max() / sizeof(T)`, allocation occurs and `allocate` get invoked 
and `pointer ret = (pointer)(::operator new(num*sizeof(T)));` that assigns more memory, where the new memory size `num` grows exponentially specified by `std::vector<T>`.

* `allocate_at_least` (C++23)
 
(public member function) allocates uninitialized storage at least as large as requested size

* `deallocate`
 
(public member function) deallocates storage

* `max_size` (until C++20)
 
returns the largest supported allocation size
(public member function)

* `construct` (until C++20)
 
(public member function) Constructs an object of type T in allocated uninitialized storage pointed to by p, using placement-new

Calls `new((void *)p) T(val)`

Calls `::new((void *)p) U(std::forward<Args>(args)...)`

* `destroy` (until C++20)
 
(public member function) destructs an object in allocated storage

### `new` When Memory is Insufficient

Linux attempts to swap data between disk and primary memory through paging mapping.

`std::bad_alloc` can occur if allocation fails.

### Allocator with `placement new`

Recall that `placement new` is faster for no need to re-allocate all the time. It is useful for object been re-constructed multiple times.

It has two ways of declaration:
```cpp
void* operator new(std::size_t, const std::nothrow_t&) throw();
void* operator new(std::size_t, void*) throw();
```

`std::allocator` member function `construct` is the implementation of `placement new`, that is automatically invoked initializing 
```cpp
std::vector<int,MyLib::MyAlloc<int> > v;

for (int i = 0; i < 10; i++)
    v.push_back(i);
```