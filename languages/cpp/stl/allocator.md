# Allocator

The `std::allocator` class template is the default Allocator used by all standard library containers.

```cpp
// default allocator for ints
std::allocator<int> alloc;

// demonstrating the few directly usable members
static_assert(std::is_same_v<int, decltype(alloc)::value_type>);
int* p = alloc.allocate(1);  // space for one int
alloc.deallocate(p, 1);      // and it is gone
```

## Allocator vs `new`