# Kernel Memory

## `kmalloc`、`kzalloc`、`vmalloc`

### `kmalloc`

`kmalloc` can allocate contiguous physical memory; max allocation size is $128$ kb. 
```cpp
void * kmalloc(size_t size, gfp_t, flags);
void kfree(const void * objp);
```
Typical flags are
* `GFP_ATOMIC`: Atomic allocation operation, cannot be interrupted by high-priority processes
* `GFP_KERNEL`: Memory allocation as normally
* `GFP_DMA`: Use DMA to allocate memory (DMA requires both virtual and physical memory being contiguous)

### `kzalloc`

`kzalloc` is same as `kmalloc` besides adding `__GFP_ZERO` that sets memory to zeros, such as
```cpp
static inline void* kzalloc(size_t size, gfp_t, flags){
    return kmalloc(size, flags | __GFP_ZERO);
}
```

### `vmalloc`

`vmalloc` allocates a contiguous block of memory on virtual memory (might not be contiguous on physical devices), good for large size memory allocations.

```cpp
void * vmalloc(unsigned long size);
void vfree(const void * addr);
```