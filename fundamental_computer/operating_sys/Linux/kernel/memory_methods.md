# Kernel Memory Methods

## Kernel memory allocation

### `kmalloc`

`kmalloc` can allocate contiguous physical memory in kernel space; max allocation size is $128$ kb. 
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

## Kernel and user memcpy

The below `move_addr_to_kernel` and `move_addr_to_user` are used to pass data between kernel space and user space.

The implementations of `memcpy_fromfs` and `memcpy_tofs` are simple `memcpy`. `fs` in this context refers to a segment register in CPU that performs linear addressing. 

```cpp
static int move_addr_to_kernel(void *uaddr, int ulen, void *kaddr)
{
    int err;
    if(ulen<0||ulen>MAX_SOCK_ADDR)
        return -EINVAL;
    if(ulen==0)
        return 0;
    // if user memory block pointed by `uaddr` is contiguous and readable
    if((err=verify_area(VERIFY_READ,uaddr,ulen))<0)
        return err;
    memcpy_fromfs(kaddr,uaddr,ulen);
        return 0;
}
```


```cpp
static int move_addr_to_user(void *kaddr, int klen, void *uaddr, int *ulen)
{
    int err;
    int len;

    // if user memory block pointed by `ulen` is writable
    if((err=verify_area(VERIFY_WRITE,ulen,sizeof(*ulen)))<0)
        return err;
    len=get_fs_long(ulen); //get the size of ulen 
    if(len>klen)
        len=klen;// limit the size
    if(len<0 || len> MAX_SOCK_ADDR)
    return -EINVAL;
    if(len)
    {
    // // if user memory block pointed by `ulen` is writable
    if((err=verify_area(VERIFY_WRITE,uaddr,len))<0)
        return err;
    memcpy_tofs(uaddr,kaddr,len);
    }
    put_fs_long(len,ulen);// *ulen = len, returns the actual copied length of data as the result
    return 0;
}
```