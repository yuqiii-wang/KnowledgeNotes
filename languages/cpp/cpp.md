# CPP Common Interview Questions

## Shared Pointer's Realization

## STL Container Thread-Safe Access and Modification

## Garbage Collection, Constructor and Destructor

### placement new

As it allows to construct an object on memory that is already allocated, it is required for optimizations as it is faster not to re-allocate all the time. It is useful for object been re-constructed multiple times.

```cpp
int main() {
    // buffer on stack, init with 2 elems
    unsigned char buf[sizeof(int)*2] ;
  
    // placement new in buf
    int *pInt = new (buf) int(3);
    int *qInt = new (buf + sizeof (int)) int(5);

    // pBuf and pBuf are addrs of buf and buf+1 respectively, with init int values. 
    int *pBuf = (int*)(buf+0) ;
    int *pBuf = (int*) (buf + sizeof(int) * 1);

    return 0;
}
```