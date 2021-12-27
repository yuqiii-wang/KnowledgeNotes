# CPP Common Interview Questions

## Rvalue vs Lvalue

Put simply, an lvalue is an object reference and an rvalue is a value. An lvalue refers to an object that persists beyond a single expression. An rvalue is a temporary value that does not persist beyond the expression that uses it.

function return rvalue.

```cpp
int get(){
    return 0;
}
int main(){
    int val = get(); // get() returns rvalue
    return 0;
}
```

referece takes lvalue, so that
```cpp
void setVal(int& val){}
int main(){
    // this line of code below will fail compiler for that 10 is an immediate val (a temp rvalue)
    setVal(10);

    // however, if we assign an addr, compiler will pass it
    int val = 10;
    setVal(val);

    return 0;
}

```

for lvalue and rvalue as arguments, it is recommended to add overloads, rather than by `const&`
```cpp
void foo(X& x) {}
void foo(X&& x) { foo(x); }

// although the following func takes both, it is not of good practice
void foo(const X& x){}
```

### Example

Consider this, 
```cpp
Object A, B, C, D, E;
A = (B + (C + (D + E)));
```
However, `operator+` such as on `(D + E)` whose result is `+` to `C`, creates multiple temp objects and soon useless thus deleted, given the following Object definition.
```cpp
Object Object::operator+ (const Object& rhs) {
    Object temp (*this);
    // logic for adding
    return temp;
}
```

This can be addressed by 
```cpp
Object& Object::operator+ (Object&& rhs) {
    // logic to modify rhs directly
    return rhs;
}
```
This is known as `move` semantics, in which `const Object& rhs` is a lvalve reference and `Object&& rhs` changes to a modifiable rvalue reference.

### & vs && in var declaration

`auto& x` is simply a var reference.

`auto&& x` treats x as a temp rvalue without making it as `const&`.


## Move and Forward

### Summary

Both `move` and `forward` are introduced to deal with expensive "copy" operation when passing params; `move` is used to force using `rvalue reference` without copy, while `forward` is of added compatability handling `rvalue/lvalue reference` as well as `const reference`.

### Further explained

Consider this stastement:

```cpp
std::String src_str = "hello";
std::String dest_str = src_str;
```

inside, `=` is actually a copy assignment operator. If src_str is no long used, we can actually assign the addr of src_str to dest_str. To do that, we can

```cpp
std::String dest_str = std::move(src_str);
// The above statement is same as
std::String dest_str((std::String&&)src_str);
```
so that `src_str` becomes a temporary (an rvalue).

In contrast to `std::move` that treats an object as a temp rvalue, `std::forward` has a single use case: to cast a templated function parameter (inside the function) to the value category (lvalue or rvalue) the caller used to pass it. This allows rvalue arguments to be passed on as rvalues, and lvalues to be passed on as lvalues, a scheme called “perfect forwarding.”

In the example below, `wrap(T&& t)` has a param not deduced until invocation.
```cpp
struct S{};

void foo(S& s){}
void foo(S&& s){}

template<typename T>
void wrap(T&& t){
    foo(std::forward<T>(t));
}

int main(){
    S s;
    wrap(s); // lvalue
    wrap(S()); // rvalue
    return 0;
}
```

## Shared Pointer's Realization

## Virtual Function Realization

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