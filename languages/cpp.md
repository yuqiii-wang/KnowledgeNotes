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

## Move and Forward

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
    wrap(s); // lvalaue
    wrap(S()); // rvalue
    return 0;
}
```

## Shared Pointer's Realization

### Virtual Fuction Realization

### Factory in CPP

### Private, Protected and Friend

### GDB Common Debugging Practicess

* i++ vs ++i

++i will increment the value of i, and then return the incremented value.
```cpp
int i = 1;
int j = ++i;
// (i is 2, j is 2)
```
i++ will increment the value of i, but return the original value that i held before being incremented.
```cpp
int i = 1;
int j = i++;
// (i is 2, j is 1)
```