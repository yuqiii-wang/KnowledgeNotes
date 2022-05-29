# Lvalue vs Rvalue Performance

## Use of Universal References and Overloading

Given a global obj `names` and a method that adds `std::string` to the global obj `names`:
```cpp
std::multiset<std::string> names;

void logAndAdd(const std::string& name)
{
  auto now = std::chrono::system_clock::now();// get current time
  log(now, "logAndAdd"); // make log entry
  names.emplace(name);   // add name to global data
                          // structure
}
```

All three forms of invocation need copy, the first one is lvalue copy, no way to optimize; The second and third are rvalue copy, can be optimized by universal reference `T&&`
```cpp
std::string petName("Darla");
logAndAdd(petName); // pass lvalue std::string

logAndAdd(std::string("Persephone")); // pass rvalue std::string,
                                      // the temporary std::string explicitly created 
                                      // from "Persephone".

logAndAdd("Patty Dog"); // pass string literal, 
                        // std::string that’s implicitly created from "Patty Dog"
```

Rewrite `logAndAdd` to using `T&&` with `forward`
```cpp
template<typename T>
void logAndAdd(T&& name)
{
    auto now = std::chrono::system_clock::now();
    log(now, "logAndAdd");
    names.emplace(std::forward<T>(name));
}
```

Now the behavior of three invocations is optimized:
```cpp
// as before, copy
// lvalue into multiset
std::string petName("Darla");
logAndAdd(petName);

// move rvalue instead
// of copying it
logAndAdd(std::string("Persephone"));

// create std::string
// in multiset instead
// of copying a temporary
// std::string
logAndAdd("Patty Dog"); 
```

However, the above code might be bulky if `logAndAdd` is overloaded, such as read/write by `int`:
```cpp
std::string nameFromIdx(int idx); // return name
                                  // corresponding to idx

void logAndAdd(int idx)
{
    auto now = std::chrono::system_clock::now();
    log(now, "logAndAdd");
    names.emplace(nameFromIdx(idx));
}
```

Given the use of template `T&&`, when running 
```cpp
int nameIdxInt = 2;
logAndAdd(nameIdxInt); // good, we have an overloaded int func

short nameIdxShort = 2;
logAndAdd(nameIdxShort); // error, short does not match int overloaded func, 
                         // it calls `void logAndAdd(T&& name)` instead
```
Error occurs as overloading fails for invoking this function included using `names.emplace(std::forward<T>(name));`, in which `std::multiset<std::string> names` does not take `short` to construct a new element.
```cpp
template<typename T>
void logAndAdd(T&& name)
{
    auto now = std::chrono::system_clock::now();
    log(now, "logAndAdd");
    names.emplace(std::forward<T>(name));
}
```

The solution is 

* To add `explicit` that prevents using implicit conversions and copy-initialization.

* Use *Service-to-Implementation* design, such as
```cpp
template<typename T>
void logAndAddService(T&& name)
{
  logAndAddImpl(
    std::forward<T>(name),
    std::is_integral<typename std::remove_reference<T>::type>()
  );
}

// when passed int
void logAndAddImpl(int idx, std::true_type)
{
  logAndAdd(nameFromIdx(idx));
}

// when passed str
template<typename T>
void logAndAddImpl(T&& name, std::false_type)
{
  auto now = std::chrono::system_clock::now();
  log(now, "logAndAdd");
  names.emplace(std::forward<T>(name));
}
```

## Perfect Forwarding

A `forward` can be defined as below
```cpp
template<typename T>
T&& forward(typename
remove_reference<T>::type& param)
{
  return static_cast<T&&>(param);
}
```

It means that a function template can pass its arguments through to another function whilst retaining the lvalue/rvalue nature of the function arguments by using `std::forward`. This is called "perfect forwarding", avoids excessive copying, and avoids the template author having to write multiple overloads for lvalue and rvalue references.

### Forward Failures

* Compilers are unable to deduce a type for one or more of fwd’s parameters
* Compilers deduce the “wrong” type

For example,
```cpp
void f(const std::vector<int>& v);

template<typename T>
void fwd(T&& param)
{
  f(std::forward<T>(param));
}
```

Below is a failure as a braced initializer needs implicit conversion.
```cpp
fwd({ 1, 2, 3 }); // error! doesn't compile
```