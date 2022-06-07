# Smart Pointer's Realization

## Shared Pointer

Shared pointers track the shared ownership through a reference count property. When a pointer goes out of scope, it decreases one reference count from its total number of registered ownership. When it goes to zero, the pointed memory is freed.

```cpp
#include <iostream> //main header
#include <memory>   //for smart pointers
using namespace std;//namespace

int main()
{
    std::shared_ptr<int> sh1 (new int);   
    std::shared_ptr<int> sh2 = shp1;    
    auto sh3 = sh1;                       
    auto sh4 = std::make_shared<int>();

    cout << sh1.get() << endl;
    cout << sh2.get() << endl;
    cout << sh3.get() << endl;
    cout << sh4.get() << endl;

/* 
    output: only sh4 is new
    0x1c66c20
    0x1c66c20
    0x1c66c20
    0x1c66c70
*/

  return 0;  
}
```

### Shared Pointer counter

Depending on realization, for `boost::shared_ptr`, a counter is defined in `private` in a `shared_ptr` container, in which it `new`s a counter. As a result, a `shared_ptr` counter resides in heap.

### Shared Pointer Passing Cost

When heavily passing shared pointer pointed var, use reference `const shared_ptr<T const>&` rather than by value (it creates a new pointer every time calling copy constructor `shared_ptr<T>::shared_ptr(const shared_ptr<T> &)`)
```cpp
void f(const shared_ptr<T const>& t) {...} 
```

## Weak Pointer

Dangling pointers and wild pointers are pointers that do not point to a valid object of the appropriate type. Weak pointers are used to "try" access the pointer to see if it is a dangling pointer by `lock()`.

Weak pointer manages more reference count than shared pointer, as it needs to track 

```cpp
// empty definition
std::shared_ptr<int> sptr;

// takes ownership of pointer
sptr.reset(new int);
*sptr = 10;

// get pointer to data without taking ownership
std::weak_ptr<int> weak1 = sptr;

// deletes managed object, acquires new pointer
sptr.reset(new int);
*sptr = 5;

// get pointer to new data without taking ownership
std::weak_ptr<int> weak2 = sptr;

// weak1 is expired!
if(auto tmp = weak1.lock())
    std::cout << "weak1 value is " << *tmp << '\n';
else
    std::cout << "weak1 is expired\n";

// weak2 points to new data (5)
if(auto tmp = weak2.lock())
    std::cout << "weak2 value is " << *tmp << '\n';
else
    std::cout << "weak2 is expired\n";
```
that outputs
```bash
weak1 is expired
weak2 value is 5
```

## Prefer `std::make_unique` and `std::make_shared` to direct use of `new`

Rule of thumb: Try to use standard tools as much as possible, otherwise, you risk program failure when OS or compiler upgrade to a higher version.

Smart pointer make is a simple forward operation as below:
```cpp
template<typename T, typename... Ts>
std::unique_ptr<T> make_unique(Ts&&... params)
{
    return std::unique_ptr<T>(new T(std::forward<Ts>(params)...));
}
```

## Auto Pointer and Unique Pointer

Note: `std::auto_ptr` is deprecated and `std::unique_ptr` is its replacement.

Same as `std::shared_ptr` but without counter.