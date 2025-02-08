# Resource Acquisition Is Initialization (RAII)

RAII is a practice philosophy: a resource must be acquired before use to the lifetime of an object, and must be freed after use.

In other words, RAII rules should be kept in mind when developing code (especially pointer-alike resources) that when allocating resources (typically memory) during construction, should think of when they will be released.

Some rules:

* Constructors must throw exception when acquiring recourses fails

* Check and delete resources after the resources go out of scope (such as using `delete` to free resources and disconnecting a DB)

* Must apply lock for resources being modified by multiple threads (such as vars and files)

* Use smart pointers to manage resources that are used in different scopes

* Prevent multi-phase initialization: half init objects can be bad, some resource management can be hard

* Try best to allocate resource as local, as they will be released when out of scope

* Join the spawned threads or use thread pool management tool

* Use scope management wrapper to resources when init: `vector<int> a(10);` rather than `int a[10];`, use `lock_guard` on mutex rather than raw mutex. 

## Practice Examples

### `lock_guard` for `std::mutex`

The below code is bad for `std::mutex m` might never be freed.

```cpp
std::mutex m;
 
void bad() 
{
    m.lock();                    
    f();                         // if f() throws exception, mutex m will never be freed
    if(!everything_ok()) return; // return early, mutex m will never be freed
    m.unlock();                  
}
```

The problem can be solved by applying `lock_guard` that automatically checks ownership and releases resources when out of scope.

The class `lock_guard` is a mutex wrapper that provides a convenient RAII-style mechanism for owning a mutex for the duration of a scoped block.

```cpp
void good()
{
    std::lock_guard<std::mutex> lk(m); // RAII: apply resource management when the obj is init.
    f();                               
    if(!everything_ok()) return;       
}
```

### Remember Resource Release Before Throwing Exception

The function `f` is error-prone for `int *p` might not be released.
```cpp
void f(int i)   // Bad: possible leak
{
    int* p = new int[12];
    // ...
    if (i < 17) throw Bad{"in f()", i};
    // ... *p will not be released
}
```

* Correction by auto release by `unique_ptr`

```cpp
void f1(int i)   // OK: resource management done by a handle (but see below)
{
    auto p = make_unique<int[]>(12);
    // ...
    helper(i);   // might throw
    // ...
}
```

* Correction by using a local resource object

```cpp
void f5(int i)   // OK: resource management done by local object
{
    vector<int> v(12);
    // ...
    helper(i);   // might throw
    // ...
}
```

### Custom `final` to `try`/`catch`

C++ has no `final` keyword. Instead, consider implementing the `final` semantics in `catch` to clean up allocated resources.

### Use global error code to handle exception and resource cleanup

Launch a thread periodically checking if there is any error code flag being set to true, and checking if CPU is idle, plus some custom conditions such as in quant trading, do not do cleanup work during market open/close hours, since these periods often see high trading volumes.

If good, let the thread do cleanup tasks.

For any thrown exception, rather than immediately performing cleanup tasks `delete`ing resources, just return an error code to a global variable to be periodically checked by a dedicated recourse cleanup thread.

Directly handling exception from `catch` can be time-consuming, since CPU's prefetched instructions are discarded, and needs to load totally different blocks of code for exception handling.

* For example, to process a large number of trades in the quant business, first allocate a large number of `class Trade;` linked against each other in a list.
Then fill each `Trade` element's memory with the received data from MQ.
There are many old/invalid trades to be deleted as more and more trades come in.
Just need to `pop_front` the old element trades to a to-be deleted list, which will be picked by the cleanup thread performing `delete` task at a later time.

### Throw exception in constructor if some logic fails, preventing a not-init object

```cpp
class FileProc {
private:
    FILE* f;
public:
    FileProc(const string& name)
        :f{fopen(name.c_str(), "r")} {
            if (!f) throw runtime_error{"could not open" + name};
        // ...
    }
};

void f() {
    FileProc fileProc {"Zeno"}; // throws if file isn't open
}
```

### Lambda and Co-Routine Hazard

A lambda results in a closure object with storage, often on the stack, that will go out of scope at some point. 
When the closure object goes out of scope the captures will also go out of scope. 
Normal lambdas will have finished executing by this time so it is not a problem. 

Coroutine lambdas may resume from suspension after the closure object has destructed and at that point all captures will be use-after-free memory access.

```cpp
int value = get_value();
std::shared_ptr<Foo> sharedFoo = get_foo();
{
  const auto lambda = [value, sharedFoo]() -> std::future<void>
  {
    co_await something();
    // "sharedFoo" and "value" have already been destroyed
    // the "shared" pointer didn't accomplish anything
  };
  lambda();
} // the lambda closure object has now gone out of scope
```

Correction

```cpp
std::future<void> Class::do_something(int value, std::shared_ptr<Foo> sharedFoo)
{
  co_await something();
  // sharedFoo and value are still valid at this point
}

void SomeOtherFunction()
{
  int value = get_value();
  std::shared_ptr<Foo> sharedFoo = get_foo();
  do_something(value, sharedFoo);
}
```

### Temp object destruction

Compiler auto invokes destructor of a temp obj once its execution finishes.

The content of `p1` is undefined behavior, that `substr(1)` returns a temporary object which is soon destroyed automatically once this line of expression finishes running.
```cpp
string s1 = string("string1");
const char* p1 = s1.substr(1).data();
```

The correction would be this below.
```cpp
string s1 = string("string1");
string sTmp = s1.substr(1);
const char* p1 = sTmp.data();
```

### Threading

* Detached thread

A detached thread should have global/static resources,
for that a detached thread can be viewed as a global resource, that should have corresponding global variables to its scope.

Detached threads are not recommended in use for no joining in parent thread, hence hard to monitor resource release situations.

```cpp
void f(int* p)
{
    // ...
    *p = 99;
    // ...
}

int glob = 33;

void some_fct(int* p)
{
    int x = 77;
    std::thread t0(f, &x);           // bad
    std::thread t1(f, p);            // bad
    std::thread t2(f, &glob);        // OK
    auto q = make_unique<int>(99);
    std::thread t3(f, q.get());      // bad
    // ...
    t0.detach();
    t1.detach();
    t2.detach();
    t3.detach();
    // ...
}
```

* Pass POD variables by value not by reference/pointer unless necessary.

POD variable copy has little cost in comparison to passing by reference/pointer, but much better being managed.

* Use `shared_ptr` passing objects between threads



### Named Return Value Optimization (NRVO) and `constexpr`

NRVO is forbidden for constant expression.

`A a` gives an error for `constexpr A(): p(this) {}` that sets the object `this` pointer to `p`, then `return a;` does not guarantee the validity of the pointed `A a`;.

```cpp
struct A
{
    void *p;
    constexpr A(): p(this) {}
};
 
constexpr A g()
{
    A a;
    return a;
}
 
constexpr A a = g(); // error: a.p would be dangling and would point to a temporary
                     // with automatic storage duration
```

## A Typical Obj Lifecycle

```cpp
#include <memory>


template <typename T>
void life_of_an_object
{
    std::allocator<T> alloc;

    // 1. allocate/malloc 
    T * p = alloc.allocate(1);

    // 2. placement new run constructor
    new (p) T(); 

    // 3. to destroy the obk, run destructor
    p->~T();

    // 4. deallocate/free
    alloc.deallocate(p, 1);
}
```


## Container Cautions

When using containers such as `std::vector<T>`, if `T` has sub objects with allocated memory, must first free `T` before let `std::vector<T>` run out of scope. Smart pointer cannot detect if sub object memory is freed.