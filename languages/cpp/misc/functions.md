# Functions

* Function Pointers - These are a feature of the C language and so form part of the C++ standard. A function pointer allows a pointer to a function to be passed as a parameter to another function.

* Function Objects (Functors) - C++ allows the function call operator() to be overloaded, such that an object instantiated from a class can be "called" like a function.

* Lambda - Anonymous function.

* C++11 `<function>` - C++11 brought new changes to how functors were handled. In addition, anonymous functions (lambdas) are now supported.

## Function Pointer

In the code below, inside `binary_op(...)`, `(*f)` can take either `add(..)` or `multiply(...)`.

```cpp
double add(double left, double right) {
    return left + right;
}

double multiply(double left, double right) {
    return left * right;
}

double binary_op(double left, double right, double (*f)(double, double)) {
    return (*f)(left, right);
}

int main() {
    double a = 5.0;
    double b = 10.0;

    binary_op(a, b, add);
    binary_op(a, b, multiply);

    return 0;
}
```

### Brain Teaser: Order of Execution

```cpp
int (*((*ptr(int, int)))) (int); 
```

Explain:
```cpp
// function return to a pointer
*ptr(int, int)

// take the return pointer as an arg
(*ptr(int, int))

// extra parentheses does not make any difference
((*ptr(int, int)))

// function pointer to pointer
*((*ptr(int, int)))

// function pointer to int pointer
int (*((*ptr(int, int))))
```

## Function Object

Overloading `operator()` to make a function objetc.

```cpp
struct A {
   int x; // state member can even be made private! Instance per functor possible
   int operator()(int y) { return x+y }
};
```

The lambda can be a function object as well
```cpp
auto lambda = [&x](int y) { return x+y };
```

## Lambda: Anonymous Function

```cpp
float x[5] = {5,4,3,2,1};
std::sort(x, x + n,
    [](float a, float b) {
        return (std::abs(a) < std::abs(b));
    }
);
```

`[]` is called capture clause, that `[=]` means *by value capture* while `[&]` is *by reference capture*.

### Closure

Compiler generates a *closure class* and a derived *closure object* from lambda. *Closure* is the code block of a lambda.

Data inside a closure has lifecycle inside the closure.

### Default by-reference capture can lead to dangling references

If the lifetime of a closure created from that lambda exceeds the lifetime of the local variable or parameter, the reference in the closure will dangle.

For example, we want to create a vector of functions (each element is a function handle), and the function is in risks of having dangling vars.
```cpp
using FilterContainer = std::vector<std::function<bool(int)>>; 
FilterContainer filters;

void addDivisorFilter()
{
    auto calc1 = computeSomeValue1();
    auto calc2 = computeSomeValue2();
    auto divisor = computeDivisor(calc1, calc2);

    // divisor reference will dangle, is dead after filters.emplace_back returns
    filters.emplace_back(
        [&](int value) { return value % divisor == 0; }
    );
}
```

Solution is 
1. by using `[=]` copy capture
2. directly finish computation if possible, rather than storing function handles to a vector containers
```cpp
std::all_of(begin(container), end(container),
                [&](const auto& value) { return value % divisor == 0; }
)
```

### Capture by move

Be careful about using `move`

Below: create a data member `pw` in the closure, and initialize that data member with the result of applying `std::move` to the local variable `pw`. `pw` expires after lambda returns.

```cpp
auto pw = std::make_unique<Widget>();

auto func = [pw = std::move(pw)] { 
                return pw->isValidated() && pw->isArchived(); 
            };
```

### `std::bind`

`std::bind` generates a forwarding call wrapper for `f`. Calling this wrapper is equivalent to invoking f with some of its arguments bound to args:
```cpp
template< class F, class... Args >
T bind( F&& f, Args&&... args );
```
where:
1.	`f`: Callable object (function object, pointer to function, reference to function, pointer to member function, or pointer to data member) that will be bound to some arguments
2. `args`: the unbound arguments replaced by the placeholders `_1`, `_2`, `_3`... of namespace `std::placeholders`

### Arg Invocation Order

Example: `setAlarm` is a customary function that gets triggered after one hour and goes "beep" for 30 seconds.

However, in the below code, `steady_clock::now() + 1h` is passed as an argument to `std::bind`, not directly to `setAlarm`. This is wrong in business logic that `steady_clock::now() + 1h` should be evaluated when `setAlarm` is invoked.

```cpp
using namespace std::chrono;
using namespace std::literals; // as above
using namespace std::placeholders; // needed for use of "_1"

auto setSoundB = \
std::bind(setAlarm,
        steady_clock::now() + 1h,
        _1,
        30s);
```

Solution is as below
```cpp
auto setSoundB =
std::bind(setAlarm,
        std::bind(std::plus<steady_clock::time_point>(), 
                steady_clock::now(), 
                1h),
        _1,
        30s);
```

## `std::function`

`std::function` is a function wrapper that makes a function an object so that it can be assigned/passed as an object.
```cpp
int add(int a, int b) {
    return a + b;
}
int main()
{
    std::function<int(int, int)> f1 = add;
    std::cout << f1(3, 5) << std::endl; // 8
    return 0;
}
```


### `std::function` performance issues

Reference source: https://blog.demofox.org/2015/02/25/avoiding-the-performance-hazzards-of-stdfunction/

## 