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

* Compilers are unable to deduce a type for one or more of fwd's parameters
* Compilers deduce the "wrong" type

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

## Move Implementation

The standard template provides this implementation.

```cpp
template<typename T>
constexpr std::remove_reference_t<T>&& move(T&& t) noexcept
{
    return static_cast<std::remove_reference_t<T>&&>(t);
}
```

It’s essentially a `static_cast`: take in some reference – lvalue or rvalue, const or non-const – and casting it to an rvalue reference.

When write `Type obj = std::move(other_obj);`, overload resolution should call the move constructor `Type(Type&& other)` instead of the copy constructor `Type(const Type& other)`.

### `noexcept` and Move

Compiler would not use the move constructor of an object if that can throw an exception. 
This is because if an exception is thrown in the move then the data that was being processed could be lost, where as in a copy constructor the original will not be changed.

In code, `noexcept` should be added for move to be selected in use.
```cpp
class A {
public:
    A(A&& _A) noexcept {}
};

class B {
public:
    B(A&& _B) {}
};

int main() {
    std::vector<A> va;
    A a;
    va.push_back(a); // call move constructor

    std::vector<B> vb;
    B b;
    vb.push_back(b); // call copy constructor
}

```

### Move Failure

* Move pointer: move cannot work on pointer (the pointed object does not change before and after move operation)
* Return Value Optimization may have different implementations of whether it uses default/copy/move constructor

Bad for copy elision:
```cpp
S f()
{
  S result;
  return std::move(result);
}
```

Good:

```cpp
S f() {
  S result;
  return result;
}
```

* When constructor could throw error, by default copy constructor is called

## Move And Perfect Forwarding Example

General rules:

* `std:move`: Resource Management: Classes managing resources like file handles, network connections, copying data from to-be-demised resources by efficient ownership transfer
* `std::forward`: Forward variables with retained value category (l-value or r-value).

For example, below code (compile by `g++ move_perf_forwarding.cpp -std=c++20`) shows a `CustomVector` how it grows with new allocated memory.

Given objects added by `emplace_back(Args&&... args)`

1. `auto newData = std::make_unique<T[]>(capacity_);` invokes default constructor for memory allocation only (hence the default constructor should do nothing)
2. `new (&newData[i]) T(std::move(data_[i]));` invokes move constructor to "move" objects from old memory addr to new larger memory space.
3. For new objects that can be allocated in the existing vector memory space, use placement new `new (&data_[size_]) T(std::forward<Args>(args)...);` to build the object.

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <memory>

struct Person;

template <typename T>
class CustomVector {
public:
    // data_(std::make_unique<T[]>(1)) inits a T array of size 1, invoked once default constructor 
    CustomVector() : size_(0), capacity_(1), data_(std::make_unique<T[]>(1)) {}

    template <typename... Args>
    void emplace_back(Args&&... args) {
        if (size_ >= capacity_) {
            // Resize the storage if needed
            resize();
        }
        // Construct the element in place using placement new and perfect forwarding
        new (&data_[size_]) T(std::forward<Args>(args)...);
        ++size_;
    }

    void resize() {
        capacity_ *= 2;
        // called default constructors for mem allocation (hence the default constructor is empty of code execution)
        // it allocates a block of contiguous mem for the T array
        // then use placement new to "move construct" the exact values to the allocated mem
        auto newData = std::make_unique<T[]>(capacity_);
        for (size_t i = 0; i < size_; ++i) {
            new (&newData[i]) T(std::move(data_[i])); // Move constructor
            data_[i].~T(); // Explicitly destroy the old elements
        }
        data_ = std::move(newData);
        std::cout << "Resize triggered, new capacity is " << capacity_ << std::endl;
        std::cout << "New data addr is" << std::endl;
        print_data_addr();
    }

    T& operator[](size_t index) {
        if (index >= size_) {
            throw std::out_of_range("Index out of range");
        }
        return data_[index];
    }

    void print_data_addr(){
        for (size_t i = 0; i < size_; ++i) {
            std::cout << &data_[i] << std::endl;
        }
    }

    int get_size() {
        return size_;
    }

    ~CustomVector() {
        for (size_t i = 0; i < size_; ++i) {
            data_[i].~T(); // Explicitly destroy elements
        }
    }

private:
    size_t size_;
    size_t capacity_;
    std::unique_ptr<T[]> data_;
};

struct Person {
    Person() {
        std::cout << "Default person constructor is invoked." << std::endl;
    };

    Person(std::string name, int age, char gender):
    _name(name), _age(age), _gender(gender) {
        std::cout << "Person is identified as " << _name << std::endl;
    }

    // Move constructor
    Person(Person&& another_person) noexcept : 
    _name(another_person._name), _age(another_person._age), _gender(another_person._gender) {
        std::cout << "Move constructor is invoked." << std::endl;
    }

    // Copy constructor
    Person(const Person& another_person) : 
    _name(another_person._name), _age(another_person._age), _gender(another_person._gender) {
        std::cout << "Copy constructor is invoked." << std::endl;
    }

    friend std::ostream& operator<<(std::ostream& os, const Person& person);

    std::string _name;
    int _age;
    char _gender;
};


// Overload the << operator for the Person class
std::ostream& operator<<(std::ostream& os, const Person& person) {
    os << "Name: " << person._name << ", Gender: " << person._gender << ", Age: " << person._age;
    return os;
}  

int main() {
    CustomVector<Person> vec;

    vec.emplace_back("Yuqi", 29, 'M') ;
    vec.emplace_back("Sexy Yuqi", 28, 'M') ;
    vec.emplace_back("Wild Yuqi", 27, 'M') ;
    vec.emplace_back("Crazy Yuqi", 26, 'M') ;
    vec.emplace_back("Magnificent Yuqi", 25, 'M') ;

    for (int idx = 0; idx < vec.get_size(); idx++) {
        std::cout << "Person: {" << vec[idx] << "}" << std::endl;
    }
    
    return 0;
}
```

that see this output

```txt
Default person constructor is invoked.
Person is identified as Yuqi
Default person constructor is invoked.
Default person constructor is invoked.
Move constructor is invoked.
===============================================
Resize triggered, new capacity is 2
New data addr is
0x7f7c9a705be8
Person is identified as Sexy Yuqi
Default person constructor is invoked.
Default person constructor is invoked.
Default person constructor is invoked.
Default person constructor is invoked.
Move constructor is invoked.
Move constructor is invoked.
===============================================
Resize triggered, new capacity is 4
New data addr is
0x7f7c9a705c38
0x7f7c9a705c58
Person is identified as Wild Yuqi
Person is identified as Crazy Yuqi
Default person constructor is invoked.
Default person constructor is invoked.
Default person constructor is invoked.
Default person constructor is invoked.
Default person constructor is invoked.
Default person constructor is invoked.
Default person constructor is invoked.
Default person constructor is invoked.
Move constructor is invoked.
Move constructor is invoked.
Move constructor is invoked.
Move constructor is invoked.
===============================================
Resize triggered, new capacity is 8
New data addr is
0x7f7c9a705cc8
0x7f7c9a705ce8
0x7f7c9a705d08
0x7f7c9a705d28
Person is identified as Magnificent Yuqi
Person: {Name: Yuqi, Gender: M, Age: 29}
Person: {Name: Sexy Yuqi, Gender: M, Age: 28}
Person: {Name: Wild Yuqi, Gender: M, Age: 27}
Person: {Name: Crazy Yuqi, Gender: M, Age: 26}
Person: {Name: Magnificent Yuqi, Gender: M, Age: 25}
```