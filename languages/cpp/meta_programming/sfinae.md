# Substitution Failure Is Not An Error (SFINAE)

SFINAE remediates the issue of template instantiation failure during compilation for that there are multiple same function signatures (same function names and arguments), that compiler cannot determine an exact match.
"Substitution" means passing/implementing actual variables to template `typename T`.

### Failure vs Error

Failure and Error refer to two different things.

For example, during compilation, when a syntax parser finds out that a token is not recognizable (does not match any reserved keyword), it throws an error. When it finds out that the token is part of the reserved keyword, but not compliant with syntax rules, it throws failure.

This means "failure" is a result of syntax rule not matched. SFINAE is a solution to this type of failure that tricks the compiler and lets it know how to find an exact rule match.

### Substitution

Compiler needs to determine the actual type of a templated function (its argument types). The resultant function signature should be unique.

One salient example is overloading. In the code below, compiler needs to know what variables to be passed to `foo`. 
This means compiler would try many type matching solutions, and reports failure once not found a matching one.

```cpp
struct X {
  typedef int type;
};

struct Y {
  typedef int type2;
};

template <typename T> void foo(typename T::type);    // Foo0
template <typename T> void foo(typename T::type2);   // Foo1
template <typename T> void foo(T);                   // Foo2

void callFoo() {
   foo<X>(5);    // Foo0: Succeed, Foo1: Failed,  Foo2: Failed
   foo<Y>(10);   // Foo0: Failed,  Foo1: Succeed, Foo2: Failed
   foo<int>(15); // Foo0: Failed,  Foo1: Failed,  Foo2: Succeed
   foo<float>(15.0); // All failed
}
```

## SFINAE Motivation

Define `Counter` inherited from `ICounter`; 

Define a function `inc_counter` that has two signatures: one to take `counterObj` then do `counterObj.increase();`; another one to take `int` and do `++intTypeCounter;`.

The template forms for the two signatures
`void inc_counter(T& intTypeCounter)` and `void inc_counter(T& counterObj)` are identical in the view of compiler, and a `redefinition error` is raised.

However, there should be a way to let compiler distinguish between `void inc_counter(T& intTypeCounter)` and `void inc_counter(T& counterObj)`.

```cpp
struct ICounter {
  virtual void increase() = 0;
  virtual ~ICounter() {}
};

struct Counter: public ICounter {
   void increase() override {
      // Implements
   }
};

template <typename T>
void inc_counter(T& counterObj) {
  counterObj.increase();
}

template <typename T>
void inc_counter(T& intTypeCounter){
  ++intTypeCounter;
}

void doSomething() {
  Counter cntObj;
  uint32_t cntUI32;

  // wrong ! compiler does not discriminate between the two function signatures
  inc_counter(cntObj);
  inc_counter(cntUI32);
}
```

## SFINAE Remediations

### With `std::decay`

`std::decay` applies lvalue-to-rvalue, array-to-pointer, and function-to-pointer implicit conversions to the type `T`, removes cv-qualifiers, and defines the resulting type as the member typedef type. 

```cpp
struct Counter {
   void increase() {
      // Implements
   }
};

template <typename T>
void inc_counter(T& intTypeCounter, std::decay_t<decltype(++intTypeCounter)>* = nullptr) {
  ++intTypeCounter;
}

template <typename T>
void inc_counter(T& counterObj, std::decay_t<decltype(counterObj.increase())>* = nullptr) {
  counterObj.increase();
}

void doSomething() {
  Counter cntObj;
  uint32_t cntUI32;

  // blah blah blah
  inc_counter(cntObj);
  inc_counter(cntUI32);
}
```

### With `std::enable_if`

`std::enable_if<T>` given the below definition, can be used to modifying function signatures conditional on the input to the boolean evaluation result of `T`.

```cpp
template< bool B, class T = void >
struct enable_if;
```
If `B` is true, `std::enable_if` has a public member typedef type, equal to `T`; otherwise, there is no member typedef.

In the example below, if `std::is_integral<T>::value` evaluates to true, `inc_counter` becomes `void inc_counter<int>(int & counterInt, void* dummy = nullptr)`.

If `std::is_integral<T>::value` evaluates to false, the expression evaluated to substitution failure. This just means this substitution match fails. If there exists another match, compiler still succeeds in compilation. 

```cpp
template <typename T> void inc_counter(
  T& counterObj, 
  typename std::enable_if<
    is_base_of<T, ICounter>::value
  >::type* = nullptr );

template <typename T> void inc_counter(
  T& counterInt,
  typename std::enable_if<
    std::is_integral<T>::value
  >::type* = nullptr );
```

## Concept (Supported since c++20)

`concept` is named Boolean predicates on template parameters, evaluated at compile time.

```cpp
struct Counter {
   void increase() {
      // Implements
   }
};

// adds constraints to template, it must be an int 
template <class T>
concept SignedIntegralT = std::is_integral_v<T> && std::is_signed_v<T>;

template <SignedIntegralT T>
void inc_counter(T& intTypeCounter) {
  ++intTypeCounter;
}

template <typename T>
void inc_counter(T& counterObj) {
  counterObj.increase();
}

void doSomething() {
  Counter cntObj;
  uint32_t cntUI32;

  // blah blah blah
  inc_counter(cntObj);
  inc_counter(cntUI32);
}
```