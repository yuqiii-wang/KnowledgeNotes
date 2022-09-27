# Curiously Recurring Template Pattern (CRTP)

CRTP refers to a class `X` derives from a class template instantiation using `X` itself as a template argument.

Assume both `class Derived1` and `class Derived2` have the same methods `foo()` and `bar()`.
By the typical inheritance approach, during compilation, `class Derived1` and `class Derived2` add `vtable` to reference their virtual methods. This referencing work is unnecessary.
```cpp
class Base {
public:
  virtual void foo() = 0;
  virtual void bar() = 0;
};
​
class Derived1 : public Base {
public:
  virtual void foo() override final { cout << "Derived1 foo" << endl; }
  virtual void bar() override final { cout << "Derived1 bar" << endl; }
};

class Derived2 : public Base {
public:
  virtual void foo() override final { cout << "Derived2 foo" << endl; }
  virtual void bar() override final { cout << "Derived2 bar" << endl; }
};
```

In the CRTP approach, notice `static_cast<T *>(this)->internal_foo()` and `static_cast<T *>(this)->internal_bar()`. By `static_cast` to the derived class, `foo()` and `bar()` can have different implementation same as inheritance, but without the employment of `vtable`. In execution, user can just call `Derived1 d1; d1.foo();` that dereferences to using the `Derived1->internal_foo()`
```cpp
template<typename T> 
class Base {
public:
  void foo() { static_cast<T *>(this)->internal_foo(); }
  void bar() { static_cast<T *>(this)->internal_bar(); }
};
​
class Derived1 : public Base<Derived1> {
public:
  void internal_foo() { cout << "Derived1 foo" << endl; }
  void internal_bar() { cout << "Derived1 bar" << endl; }
};

class Derived2 : public Base<Derived2> {
public:
  void internal_foo() { cout << "Derived2 foo" << endl; }
  void internal_bar() { cout << "Derived2 bar" << endl; }
};
```

When working on computation-intensive tasks such as having `vec1f`, `vec2f`, etc., that inherit from `vec_base`. Repeated referencing to the base class is time-consuming.