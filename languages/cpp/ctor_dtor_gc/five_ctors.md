# Constructor Types

Given a class, there should be at least one constructor (if not defined, compiler would create one for you).

### Default 

A default constructor does nothing when invoked, such as
```cpp
class A {
    A() = default;
};
```

### Copy Constructor

A copy constructor is used to init a new class by taking an existing class members and copying to the new class object.

```cpp
class A {
    A(const A& a){
        this->val1 = a.val1;
        this->val2 = a.val2;
        this->val3 = a.val3;
        // and more values being copied
    }
};

int main() {
    A a1;
    A a2 = a1; // copy constructor

    return 0;
}
```

### Copy Assignment

```cpp
class A {
    A& operator=(const A& a) {
        return *this;
    }
}

int main() {
    A a1, a2;
    a2 = a1; // copy assignment

    return 0;
}
```

### Move Constructor

Args passed with `std::move(a)`

```cpp
class A {
    A(A&& other) noexcept {}
};
```

### Move Assignment

Args passed with `std::move(a)`

```cpp
class A {
    A& operator=(A&& a) noexcept {
        return *this = A(a);
    }
};
```