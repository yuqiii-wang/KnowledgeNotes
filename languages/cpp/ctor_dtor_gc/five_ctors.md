# Constructor Types

Given a class, there should be at least one constructor (if not defined, compiler would create one for you).

### Default 

A default constructor does nothing when invoked, such as
```cpp
class A {
    A() = default;
};
```

Invoked by `A a;`

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

A f(A a) {
    return a;
}

int main() {
    A a1 = f(A());
    A a2 = std::move(a1);

    return 0;
}
```

### Move Assignment

Args passed with `std::move(a)`

```cpp
class A {
    A& operator=(A&& a) noexcept {
        return *this = A(a);
    }
};

int main()
{
    A a1, a2;
    a2 = std::move(a1);
}
```

## Move Constructor vs Copy Constructor

Basically, the diff is about whether the member fields of an object are *copied* or *moved* into the new object.

* Copy

Copy constructor must allocate its own copy of object's data for itself. 

* Move

Move constructor takes ownership of pointer that refers to data.

Move semantics are most commonly used with pointers/handles to dynamic resources by transferring ownership of pointers.

Move semantics **doesn't help** improve efficiency is when the data being "moved" is **POD** data (plain old data, i.e., integers, floating-point decimals, booleans, structure/array aggregates, etc).
"Moving" such data is the same as "copying" it (should make POD a pointer, then move by pointer).

### Example

`MyIntArray(int size)` creates `size` number of int stored in `int *arr`.

Copy works by allocating a new memory block to store data, while move works by swapping pointers.

```cpp
class MyIntArray
{
private:
    int *arr = nullptr;
    int size = 0;
public:
    MyIntArray() = default;

    MyIntArray(int size) {
        arr = new int[size];
        this->size = size;
        for(int i = 0; i < size; ++i) {
            arr[i] = i;
        }
    }

    // copy constructor
    MyIntArray(const MyIntArray &src) {
        // allocate a new copy of the array...
        arr = new int[src.size];
        size = src.size;
        for(int i = 0; i < src.size; ++i) {
            arr[i] = src.arr[i];
        }
    }

    // move constructor
    MyIntArray(MyIntArray &&src) {
        // just swap the array pointers...
        src.swap(*this);
    }

    ~MyIntArray() {
        delete[] arr;
    }

    // copy assignment operator
    MyIntArray& operator=(const MyIntArray &rhs) {
        if (&rhs != this) {
            MyIntArray temp(rhs); // copies the array
            temp.swap(*this);
        }
        return *this;
    }

    // move assignment operator
    MyIntArray& operator=(MyIntArray &&rhs) {
        MyIntArray temp(std::PODmove(rhs)); // moves the array
        temp.swap(*this);
        return *this;
    }

    /*
    or, the above 2 operators can be implemented as 1 operator, like below.
    This allows the caller to decide whether to construct the rhs parameter
    using its copy constructor or move constructor...

    MyIntArray& operator=(MyIntArray rhs) {
        rhs.swap(*this);
        return *this;
    }
    */

    void swap(MyIntArray &other) {
        // swap the array pointers...
        std::swap(arr, other.arr);
        std::swap(size, other.size);
    }
};
```