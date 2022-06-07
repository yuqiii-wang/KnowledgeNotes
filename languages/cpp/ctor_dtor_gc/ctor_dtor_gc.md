# Garbage Collection, Constructor and Destructor

## `new`

* It allocates enough memory to hold an object of the type requested.
    
* It calls a constructor to initialize an object in the memory that was allocated.

### `operator new`

* Syntax: `void * operator new (size_t size);`
```
	void *pRawMemory = operator new(sizeof(string));
```

* The operator new returns a pointer to a chunk of memory enough to hole a string object.
* The operator new is similar to `malloc` in that it's responsible only for allocating memory. 

## Constructor

After `new` allocating memory, class members are not init (similar to `malloc(sizeof(ClassName))`, need to call class members' constructor, init values, etc).

Constructor is used to initialize some valid values to the data members of an object.

Constructors are also used to locate memory at run time using the new operator.

### No virtual constructor

The virtual mechanism works only when we have a base class pointer to a derived class object. In C++, constructor cannot be virtual, because when constructor of a class is executed there is no virtual table in the memory, means no virtual pointer defined yet.

### Constructors during Inheritance

When having inheritance, derived class must call base class's constructor, default `base()` if not specified. `derived () : base (number) {}` is used to explicitly invoke user defined base constructor.
```cpp
class base {
  public:
  base (int arg) {}
};

class derived : public base {
  public:
  derived () : base (number) {}
};
```

### Virtual base classes

If one base class is inherited from multiple parents, the base constructor is called multiple times as well. 

![virtual_base_class](imgs/virtual_base_class.png "virtual_base_class")

For example, `GrandParent` is called twice given the below definitions:
```cpp
class Parent1 : public GrandParent {...};
class Parent2 : public GrandParent {...};
```

Compiler throws Ambiguity error for same class member names declared across the chain of inheritance. 

To resolve this ambiguity when class `GrandParent` is inherited in both class `Parent1` and class `Parent2`, use 
```cpp
class Parent1 : virtual public GrandParent
class Parent2 : virtual public GrandParent
```

```cpp
#include <iostream>
using namespace std;

class GrandParent {
public:
	GrandParent(){
		cout << "Inside very first base class" << endl;
	}
};
class Parent1 : virtual public GrandParent{
public:
	Parent1(){
		cout << "Inside first base class" << endl;
	}
};
class Parent2 : virtual public GrandParent{
public:
	Parent2() {
		cout << "Inside second base class" << endl;
	}
};
class Child : public Parent1, public Parent2 {
public:
	Child(){
		cout << "Inside child class" << endl;
	}
};

int main() {
	Child child1;
	return 0;
}  
```

## Destructor

A `destructor` is a member function that is invoked automatically when the object goes out of scope or is explicitly destroyed by a call to delete. 

1. The class's destructor is called, and the body of the destructor function is executed.

2. Destructors for non-static member objects are called in the reverse order in which they appear in the class declaration. The optional member initialization list used in construction of these members does not affect the order of construction or destruction.

3. Destructors for non-virtual base classes are called in the reverse order of declaration.

4. Destructors for virtual base classes are called in the reverse order of declaration.

### Virtual Destructor

When a pointer object of the base class is deleted that points to the derived class, only the parent class destructor is called due to the early bind by the compiler. 
In this way, it skips calling the derived class' destructor, which leads to memory leaks issue in the program. 
And when we use virtual keyword preceded by the destructor tilde (~) sign inside the base class, it guarantees that first the derived class' destructor is called. Then the base class' destructor is called to release the space occupied by both destructors in the inheritance class.

So that, we need to add `virtual` to base class destructor:
```cpp
virtual ~base(){};
```

### `delete` vs `delete[]`

`delete[]` calls each element's destructor (synonym: array version of a single `delete`)