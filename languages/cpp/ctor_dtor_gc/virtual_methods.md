# Virtual functions

A virtual function is unknown at compile time and is defined at run time.

A normal (opposed to pure virtual) virtual function can be defined in a base/parent class while derived class can override it.

Compiler adds a virtual pointer (`VPTR`) and a virtual table (`VTABLE`) to a *class* (not to an object) when found it has virtual functions:
1. If object of that class is created then a virtual pointer (`VPTR`) is inserted as a data member of the class to point to `VTABLE` of that class. For each new object created, a new virtual pointer is inserted as a data member of that class.
2. Irrespective of object is created or not, class contains as a member a static array of function pointers called `VTABLE`. Cells of this table store the address of each virtual function contained in that class.

Given the virtual method materialization, there is a saying "Class member function invocation is determined at run time, not compile time", since the compiled code does not know what function to call in derived classes, but needs to look up the `VTABLE` to locate the member function.

## Pure virtual

A virtual function that is required to be implemented by a derived class if the derived class is not abstract. The derived class must define the virtual function.

A pure virtual function is defined with declaration `=0` such as `virtual void f() = 0;`.

## Virtual destructor

Deleting a derived class object using a pointer of base class type that has a non-virtual destructor results in undefined behavior (likely **derived class has memory leak**), requiring a defined virtual destructor as a resolution to this issue.

Once execution reaches the body of a base class destructor, any derived object parts have already been destroyed and no longer exist. If the Base destructor body were to call a virtual function, the virtual dispatch would reach no further down the inheritance hierarchy than Base itself. In a destructor (or constructor) body, further-derived classes just don't exist any more (or yet).

```cpp
#include <iostream>
using namespace std;
 
class base {
  public:
    base()    
    { cout << "Constructing base\n"; }
    ~base()
    { cout<< "Destructing base\n"; }    
};
 
class derived: public base {
  public:
    derived()    
     { cout << "Constructing derived\n"; }
    ~derived()
       { cout << "Destructing derived\n"; }
};
 
int main()
{
  derived *d = new derived(); 
  base *b = d;
  delete b;
  sleep(1);
  return 0;
}
```
which outputs
```bash
Constructing base
Constructing derived
Destructing base
```
in which `derived` destructor is not called, resulting in resource leak. Instead, destructors should have been virtual such as 
```cpp
virtual ~base(){ cout << "Destructing base\n"; }
virtual ~derived(){ cout << "Destructing derived\n"; }
```

## Virtual Constructor

Constructor cannot be virtual, because when a constructor of a class is executed there is no virtual table in the memory, means no virtual pointer defined yet.

## Virtual method table

A virtual method table is implemented to map a base class virtual method to a derived class defined/override method at run time. 
Compiler assigns a virtual method table to each class that has a virtual method or inherits from a base class with virtual methods.

Typically, the compiler creates a separate virtual method table for each class (**per class**). 
When an object is created, a pointer to this table, called the virtual table pointer, `vpointer` or `VPTR`, is added as a hidden member of this object. 
As such, the compiler must also generate "hidden" code in the constructors of each class to initialize a new object's virtual table pointer to the address of its class's virtual method table. 

Again given the above `base`/`derived` class example, compiler might augment/expand destructor source code to incorporate base class destructor code.

```cpp
derived::~derived(){
// source drived destructor code
cout << "Destructing derived\n";

// Compiler augmented code, Rewire virtual table
this->vptr = vtable_base; // vtable_base = address of static virtual table

// Call to base class destructor
base::~base(this); 
}
```

For this reason (a derived class destructor might invoke its base destructor), if base class destructor is pure virtual, such as
```cpp
virtual ~base()=0;
```

compiler might throw linker error since compiler could not resolve `base::~base()`
```txt
main.cpp:(.text._ZN7derivedD2Ev[_ZN7derivedD2Ev]+0x11): undefined reference to `base::~base()'
clang: error: linker command failed with exit code 1 (use -v to see invocation)
```

## Virtual Table Types

A virtual table is built at compile time, but it is called/looked up at runtime to determine polymorphism.

Many compilers place the virtual table pointer as the last member of the object; 
other compilers place it as the first; portable source code works either way.[2] For example, g++ previously placed the pointer at the end of the object.

### Category 0: Trivial
Structure:
* No virtual base classes.
* No virtual functions.

Such a class has no associated virtual table, and an object of such a class contains no virtual pointer.

### Category 1: Leaf
Structure:
* No inherited virtual functions.
* No virtual base classes.
* Declares virtual functions.

The virtual table contains offset-to-top and RTTI fields followed by virtual function pointers. 
There is one function pointer entry for each virtual function declared in the class, in declaration order, with any implicitly-defined virtual destructor pair last.

### Category 2: Non-Virtual Bases Only

Structure:
* Only non-virtual proper base classes.
* Inherits virtual functions.

The class has a virtual table for each proper base class that has a virtual table. 

### Category 3: Virtual Bases Only

Structure:
* Only virtual base classes (but those may have non-virtual bases).
* The virtual base classes are neither empty nor nearly empty.

The class has a virtual table for each virtual base class that has a virtual table.
In other words, it has one virtual table and another one `vtt` manages virtual table 

## Virtual tables During Object Construction

During the construction of a class object, the object assumes the type of each of its proper base classes, as each base class subobject is constructed. 
RTTI queries in the base class constructor will return the type of the base class, and virtual calls will resolve to member functions of the base class rather than the complete class. 