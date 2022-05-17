# Virtual functions

A virtual function is unknown at compile time and is defined at run time.

A normal (opposed to pure virtual) virtual function can be defined in a base/parent class while derived class can override it.

Compiler adds a virtual pointer (`VPTR`) and a virtual table (`VTABLE`) to a *class* (not to an object) when found it has virtual functions:
1. If object of that class is created then a virtual pointer (`VPTR`) is inserted as a data member of the class to point to `VTABLE` of that class. For each new object created, a new virtual pointer is inserted as a data member of that class.
2. Irrespective of object is created or not, class contains as a member a static array of function pointers called `VTABLE`. Cells of this table store the address of each virtual function contained in that class.

## Pure virtual

A virtual function that is required to be implemented by a derived class if the derived class is not abstract. The derived class must define the virtual function.

A pure virtual function is defined with declaration `=0` such as `virtual void f() = 0;`.

## Virtual destructor

Deleting a derived class object using a pointer of base class type that has a non-virtual destructor results in undefined behavior, requiring a defined virtual destructor as a resolution to this issue.

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

## Virtual method table

A virtual method table is implemented to map a base class virtual method to a derived class defined/override method at run time.

Typically, the compiler creates a separate virtual method table for each class. 
When an object is created, a pointer to this table, called the virtual table pointer, `vpointer` or `VPTR`, is added as a hidden member of this object. 
As such, the compiler must also generate "hidden" code in the constructors of each class to initialize a new object's virtual table pointer to the address of its class's virtual method table. 

Again given the above `base`/`derived` class example, compiler might augment/expand destructor source code to incoporate base class destructor code.

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