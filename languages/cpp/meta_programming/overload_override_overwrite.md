# overload, override, overwrite

## Overload

Same function name with different arguments

On the same scope/class

## Override

Different scope (parent/children during inheritance)

Same function name and arguments

Base/parent class must declare `virtual`

## Overwrite

It refers to re-definition of a variable/function in the derived/children class.

Since base/parent class and derived/children class have separate variable/function scopes if without declared `virtual`, the same function in derived/children class (same function name and argument as its base/parent counterpart's) appears to "overwrite" its base/parent function (in other words, the base/parent class variable/function is hidden/not accessible from its derived/children class), as observed in code
```cpp
class Base {
public:
    int x=0;
};
class Derived: public Base {
public:
    int x=0;
};

void main(){
    Derived d;
    d.x = 10; // only Derived's x is changed,
              // Base's x is not accessible from d  
}
```   