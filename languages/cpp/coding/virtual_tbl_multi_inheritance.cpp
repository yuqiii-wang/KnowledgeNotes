#include <iostream>

// g++ -std=c++11 -g virtual_tbl_multi_inheritance.cpp && gdb ./a.out
/*
in gdb:
  set demangle-style gnu
  set print asm-demangle on
  set print demangle on
  b breakPointLabel
  run
  n
*/ 

void breakPointLabel(){ int b = 0; }

class Base1 {
public:
  virtual void Base1Func() {} ;
  int base1Val;
};

class Base2 {
public:
  virtual void Base2Func() {} ;
  int base2Val;
};

class Derived : public Base1, public Base2 {
public:
  virtual void DerivedFunc() {} ;
  int derivedVal;
};

int main() {

  Derived d;

  // just used to make gbd easy to set break point
  breakPointLabel();
}