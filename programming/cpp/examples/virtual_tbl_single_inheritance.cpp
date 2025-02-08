#include <iostream>

// g++ -std=c++11 -g virtual_tbl_single_inheritance.cpp && gdb ./a.out
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

class BaseNoVirtual
{
public:
  void foo() {}
};

class Base {
 public:
  virtual void foo() {};
  virtual void fooNotOverridden() {};
};

class Derived : public Base {
 public:
  void foo() override {};
};

int main() {
  BaseNoVirtual bn;

  Base b1, b2;
  Derived d1, d2;

  std::cout << "Size of BaseNoVirtual: " << sizeof(BaseNoVirtual) << std::endl;
  std::cout << "Size of Base: " << sizeof(Base) << std::endl;
 
  // just used to make gbd easy to set break point
  breakPointLabel();
}