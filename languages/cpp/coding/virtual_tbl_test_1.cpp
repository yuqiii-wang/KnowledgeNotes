#include <iostream>

// g++ virtual_tbl_test_1.cpp -std=c++14 && gdb ./a.out

class BaseNoVirtual
{
public:
  void foo() {};
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
  Base b1, b2;
  Derived d1, d2;

  std::cout << "Size of BaseNoVirtual: " << sizeof(BaseNoVirtual) << std::endl;
  std::cout << "Size of Base: " << sizeof(Base) << std::endl;
 
  std::cout << "done" << std::endl;
}