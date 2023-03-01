#include <iostream>

struct C {
  C() { std::cout << "A default construct was made.\n"; }
  C(const C&) noexcept { std::cout << "A copy was made.\n"; }
  C(const C&&) noexcept { std::cout << "A move was made.\n"; };
};
 
C f() {
  return C(); //Definitely performs copy elision
}

C g() {
    C c;
    return c; //Maybe performs copy elision
}

C h() {
    C c;
    return std::move(c); // Explicitly move
}

int main() {
  C obj = f(); 
  std::cout << "====\n";
  C obj2 = g();
  std::cout << "====\n";
  C obj3 = h();
}