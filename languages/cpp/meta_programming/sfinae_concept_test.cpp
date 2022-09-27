// Compile with `-std=c++20`

#include <iostream>
#include <iomanip>
#include <type_traits>

struct Counter {
   int _x = 0; 
   void increase() {
      _x++;
   }
};

// adds constraints to template, it must be a signed int 
template <class T>
concept SignedIntegralT = std::is_integral_v<T> && std::is_signed_v<T>;

template <SignedIntegralT T>
void inc_counter(T& intTypeCounter) {
  ++intTypeCounter;
}

template <typename T>
void inc_counter(T& counterObj) {
  counterObj.increase();
}

int main() {
  Counter cntObj;
  int cntUI32;

  // blah blah blah
  inc_counter(cntObj);
  inc_counter(cntUI32);

  return 0;
}