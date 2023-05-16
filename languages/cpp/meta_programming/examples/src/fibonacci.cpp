#include <iostream>

// find fibonacci using template meta-programming
template <long long N>
struct fibonacci{
  static constexpr long long value = fibonacci<N-1>::value + fibonacci<N-2>::value;
};

template<>
struct fibonacci<1>{
  static constexpr long long value = 1;
};

template<>
struct fibonacci<0>{
  static constexpr long long value = 0;
};

int main()
{
    auto result = fibonacci<10>().value;
    std::cout << result << std::endl;
    return 0;
}