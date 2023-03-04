# STL Algo

## Algorithms

### `std::all_of`, `std::any_of` and `std::none_of`

`std::all_of`, `std::any_of` and `std::none_of` can used to compute an expression over a range of vars. 
```cpp
std::vector<int> v(10, 2);

if (std::all_of(v.cbegin(), v.cend(), [](int i){ return i % 2 == 0; })) {
    std::cout << "All numbers are even\n";
}

if (std::none_of(v.cbegin(), v.cend(), std::bind(std::modulus<>(), 
                                                    std::placeholders::_1, 2))) {
    std::cout << "None of them are odd\n";
}
```

* `std::all_of`
```cpp
template< class InputIt, class UnaryPredicate >
constexpr bool all_of(InputIt first, InputIt last, UnaryPredicate p)
{
    return std::find_if_not(first, last, p) == last;
}
```

* `std::any_of`
```cpp
template< class InputIt, class UnaryPredicate >
constexpr bool any_of(InputIt first, InputIt last, UnaryPredicate p)
{
    return std::find_if(first, last, p) != last;
}
```

* `std::none_of`
```cpp
template< class InputIt, class UnaryPredicate >
constexpr bool none_of(InputIt first, InputIt last, UnaryPredicate p)
{
    return std::find_if(first, last, p) == last;
}
```

### `std::erase_if` (since C++20) and `std::remove_if`



## Tools

### `std::sort()`

The algorithm used by `sort()` is *IntroSort*. Introsort being a hybrid sorting algorithm uses three sorting algorithm to minimize the running time, *Quicksort*, *Heapsort* and *Insertion Sort*. 
