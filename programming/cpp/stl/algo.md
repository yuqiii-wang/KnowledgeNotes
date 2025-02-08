# STL Algo

## Algorithms


### `std::find_if`

```cpp
template <class InputIterator, class UnaryPredicate>   
InputIterator find_if (InputIterator first, InputIterator last, UnaryPredicate pred);
```

Returns an iterator to the first element in the range `[first,last)` for which `pred` returns `true`; if not found, return `last`.

For example, below code defines `pred` as a lambda that iterates the whole `features` to see if there is matched feature id.
The return `auto it` is the iterator pointing to `FeaturePerId`.
```cpp
auto it = find_if(features.begin(), features.end(), 
                    [to_be_matched_feature_id](const FeaturePerId &it)
                    {
                        return it.feature_id == to_be_matched_feature_id;
                    });
```

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
