# STL 

## Containers

vector、deque、stack/queue、map/set、unordered_map/unordered_set

### `std::map`

`std::map` uses red-black tree, that renders high computation cost when having too many insertion and deletion operation (to balance the tree). Read/write operations on the map is $O\big(log(n)\big)$.

### `std::string`



### `std::unordered_map`

`std::unordered_map` does not have order hence the read/write operations on the map is always $O(1)$.

Key/value pairs in `std::unordered_map` are stored in bucket depending on hash result of keys. There is no mandated implementations, that linked lists are used in the libstdc++ and Microsoft implementations, while other might use a number of vectors to represent buckets.

### `std::vector`

A vector grows exponentially, such as by $2^n$: $2$, $4$, $8$, $16$ ..., and every time it grows, there need copy operations that assigns elements from old memory to new memory addresses.

* `std::vector<bool>` 

It is an exception that `operator[]` returns `std::vector<bool>::reference` rather than a `bool`. Using `auto& elem = std::vector<bool>(1, false)[0]` to retrieve elements results in undefined behavior.

Consider use `auto highPriority = static_cast<bool>(std::vector<bool>(1, false)[0]);`

* `emplace_back` vs `push_back`

`emplace_back` is generally preferred as it does additional memory check.

`emplace_back` does type deduction so that it can be written with templates such as
```cpp
template<class T, class Allocator = allocator<T>>
class vector {
public:
    template <class... Args>
    void emplace_back(Args&&... args);
};
```
while `push_back` can only be used after instantiation (cannot be used in template)
```cpp
class vector<Widget, allocator<Widget>> {
public:
    void push_back(Widget&& x);
};
```

A better example:
```cpp
class MyKlass {
public:
  MyKlass(int ii_, float ff_) {...}

private:
};

void function() {
  std::vector<MyKlass> v;

  v.push_back(MyKlass(2, 3.14f));
  v.emplace_back(2, 3.14f);
}
```

By using `push_back`, it creates a temp obj every time:
1. Constructor for a temporary `MyKlass` object
2. Move constructor (if one was defined for `MyKlass`, otherwise a copy constructor) for the object actually allocated inside the vector
3. Destructor for the temporary

This is unnecessary since the object passed to push_back is obviously an rvalue that ceases to exist after the statement is completed.

By using `emplace_back`, only `MyKlass` constructor is called. This is the object constructed inside the vector. No temporaries are needed.

### `std::list`

`insert()`/`erase()` in a list need $O(n)$ since it iterates over the whole list to determine where to insert/delete an element.

`push_back()` only needs $O(1)$.

### `std::array`

`std::array` is a container that encapsulates fixed size arrays, init by such as 
```cpp
std::array<int, 3> a {1, 2, 3};
```

### `std::deque`

`std::deque` (double-ended queue) is an indexed sequence container that allows fast insertion and deletion at both its beginning and its end.

Deque implements pointer to the queue's first and end elements, while elements inside are chopped into chunks, each chunk is a vector, linked through a map:

![deque](imgs/deque.png "deque")

### `std::bitset`

The class template bitset represents a fixed-size sequence of N bits. Bitsets can be manipulated by standard logic operators and converted to and from strings and integers.

```cpp
template< std::size_t N >
class bitset;
```

### `std::set`

`std::set` is an associative container that contains a sorted set of unique objects of type Key. 

Sets are usually implemented as *red-black tree*s.

