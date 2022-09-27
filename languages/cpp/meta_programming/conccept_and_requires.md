# Concept and Requires (since C++20)

## Requirements

`requires` yields a prvalue expression of type bool that describes the constraints.



## Concept
A concept is a named set of requirements. The definition of a concept must appear at namespace scope.

```cpp
template < template-parameter-list >
concept concept-name = constraint-expression
```