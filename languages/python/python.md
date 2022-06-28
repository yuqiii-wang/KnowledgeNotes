# Python Notes

### `set` Usage

**`set` as key for `dict`**
```python
# frozenset does not discriminate elements in diff orders in a set
dct = {}
dct[frozenset([1,2,3])] = 'abc'
print(dct[frozenset([3,1,2])])
```

**`set` for set manipulation**
```python
set1 = {1,2}
set2 = {1,2,6}
list1 = [1,2,3,4]
set3 = set(list1)
print(set1.issubset(set3)) # subset
print(set1 | set3) # union (set or)
print(set1 & set3) # intersection (set and)
print(set1 - set2) # complementary set
```
**Diff between `set` and `tuple`**

In python, elements in `set` are unique while elements in `tuple` are immutable.

### Deepcopy

Should not just use `=` for by-value assignment in python, instead, by
```python
import copy

a = [[1,2,3], [2,3,4]]
b = copy.deepcopy(a)
```

### Variadic Arguments

```python
# *data is de-referenced to of tuple
def inputList(*data):
    print(data)
    return [x for x in data]

print(inputList(1,2,3))
```

### Dynamic Function and Function Handle

In general, a callable is something that can be called. This built-in method in Python checks and returns True if the object passed appears to be callable, but may not be, otherwise False.
```py
def Foo():
    return "yes"
  
# an object is created of Foo()
let = Foo
print(callable(let)) # print: True
```

You can pass a function handle as an arg to another function like this:
```py
def Foo(x, *args, **kwargs):
    if 'x' in kwargs:
        x = kwargs["x"]
    return x

def Bar(func, *args, **kwargs):
    func(*args, **kwargs)

Bar(Foo, x="1") # pass Foo as a func handle with kwargs["x"]
```