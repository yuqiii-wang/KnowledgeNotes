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

### `yield` usage

`iterable`: When you create a list, you can read its items one by one. Reading its items one by one is called iteration,

`generators` are iterators, a kind of iterable you can only iterate over once. Generators do not store all the values in memory, they generate the values on the fly.
```py
mygenerator = (x*x for x in range(3))
for i in mygenerator:
    print(i)
# print results: 0, 1, 4

for i in mygenerator:
    print(i)
# print results: 0, 0, 0
```

`yield` is a keyword that is used like return, except the function will return a generator.

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