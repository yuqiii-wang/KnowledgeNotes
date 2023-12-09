# Python Notes

## pip

pip upgrade by tsinghua source

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
```

pip install by tsinghua source

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package_name>
```

### Manual Install by `setup.py`

From a local directory, find `setup.py`, and `cd` to this directory.

Run by `python setup.py install`.

## Python Interpreter

* Cython vs Jython

* Spawn vs Fork

Forking and spawning are two different start methods for new processes. Fork is the default on Linux (it isn’t available on Windows), while Windows and MacOS use spawn by default.

When a process is forked the child process inherits all the same variables in the same state as they were in the parent. Each child process then continues independently from the forking point. The pool divides the args between the children and they work though them sequentially.

On the other hand, when a process is spawned, it begins by starting a new Python interpreter. The current module is reimported and new versions of all the variables are created. 

## Builtin types

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

## Call by Reference vs by Value

* by value: immutable values such as immediate numbers, strings, tuples

```python
stringExample = "test"

def testStr(str: string):
    string = "test 2"
    print(string)

testStr(stringExample) # print "test 2"
print(stringExample)   # print "test", shows that the original string does not change
```

* by reference: mutable values such as lists and dict

```python
listExample = [1,2,3]

def testList(list: listExample):
    listExample.append(4)
    print(listExample)

testList(listExample) # print "[1, 2, 3, 4]"
print(listExample)   # print "[1, 2, 3, 4]", shows that the original list is updated.
```

To make a string pass-by-reference, a typical solution is wrapping the string in a list `[stringExample]`, then accessing/updating it by `[0] = [0] + " test"`.

### Object Bindings and Containers

For the same content such as two string objects containing the same chars, they are treated as containers to the source chars.

For example, below strings `a` and `b` have the same addr/id.

```python
a = "test"
b = "test"
 
# Returns the actual location 
# where the variable is stored
print(id(a)) # print "140285250931248"
print(id(b)) # print "140285250931248"
 
# Returns true if both the variables
# are stored in same location
print(a is b) # print "True"
```

Below lists `a` and `b` have diff addrs/ids.

```python
a = [1,2,3]
b = [1,2,3]
 
# Returns the actual location 
# where the variable is stored
print(id(a)) # print "140285250915200"
print(id(b)) # print "140285250930944"
 
# Returns true if both the variables
# are stored in same location
print(a is b) # print "False"
```