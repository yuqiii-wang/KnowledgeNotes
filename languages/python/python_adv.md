# Python Advanced Usage/Knowledge

## `__new__` and Metaclass

A metaclass is a class whose instances are classes. 

In python, the builtin class `type` is a metaclass.

Given the code below, at run time, `Car` itself is an instance of `type`, despite not explicitly invoking `type`.

```py
class Car:
    def __init__(self, make: str, model: str, year: int, color: str):
        self.make = make
        self.model = model
        self.year = year
        self.color = color

    @property
    def description(self) -> str:
        """Return a description of this car."""
        return f"{self.color} {self.make} {self.model}"

# To create a car
new_car = Car(make='Toyota', model='Prius', year=2005, color='Green', engine='Hybrid')
```

The attribute settings such as `make`, `model`, etc. can be set by a custom metaclass `AttributeInitType` that inherits from `type`. `Car` can be constructed same as before.

```py
class AttributeInitType(type):
    def __call__(self, *args, **kwargs):
        """Create a new instance."""

        # First, create the object in the normal default way.
        obj = type.__call__(self, *args)

        # Additionally, set attributes on the new object.
        for name, value in kwargs.items():
            setattr(obj, name, value)

        # Return the new object.
        return obj

class Car(object, metaclass=AttributeInitType):
    @property
    def description(self) -> str:
        """Return a description of this car."""
        return " ".join(str(value) for value in self.__dict__.values())

# Create a car same as before
new_car = Car(make='Toyota', model='Prius', year=2005, color='Green', engine='Hybrid')
```

### `__new__`

When you create an instance of a class, Python first calls the `__new__()` method to create the object and then calls the `__init__()` method to initialize the object’s attributes.

The `__new__()` is a static method of the object class:

```py
object.__new__(class, *args, **kwargs)
```

When you define a new class, that class implicitly inherits from the `object` class. It means that you can override the `__new__` static method and do something before and after creating a new instance of the class.

Instead, should run by `asyncio.run(helloWorld())` that prints `"Hello World"`.

## Decorator

Decorator is used to "wrap" functions to perform helper services.
For example, `calculate_time(func)` is implemented to audit function elapsed time.

```py
import time
import math

def calculate_time(func):
    def audit_time(*args, **kwargs):
        begin = time.time()
        func(*args, **kwargs)
        end = time.time()
        print("Total time taken in : ", func.__name__, end - begin)
    return audit_time

@calculate_time
def factorial(num):
    time.sleep(1)
    print(math.factorial(num))
 
factorial(10)
```

Decorators can help in many scenarios such as below.

* `@classmethod`

Similar to `static` in C++.

* `@abstractmethod`

Similar to `virtual` in C++.

* `@contextmanager`

Used to define a factory function for `with` statement context managers, in other words, no need of declaring `__enter__()` and `__exit__()`.

For example,

```python
from contextlib import contextmanager

@contextmanager
def managed_resource(*args, **kwds):
    # Code to acquire resource, e.g.:
    resource = acquire_resource(*args, **kwds)
    try:
        yield resource
    finally:
        # Code to release resource, e.g.:
        release_resource(resource)

# managed_resource(...) can be used within `with`
# resource will be released after exiting `with`
with managed_resource(timeout=3600) as resource:
    ...
```

* `@property`

`@property` is used to easily implement `getter()`, `setter()`, and `delete()`.

A typical implementation of `@property` is

```py
lass Property:
    "Emulate PyProperty_Type() in Objects/descrobject.c"

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        if doc is None and fget is not None:
            doc = fget.__doc__
        self.__doc__ = doc

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(obj, value)

    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        self.fdel(obj)

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)
```

`@property` can be used in this scenario, where `temperature` has default `getter()`, `setter()`, and `delete()`, and it allows customization such as in `@temperature.setter`.

```py
class Celsius:
    def __init__(self, temperature = 0):
        self.temperature = temperature

    def to_fahrenheit(self):
        return (self.temperature * 1.8) + 32

    @property
    def temperature(self):
        print("Getting value")
        return self.temperature

    @temperature.setter
    def temperature(self, value):
        if value < -273:
            raise ValueError("Temperature below -273 is not possible")
        print("Setting value")
        self.temperature = value
```
