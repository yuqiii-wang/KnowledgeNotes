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

### The `__new__` Method

When you create an instance of a class, Python first calls the `__new__()` method to create the object and then calls the `__init__()` method to initialize the object's attributes.

The `__new__()` is a static method of the object class:

```py
object.__new__(class, *args, **kwargs)
```

When you define a new class, that class implicitly inherits from the `object` class. It means that you can override the `__new__` static method and do something before and after creating a new instance of the class.

Instead, should run by `asyncio.run(helloWorld())` that prints `"Hello World"`.

### Metaclass Example

A good use of metaclass is to create a singleton that is a common design for a DB connection pool.

```py
import threading

class SingletonMeta(type):
    _instance = None
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__call__(*args, **kwargs)
        return cls._instance

class DBConnectionPool(metaclass=SingletonMeta):
    def __init__(self, host_port_list: list, maxsize=3, timeout=60):
        ...
```

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

* `@abstractmethod` and `ABC`

`ABC` stands for Abstract Base Class, worked with `@abstractmethod` to define abstract class where some methods must be implemented in inherited classes.

`@abstractmethod` is similar to `virtual` in C++.

For example, below code explains how inheritance works with MUST-to-implement `__call__` as declared abstract in `class AuthHandler(ABC)`.

```py
from abc import ABC, abstractmethod

class AuthHandler(ABC):
    async def __call__(self, request: Request) -> User:
        """AuthHandler"""

class ReadOnlyAuthHandler(AuthHandler):
    async def __call__(self, request: Request) -> User:
        sub = request.state.username == "read_only"
        user, _ = await storage.get_read_only_user()
        return user

class BasicWriteAuthHandler(AuthHandler):
    async def __call__(self, request: Request) -> User:
        sub = request.state.username == "basic_write"
        user, _ = await storage.get_basic_write_user()
        return user

class SudoAuthHandler(AuthHandler):
    async def __call__(self, request: Request) -> User:
        sub = request.state.username == "sudo"
        user, _ = await storage.get_sudo_user()
        return user
```

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

* `@lru_cache`

`@lru_cache` maintains a global `dict` to store last function return values, and the subsequent calls with the same arguments are faster.

For example, `get_auth_cred` can be facilitated in retrieving credential.

```py
@lru_cache(maxsize=1)
def get_auth_cred(auth_type: str) -> AuthCredential:
    auth_cred = None
    if auth_type == "jwt_local":
        auth_cred =  get_jwt_local()
    elif auth_type == "jwt_local":
        auth_cred =  get_jwt_oidc()
    return auth_cred
```

## Where to `import`

For example, `import _ctypes` runs on python3.11.

1. `importlib`

On `import _ctypes`, python calls `importlib.util.find_spec('_ctypes')` to locate the module.
It searches for a file matching `_ctypes` in the directories listed in `sys.path`, including shared libraries like `_ctypes.cpython-311-x86_64-linux-gnu.so`.

2. Load `.so`

On Linux, Python uses `dlopen()` from `libdl` to load the `.so` file, e.g.,

```cpp
void* handle = dlopen("_ctypes.cpython-311-x86_64-linux-gnu.so", RTLD_NOW);
```

If `dlopen()` fails (e.g., due to missing dependencies), Python raises an `ImportError`.

Python locates the symbol using this from the found `.so` file.

```cpp
dlsym(handle, "PyInit__ctypes");
```

3. Init

Python calls `PyInit__ctypes()`, inside which `PyModule_Create()` creates module object, and that is created and returned to Python.

4. Integrate the Module with Python

The returned PyObject* module is added to sys.modules.
Python makes the module's attributes (functions, classes) available to the script.

### Smart Imports with `TYPE_CHECKING`

`TYPE_CHECKING` is a constant provided by `typing` that `TYPE_CHECKING` is always `False` at runtime but evaluates to `True` during typing checking.

Type checking happens before runtime to validate type annotations such as when a user write code in IDE that IDE will perform type analysis.

#### To Prevent Circular Imports

When two modules depend on each other, importing them directly can cause circular import issues.

Use `TYPE_CHECKING` to address it.
For example, below code
(Python allows type hints to reference classes or types that maynot yet be defined or imported as the time the function is defined. wrap the type in string to defer its evaluation until later)
shows two files importing each other but no circular import error raised.

```py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from module_b import ClassB

class ClassA:
    def __init__(self, b:"ClassB"):
        self.b = b
```

```py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from module_a import ClassA

class ClassB:
    def __init__(self, a:"ClassB"):
        self.a = a
```

#### Load Objects/Classes Only When Needed

For example, for `pandas` is a heavy module,
it is not loaded on program start.
It is not imported unless explicitly needed.

```py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import pd

def process_df(df: "pd.DataFrame") -> None:
    ...
```