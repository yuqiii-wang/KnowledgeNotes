# Python Software Development

## Python Software Setup and Build

* Python Egg

The `.egg` file is a distribution format for Python packages same concept as a `.jar` to java or Windows `.exe`.
`.egg` file is completely cross-platform.

Essentially, it is a `.zip` with some metadata.

* `setup.py`

`setup.py` is the python version of Java's `POM.xml` and c++'s `CMakeLists.txt`.

The python project setup is run by `python setup.py install`.

* `__init__.py`

Python defines two types of packages, regular packages (Python 3.2 and earlier) and namespace packages.

A regular package is typically implemented as a directory containing an `__init__.py` that is implicitly executed used to distinctly identify objects in a package's namespace.

* Jinja2

*Jinja2* is a web template engine that combines a template with a certain data source to render dynamic web pages.

* The `python-config`

`python-config` is a command-line utility that provides information about the configuration of a Python installation.
The primary purpose of python-config is to make it easier for developers to compile and link programs or extensions that interact with Python.

Python uses it to locate compiled c++ include and libs folders.
Error might occur for incompatible libs vs python executable binary.

Check by

```sh
python-config --includes
python-config --libs

python --version
```

* Module not found

For example, `_ctypes` module is not found.

Check sys paths:

```sh
python -c "import sys; print(sys.path)"
```

Check where is `_ctypes.cpython-3xx-x86_64-linux-gnu.so` lib

```sh
python -c "import importlib.util; spec=importlib.util.find_spec('_ctypes'); print(spec)"
```

Likely it is in `path/to/python/lib/python3.xx/lib-dynload/`

## Coroutine and Asyncio

Python have two coroutine implementations:

* `yield` + `generator`
* `asyncio`

### Coroutine and Generator

A coroutine in python is implemented by `yield` that returns a `generator` (by this time execution is paused and contexts are stored/stacked), then once reached `next`/`send`, execution resumes.

```py
def my_gen_coroutine():
    print("Coroutine started.")
    while True:
      received = yield # pause here
      print(f"Coroutine received msg: {received}.")

my_coro = my_gen_coroutine()
# print "Coroutine started."

result = next(my_coro)
# or
result = my_coro.send("Coroutine yielded")
# print "Coroutine received msg: Coroutine yielded."

my_coro.close()
```

### Asyncio

`asyncio` is a python library providing async concurrency functions implemented by coroutines + multiplexing I/O over socket.

`asyncio` uses `await` to conduct pauses and resumptions of execution, that are all managed in an asyncio event loop.

```py
import asyncio

async def my_async_coroutine():
    print("async coroutine started.")
    await asyncio.sleep(1) # execution pause and resumption
    print("async coroutine ended.")

my_async_coro = my_async_coroutine()

my_async_task = asyncio.create_task(my_async_coro)
# my_async_task is a coroutine object

# get my_async_task and run
asyncio.run(my_async_task)
# or (in older version python)
loop = asyncio.get_event_loop()
loop.run_until_complete(my_async_task)
```

`nest_asyncio` is used to run nested asyncio, because by default, asyncio does not support running an event inside another event loop.
This is useful, for example, in Jupyter notebook that runs an asyncio loop by default.

```py
import asyncio
import nest_asyncio

nest_asyncio.apply()

async def inner_coro():
    print("Started inner coroutine")
    await asyncio.sleep(1)
    print("Done inner coroutine")

async def outer_coro():
    print("Started outer coroutine")
    await asyncio.sleep(1)
    print("Started inner coroutine")
    await inner_coro()
    print("Done inner coroutine")

async def main():
    print("Started main coroutine")
    await outer_coro()
    print("Done main coroutine")

asyncio.run(main())
```

### Iterables vs Generators

`iterable`/`__iter__`: When you create a list, you can read its items one by one. Reading its items one by one is called iteration,

`generator` are iterators, a kind of iterable you can only iterate over once. Generators do not store all the values in memory, they generate the values on the fly.

### `@contextmanager` and Coroutine

`@contextmanager` annotation is to simulate a full class scope management with `with` (`__enter__` and `__exit__` methods).

Define a generator function, where a `yield` statement is executed when entering the context, and the code after the `yield` statement is executed when exiting the context.

```py
import psycopg2
from contextlib import contextmanager

@contextmanager
def postgresql_connection(dbname, user, password, host='localhost', port=5432):
    conn = None
    try:
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
        yield conn
    finally:
        if conn is not None:
            conn.close()

# Usage example
with postgresql_connection('mydatabase', 'myuser', 'mypassword') as conn:
    with conn.cursor() as cursor:
        cursor.execute('SELECT * FROM mytable')
        result = cursor.fetchall()
        print(result)
```

## Type Safe by `pydantic`

`pydantic` is a Python library that provides runtime type checking and data validation using Python type annotations.

* `Field`

```py
from pydantic import BaseModel, validator, Field
from Typing import Optional

class Person(BaseModel):
    name: str = Field(default="Anonymous", description="The name of the person")
    age: int = Field(default=18, gt=0, description="The age of the person, be greater than 0")
    sex: Optional[str] = Field(default="Unknown", description="The sex of the person, either 'm', 'f', or 'Unknown'")

    @validator("sex")
    def validate_sex(cls, value):
        if value not in ["m", "f", "Unknown"]:
            raise ValueError("Sex must be either 'm', 'f', or 'Unknown'")
        return value

class Student(Person):
    student_id: str = Field(description="The unique ID of the student")
    courses: List[str] = Field(default_factory=list, description="List of courses the student is enrolled in")

#################
# Create a Student instance and test it
student_data = {
    "name": "Alice",
    "age": 20,
    "sex": "f",
    "student_id": "S12345",
    "courses": ["Math", "Physics"]
}

student = Student(**student_data)
print(student.model_dump())
```

## Python Object LifeCycle Management

Acquired resource must be released if no longer used.
Some typical concerns are DB connection sessions are limited, and when DB query ends, need to `close()` DB connection.

* Use `with` statement (context manager)

For example, for postgresql, the connected session will be closed once exited the `with` scope.

```py
with psycopg2.connect("<db connection>") as connection:
    with connection.cursor() as cursor:
        cursor.execute("SELECT current_database();")
        query_result = cursor.fetch_one()
        print(query_result)
```

* `try`-`except`-`finally` statement

```py
try:
    conn = psycopg2.connect("<db connection>")
    cursor.execute("SELECT current_database();")
    query_result = cursor.fetch_one()
    print(query_result)
except Exception as err:
    print(f"DB err: {err}")
finally:
    conn.close()
```

* Use `__del__()` and manage by python garbage collector `gc.collect()`

When an object reference count goes to zero, the object is marked to be deleted.
Python garbage collector will pick up a time to actually release the object's resources (the exact timing is uncertain).
To immediately trigger garbage collector execution, use `gc.collect()`.

```py
import gc

class MyCls:
    def __init__(self):
        print("MyCls created")
    def __del__(self):
        print("MyCls ended")

# created
my_cls = MyCls()

# reference count to two
my_another_same_cls = my_cls

# `__del__()` will NOT be called for reference count is one
del my_cls

# `__del__()` will be called for reference count is zero
del my_another_same_cls

# However, the exact timing of `__del__()` execution is uncertain depending on 
# how garbage collector schedules executions
# to immediately trigger garbage collector execution, use `gc.collect()`
gc.collect()
```

Reference count fails when there is circular reference:

```py
class A:
    def __init__(self):
        self.field = None
class B:
    def __init__(self):
        self.field = None

a = A()
b = B()

# circular dependency that if there are
# `del a` and `del b`, their reference count will not go to zero.
a.field = b
b.field = a

# use weak reference to solve circular dependency,
# so that `del a` and `del b` will actually get their reference count to zeros.
import weakref
a.field = weakref.ref(b)
b.field = weakref.ref(a)
```

## Python Garbage Collection

Python has two strategies managing object de-allocation: *reference count* and *garbage collection*

### Reference count

The main garbage collection mechanism in CPython is through reference counts. Whenever you create an object in Python, the underlying C object has both a Python type (such as list, dict, or function) and a reference count.

When references to an object are removed, the reference count for an object is decremented. When the reference count becomes zero, the object is deallocated.

You can get a var's reference count by

```py
import sys
a = 'my-string'
b = [a] # Make a list with a as an element.
c = { 'key': a } # Create a dictionary with a as one of the values.
sys.getrefcount(a)

###
# result is 4
```

### GC module

#### Motivation: de-allocation failure

This is never freed since `x` has a reference to itself, and reference counter does not work on this scenario.

```py
x = []
x.append(x) # `x` has a reference to itself

del x # `x` is not longer accessable, but not freed
```

#### `gc` solution

`gc` module comes to rescue by checking the number of allocated objects and de-allocated objects (allocations and de-allocations mismatch demonstrates that there are garbages missed removed).

If the number is greater than a threshold, objects are scheduled being de-allocated. It detects items such as `del` and existence of objects in memory.

You can check the threshold by this.

```py
# loading gc
import gc
 
# get the current collection
# thresholds as a tuple
print("Garbage collection thresholds:",
                    gc.get_threshold())
```

### Manual trigger

You can manually trigger `gc` to start collecting objects, if the scheduled auto-recycling has not yet met the threshold.

```py
import gc
i = 0
 
# create a cycle and on each iteration x as a dictionary
# assigned to 1
def create_cycle():
    x = { }
    x[i+1] = x
    print x
 
# lists are cleared whenever a full collection or
# collection of the highest generation (2) is run
collected = gc.collect() # or gc.collect(2)
print "Garbage collector: collected %d objects." % (collected)
 
print "Creating cycles..."
for i in range(10):
    create_cycle()
 
collected = gc.collect()
 
print "Garbage collector: collected %d objects." % (collected)
```

which prints

```bash
Garbage collector: collected 0 objects.
Creating cycles...
{1: {...}}
{2: {...}}
{3: {...}}
{4: {...}}
{5: {...}}
{6: {...}}
{7: {...}}
{8: {...}}
{9: {...}}
{10: {...}}
Garbage collector: collected 10 objects.
```
