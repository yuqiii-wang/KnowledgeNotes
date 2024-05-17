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

## Python Object LifeCycle Management

Acquired resource must be released if no longer used.
Some typical concerns are DB connection sessions are limited, and when DB query ends, need to `close()` DB connection.

* Use `with` statement

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

# use weak reference to solve circular dependency
import weakref
a.field = weakref.ref(b)
b.field = weakref.ref(a)
```
