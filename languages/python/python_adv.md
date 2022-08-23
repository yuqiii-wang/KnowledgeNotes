# Python Advanced Usage/Knowledge

## GIL (Global Interpreter Lock)

A global interpreter lock (GIL) is a mechanism used in computer-language interpreters to synchronize the execution of threads so that only one native thread can execute at a time, even on multi-core processor, hence rendering low concurrency code execution.

Consider the code
```py
import time
from threading import Thread
from multiprocessing import Pool
  
def countdown(n):
    while n>0:
        n -= 1

# one thread running countdown
COUNT = 200000000
t0 = Thread(target = countdown, args =(COUNT, ))

start = time.time()
t0.start()
t0.join()
end = time.time()
print('Time taken (one thread) in seconds:', end - start)

# four threads running countdown
t1 = Thread(target = countdown, args =(COUNT//4, ))
t2 = Thread(target = countdown, args =(COUNT//4, ))
t3 = Thread(target = countdown, args =(COUNT//4, ))
t4 = Thread(target = countdown, args =(COUNT//4, ))
  
start = time.time()
t1.start()
t2.start()
t3.start()
t4.start()
t1.join()
t2.join()
t3.join()
t4.join()
end = time.time()
print('Time taken (four threads) in seconds: ', end - start)

pool = Pool(processes=4)
start = time.time()
r1 = pool.apply_async(countdown, [COUNT//4])
r2 = pool.apply_async(countdown, [COUNT//4])
r3 = pool.apply_async(countdown, [COUNT//4])
r4 = pool.apply_async(countdown, [COUNT//4])
pool.close()
pool.join()
end = time.time()
print('Time taken (four processes) in seconds: ', end - start)
```
which outputs
```
Time taken (one thread) in seconds: 7.321912527084351
Time taken (four threads) in seconds:  7.665801525115967
Time taken (four processes) in seconds:  2.1135129928588867
```
where there is no facilitated computation (four threads should have rendered 1/4 countdown time of by one thread). This is caused by GIL that forces CPU to run by only one thread.

However, it has no restriction on multi-processes.

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

## `__iter__`, `generators` and Coroutine

`iterable`: When you create a list, you can read its items one by one. Reading its items one by one is called iteration,

`generators` are iterators, a kind of iterable you can only iterate over once. Generators do not store all the values in memory, they generate the values on the fly.

### `yield` usage

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

### Coroutine

Coroutines declared with the async/await syntax is the preferred way of writing asyncio applications.

```py
import asyncio

async def helloWorld():
    print("Hello")
    asyncio.sleep(1)
    print("World")
```

Just running `helloWorld()` does not execute the code:
```bash
<coroutine object main at 0x1053bb7c8>
```

Instead, should run by `asyncio.run(helloWorld())` that prints `"Hello World"`.

## Argument pass by object-reference