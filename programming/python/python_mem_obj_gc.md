# Python Memory Investigation, Object Management, and Garbage Collection


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

##


