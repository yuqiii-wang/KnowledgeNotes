# Python Garbage Collection

Python has two strategies managing object de-allocation: *reference count* and *garbage collection*

## Reference count

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


## GC module

### Motivation: de-allocation failure

This is never freed since `x` has a reference to itself, and reference counter does not work on this scenario.
```py
x = []
x.append(x) # `x` has a reference to itself

del x # `x` is not longer accessable, but not freed
```

### `gc` solution

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