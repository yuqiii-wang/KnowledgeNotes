# Multiprocessing

## Use of `__name__== '__main__'`

The multiprocessing module works by creating new Python processes that will import your module.

If you did not add `__name__== '__main__'` protection then you would enter a never ending loop of new process creation. It goes like this:

1. Your module is imported and executes code during the import that cause multiprocessing to spawn 4 new processes.
2. Those 4 new processes in turn import the module and executes code during the import that cause multiprocessing to spawn 16 new processes.
4. Those 16 new processes in turn import the module and executes code during the import that cause multiprocessing to spawn 64 new processes.

## `Fork` vs `Spawn` in Python Multiprocessing

Fork is the default on Linux (it isn’t available on Windows), while Windows and MacOS use spawn by default.

### Fork

When a process is forked the child process inherits all the same variables in the same state as they were in the parent. Each child process then continues independently from the forking point. The pool divides the args between the children and they work though them sequentially.

### Spawn

When a process is spawned, it begins by starting a new Python interpreter. The current module is reimported and new versions of all the variables are created. 

## Join, Close, and Terminate 

`Pool.close()` is typically called when the parallelizable part of your main program is finished. Then the worker processes will terminate when all work already assigned has completed.

`Pool.join()` is used to wait for the worker processes to terminate. 

`.kill()` or `.terminate()` forces killing a running process.

The normal use case would be to call `.close()` immediately after `.join()` (or `.kill()` or `.terminate()`); the Process would eventually release the resources even if you don't do that, but it might not happen immediately.

In python, `with` is often used to handle the multi-processing context:
* New in version 3.3: `Pool` objects now support the context management protocol – see Context Manager Types. `__enter__()` returns the `pool` object, and `__exit__()` calls `terminate()`.

```py
from multiprocessing import Pool
  
def countdown(n):
    while n>0:
        n -= 1
    return n

procNum = 12
COUNT = 2000000
with Pool(processes=procNum) as poolProcs:   
            
    procEachResults = [None] * procNum
    for idx in range(procNum):  
        procEachResults[idx] = poolProcs.apply_async(countdown, [COUNT//procNum])

    resultN = [None] * procNum
    for idx in range(procNum): 
        # blocking/waiting for function execution to finish
        resultN[idx] = procEachDayResults[idx].get()    
```