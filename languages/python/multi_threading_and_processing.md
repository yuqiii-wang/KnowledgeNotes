# Python Multi Threading and Multiprocessing

## GIL (Global Interpreter Lock)

A *global interpreter lock* (GIL) is a mechanism used in computer-language interpreters to synchronize the execution of threads so that only one native thread can execute at a time, even on multi-core processor, hence rendering low concurrency code execution.

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

```txt
Time taken (one thread) in seconds: 7.321912527084351
Time taken (four threads) in seconds:  7.665801525115967
Time taken (four processes) in seconds:  2.1135129928588867
```

where there is no facilitated computation (four threads should have rendered 1/4 countdown time of by one thread). This is caused by GIL that forces CPU to run by only one thread.

However, it has no restriction on multi-processes.

## `Fork` vs `Spawn` in Python Multiprocessing

Fork is the default on Linux (it isn't available on Windows), while Windows and MacOS use spawn by default.

### Fork

When a process is forked the child process inherits all the same variables in the same state as they were in the parent. Each child process then continues independently from the forking point. The pool divides the args between the children and they work though them sequentially.

### Spawn

When a process is spawned, it begins by starting a new Python interpreter. The current module is reimported and new versions of all the variables are created.
