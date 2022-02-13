# Python Advanced Usage/Knowledge

## GIL (Global Interpreter Lock)

A global interpreter lock (GIL) is a mechanism used in computer-language interpreters to synchronize the execution of threads so that only one native thread can execute at a time, even on multi-core processor, hence rendering low concurrency code execution.

Consider the code
```py
import time
from threading import Thread
  
def countdown(n):
    while n>0:
        n -= 1

# one thread running countdown
COUNT = 50000000
t0 = Thread(target = countdown, args =(COUNT, ))

start = time.time()
t0.start()
t0.join()
end = time.time()
print('Time taken (one thread) in seconds:', end - start)

# four threads running countdown
COUNT = 50000000
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
  
print('Time taken (four thread) in seconds: ', end - start)
```
which outputs
```
Time taken (one thread) in seconds: 3.64469313621521
Time taken (four thread) in seconds: 3.3186004161834717
```
where there is no facilitated computation (four threads should have rendered 1/4 countdown time of by one thread). This is caused by GIL that forces CPU to run by only one thread.

The GIL has restrictions on multi-processing as well (no significant facilitations when running on multi-processing mode).

## `__new__`

## meta class