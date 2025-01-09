# Python Multi Threading and Multiprocessing

## GIL (Global Interpreter Lock)

The Global Interpreter Lock (GIL) in Python is a mutex that ensures only one thread executes Python bytecode at a time.

GIL guarantees atomicity of updating a variable and the below code outputs 4000 as the result.

```py
import threading

shared_data = 0

def increment():
    global shared_data
    for _ in range(1000):
        current = shared_data
        shared_data = current + 1

threads = [threading.Thread(target=increment) for _ in range(4)]

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()
```

### Thread Safety Lock

However, GIL does not consider context switch that seemingly only one thread is executing the byte code, OS might assign different CPU threads to complete the task.

The below examples show different results.

* Not thread safe

```py
def increment():
    global shared_data
    for _ in range(1000):
        current = shared_data
        time.sleep(0.001)  # Artificial delay to force context switching
        shared_data = current + 1

threads = [threading.Thread(target=increment) for _ in range(4)]
```

* Thread safe with lock

```py
def increment():
    global shared_data
    for _ in range(1000):
        with lock:  # Acquire lock before modifying shared_data
            current = shared_data
            time.sleep(0.001)  # Artificial delay to force context switching
            shared_data = current + 1

threads = [threading.Thread(target=increment) for _ in range(4)]
```

## `Fork` vs `Spawn` in Python Multiprocessing

Fork is the default on Linux (it isn't available on Windows), while Windows and MacOS use spawn by default.

### Fork

When a process is forked the child process inherits all the same variables in the same state as they were in the parent. Each child process then continues independently from the forking point. The pool divides the args between the children and they work though them sequentially.

### Spawn

When a process is spawned, it begins by starting a new Python interpreter. The current module is reimported and new versions of all the variables are created.
