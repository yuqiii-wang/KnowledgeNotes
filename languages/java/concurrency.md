# Java Concurrency

The `java.util.concurrent` package provides tools for creating concurrent applications.

## Runnable and Multi-Threading

`java.lang.Runnable` is an interface that is to be implemented by a class whose instances are intended to be executed by a thread.

A lifecycle of Java multithreading object shows as below.

<div style="display: flex; justify-content: center;">
      <img src="imgs/java_multithreading_lifecycle.png" width="40%" height="30%" alt="java_multithreading_lifecycle" />
</div>
</br>

`Runnable` is a simple interface containing an `abstract void run();`.
User should override `run()` adding user-defined logic into it.

```java
@FunctionalInterface
public interface Runnable {
    /**
     * When an object implementing interface <code>Runnable</code> is used
     * to create a thread, starting the thread causes the object's
     * <code>run</code> method to be called in that separately executing
     * thread.
     * <p>
     * The general contract of the method <code>run</code> is that it may
     * take any action whatsoever.
     *
     * @see     java.lang.Thread#run()
     */
    public abstract void run();
}

public class ExampleClass implements Runnable {  
  
    @Override  
    public void run() {  
        // put your logic here
        System.out.println("Thread has ended");  
    }  
   
    public static void main(String[] args) {  
        ExampleClass ex = new ExampleClass();  
        Thread t1= new Thread(ex);  
        t1.start();  
        System.out.println("Hi");  
    }  
}  
```

## Blocking and Non-Blocking vs Sync and Async

## Java Container/Collection Thread Safety

### `concurrentHashMap`, `HashMap` and `HashTable`



## Executor

An object that executes submitted Runnable tasks. 
It helps decouple between a task actually that actually runs and the task submission to a thread.

For example, rather than by `new Thread(new(RunnableTask())).start()`, should try
```java
Executor executor = new Executor();
executor.execute(new RunnableTask1());
executor.execute(new RunnableTask2());
...
```

The `Executor` implementations provided in this package implement `ExecutorService`, which is a more extensive interface. 
The `ThreadPoolExecutor` class provides an extensible thread pool implementation. 
The `Executors` class provides convenient factory methods for these executors.

A more popular usage is by `ExecutorService` such as
```java
public class Task implements Runnable {
    @Override
    public void run() {
        // task details
    }
}

ExecutorService executor = Executors.newFixedThreadPool(10);
```

## Thread Interrupt

`java.lang.Thread.interrupt()`

## Synchronized and Lock

A piece of logic marked with synchronized becomes a synchronized block, allowing only one thread to execute at any given time.

It is basically a function level mutex.

For example, there are multiple threads simultaneously incrementing the same object `summation`.
The result, if correct, should have been `1000`.
For not having applied a proper mutex by `synchronized`, the `@Test` fails.

```java
public class Increment {

    private int sum = 0;

    public void increment() {
        setSum(getSum() + 1);
    }

    public void setSum(int _sum) { this.sum = _sum; }
    public int getSum() { return this.sum; }
}

@Test
public void MultiThreadIncrement_NoSync() {
    ExecutorService service = Executors.newFixedThreadPool(3);
    Increment summation = new Increment();

    IntStream.range(0, 1000)
      .forEach(count -> service.submit(summation::increment));
    service.awaitTermination(1000, TimeUnit.MILLISECONDS);

    assertEquals(1000, summation.getSum()); // this fails
}
```

Solution is simply adding `synchronized` to `increment()`.
And this function's execution is protected by mutex.

```java
public class IncrementWithSync {

    private int sum = 0;

    public synchronized void increment() {
        setSum(getSum() + 1);
    }

    public void setSum(int _sum) { this.sum = _sum; }
    public int getSum() { return this.sum; }
}
```

In fact, for `getSum() + 1`, the `int sum` should have been defined `AtomicInteger`, and the increment can be done by `getAndIncrement();`.

### JUC Lock and Condition

Similar to c++ mutex and conditional_variable.

The below interfaces are defined in `java.util.concurrent` (J.U.C.).

```java
public interface Lock {
    // get lock
    void lock();
    // Try to acquire the lock, failed if the current thread is called `interrupted`, throw exception `InterruptedException`
    void lockInterruptibly() throws InterruptedException;
    // try lock
    boolean tryLock();
    // try lock with timeout
    boolean tryLock(long time, TimeUnit unit) throws InterruptedException;
    // release lock
    void unlock();
    // return `Condition` which is associated with the lock
    Condition newCondition();
}

public interface Condition {
    // Make the current thread wait until notified or interrupted
    void await() throws InterruptedException;
    // Same as `await()`, but continue waiting even being interrupted
    void awaitUninterruptibly();
    // Same as `await()` but added timeout
    boolean await(long time, TimeUnit unit) throws InterruptedException;
    // Same as above `await(long time, TimeUnit unit)`, but time unit is nanosec
    long awaitNanos(long nanosTimeout) throws InterruptedException;
    // wait by a deadline, not duration
    boolean awaitUntil(Date deadline) throws InterruptedException;
    // wake up a thread associated with this condition
    void signal();
    // wake up ALL threads associated with this condition
    void signalAll();
}
```

A `ReentrantLock` is the basic mutex (mutual exclusion) lock in java.

```java
class X {
    private final ReentrantLock lock = new ReentrantLock();
    // ...

    public void m() {
        lock.lock();  // block until condition holds
        try {
        // ... method body
        } finally {
          lock.unlock();
        }
    }
}
```

A `Semaphore` maintains a set of permits.
Each `acquire()` blocks if necessary until a permit is available, and then takes it.
Each `release()` adds a permit, potentially releasing a blocking acquirer.

Semaphores are often used to restrict the number of threads than can access some (physical or logical) resource. 
For example, here is a class that uses a semaphore to control access to a pool of items.

```java
class ItemQueueUsingSemaphore {

    private Semaphore semaphore;

    public ItemQueueUsingSemaphore(int slotLimit) {
        // set the num of threads that can simultaneously access the semaphore-controlled resources
        semaphore = new Semaphore(slotLimit); 
    }

    boolean tryItem() {
        return semaphore.tryAcquire();
    }

    void releaseItem() {
        semaphore.release();
    }

    int availableSlots() {
        return semaphore.availablePermits();
    }

}
```


### Producer-Consumer Multi-Threading Message Queue Example

Below code is an example of how a message queue cache `BoundedBuffer` of 100 element buffer size can be `put` and `get` via lock acquire/release.

The `put` and `get` are assumed used in multi-threaded env, where blocking takes place `notFull.await();` for `put` when the 100-element size buffer is full;
and `notEmpty.await();` for `get` if empty.

```java
package yuqiexample;

import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.Condition;


class BoundedBuffer {
    final Lock lock = new ReentrantLock();
    final Condition notFull  = lock.newCondition();
    final Condition notEmpty = lock.newCondition();

    final Object[] items = new Object[100];
    int putptr, takeptr, count;

    public void put(Object x) throws InterruptedException {
        lock.lock();
        try {
            while (count == items.length)
                notFull.await();
            items[putptr] = x;
            if (++putptr == items.length) putptr = 0;
            ++count;
            notEmpty.signal();
        } finally {
            lock.unlock();
        }
    }

    public Object get() throws InterruptedException {
        lock.lock();
        try {
            while (count == 0)
                notEmpty.await();
            Object x = items[takeptr];
            if (++takeptr == items.length) takeptr = 0;
            --count;
            notFull.signal();
            return x;
        } finally {
            lock.unlock();
        }
    }
}
```

The usage of the `BoundedBuffer` in the below example serves as a message queue.
Elements are taken out and stored in `ArrayBlockingQueue<Integer> q` and `ArrayList<Integer> a`.

The result is that, `q` can receive all 1000 elements, while `a` fails for not supporting multi-threaded `add`.

```java
package yuqiexample;

import java.util.ArrayList;
import java.util.concurrent.*;
import java.util.stream.IntStream;


public class Main {
    public static void main(String[] args) {
        ExecutorService servicePut = Executors.newFixedThreadPool(3);
        ExecutorService serviceGet = Executors.newFixedThreadPool(3);
        BoundedBuffer boundedBuffer = new BoundedBuffer();

        IntStream.range(0, 1000)
            .forEach(count -> servicePut.submit(() -> {
                try{
                    boundedBuffer.put(1);
                }
                catch (InterruptedException e) {
                    System.out.println(e.toString());
                }
                finally {}
            }));

        ArrayBlockingQueue<Integer> q = new ArrayBlockingQueue<Integer>(2000);
        ArrayList<Integer> a = new ArrayList<Integer>();
        IntStream.range(0, 1000)
            .forEach(count -> serviceGet.submit(() -> {
                try{
                    Integer objInt = (Integer) boundedBuffer.get();
                    q.put(objInt);
                    a.add(objInt);
                }
                catch (InterruptedException e) {
                    System.out.println(e.toString());
                }
                finally {}
            }));

        try {
            servicePut.awaitTermination(1000, TimeUnit.MILLISECONDS);
            serviceGet.awaitTermination(1000, TimeUnit.MILLISECONDS);
        }
        catch (InterruptedException e) {
            System.out.println(e.toString());
        }
        finally {
            System.out.println("ArrayBlockingQueue size: " + q.size());
            System.out.println("ArrayList size: " + a.size());
            servicePut.shutdown();
            serviceGet.shutdown();
        }
    }
}
```

### Dead Lock

* During `synchronized` and `reentrantLock.lock()` acquring a lock

## Thread Pool

## Daemon Thread

Daemon is a concept referring to backend services independent from user services, and is used to provide assistance to the user services.

Java offers two types of threads: *user threads* and *daemon threads*.

Daemon thread handles low-priority tasks such as garbage collection that is often not executed when user threads are running.

A daemon thread is launched via setting a normal thread to `setDaemon(true);`.

```java
package yuqiexamples;

public class DaemonExample {
	public static void main(String[] args) {
		ThreadDemo threaddemo = new ThreadDemo();
		Thread threadson = new Thread(threaddemo);
		// set daemon
		threadson.setDaemon(true);
		// start thread
		threadson.start();
		System.out.println("bbb");
	}
}
class ThreadDemo implements Runnable{
	
	@Override
	public void run() {
		System.out.println("aaa");
        // if this thread is of user, shall never die; if of daemon, will die after the parent thread dies
		while(true); 
	}
	
}
```

When user threads are dead, daemon threads will be soon dead as well.

The code above prints
```txt
bbb
aaa
```


### Use Case in Servlet

In servlet, there are many spawned threads handling requests/responses.

If the main web container dies, all spawned threads should terminates as well.
Besides, in web container, there should be a high-priority thread that schedules what requests go to what thread, and the actual processing threads are of low-priority,

By the above consideration, the spawned threads can be set to daemons.