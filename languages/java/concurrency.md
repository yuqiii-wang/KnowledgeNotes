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
Executor executor = anExecutor;
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
    // 与前者的区别是，当等待过程中被中断时，仍会继续等待，直到被唤醒，才会设置中断状态
    void awaitUninterruptibly();
    // 让当前线程等待，直到它被告知或中断，或指定的等待时间已经过。
    boolean await(long time, TimeUnit unit) throws InterruptedException;
    // 与上面的类似，让当前线程等待，不过时间单位是纳秒
    long awaitNanos(long nanosTimeout) throws InterruptedException;
    // 让当前线程等待到确切的指定时间，而不是时长
    boolean awaitUntil(Date deadline) throws InterruptedException;
    // 唤醒一个等待当前condition的线程，有多个则随机选一个
    void signal();
    // 唤醒所有等待当前condition的线程
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
For example, here is a class that uses a semaphore to control access to a pool of items:



## Thread Pool

## Daemon Thread

Java offers two types of threads: *user threads* and *daemon threads*.

Daemon thread handles low-priority tasks such as garbage collection that is often not executed when user threads are running.

A daemon thread is launched via setting a normal thread to `setDaemon(true);`.

```java
NewThread daemonThread = new NewThread();
daemonThread.setDaemon(true);
daemonThread.start();
```