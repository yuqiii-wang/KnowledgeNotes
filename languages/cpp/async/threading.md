# Threads

*Join* in the context of thread refers to returning to the main thread when the branched thread ends. Every `std::thread` object is in one of two states: *joinable* or *unjoinable*.

Unjoinable:
* Default-constructed `std::threads`. Such std::threads have no function to
execute, hence don’t correspond to an underlying thread of execution.
* `std::thread` objects that have been moved from. The result of a move is that
the underlying thread of execution a std::thread used to correspond to (if any)
now corresponds to a different std::thread.
* `std::thread`s that have been joined. After a join, the `std::thread` object no
longer corresponds to the underlying thread of execution that has finished running.
* `std::thread`s that have been detached. A detach severs the connection
between a std::thread object and the underlying thread of execution it corresponds to.

## Implicit Join/Detach

Implicit join occurs if a thread runs out of lifecycle without explicitly finished running `join`, such as code below that `t` is implicitly joined when `doWork` returns.

```cpp
void doWork(){
    std::thread t([]{
        for (int i = 0; i < 1'000'000; i++) {
            sleep(1);
        }
    });
    t.start();
} // end of doWork(); t is not joined
```

There are two scenarios to run:

* An implicit join. In this case, a `std::thread`’s destructor would wait for its
underlying asynchronous thread of execution to complete. You might see this thread pending to stop.

* An implicit detach. In this case, a std::thread’s destructor would sever the
connection between the std::thread object and its underlying thread of execution.

### Explicitly join a thread

It is a good idea to force joining a thread defined in a class destructor when the thread's scope ends.

```cpp
~ThreadRAII()
{
    if (t.joinable()) {  // joinability test
        if (action == DtorAction::join) {
            t.join();
        } else {
            t.detach();
        }
    }
}
```