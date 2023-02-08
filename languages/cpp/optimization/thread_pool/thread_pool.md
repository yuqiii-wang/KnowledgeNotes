# Thread Pool

Given a number of tasks (e.g., many functions), design a thread pool (data structure by queue),
and assign one task per thread selected from the thread pool.

### Design philosophy:

* First obtain a task queue `queue<Task> _tasks;`, where each task element is defined as `using Task = function<void()>;`
```cpp
_tasks.emplace([task]() {  (*task)();  });
```

* Then design a thread pool. A thread pool `vector<thread> _pool;` adds more threads by placing `task` from `_tasks` in a `while` loop.
`_task_cv.wait` allows proceeding when `!_tasks.empty()` is true (there are some tasks to do in `_tasks`).
```cpp
_pool.emplace_back([this] { 
    while (_run) {
        ...
        _task_cv.wait(lock, [this] { return !_tasks.empty(); });
        ...
        task(); // task execution
        ...
    }
});
```

* 

Some other considerations fo designing a thread pool:

* Task queue is a typical *producer-consumer* pattern, should use `mutex` (to lock front/pop action from queue) and `conditional_variable` (to notify that the lock action finishes)
* Available/busy thread number should be `atomic<int>`
* In C++11, `using Task = function<void()>` takes any function
* `packaged_task` can help get function return value by `get_future`
* When thread pool is destructed, should use `join` to wait for all threads finished tasks