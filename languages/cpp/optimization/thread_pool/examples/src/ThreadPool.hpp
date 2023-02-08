#pragma once
#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <atomic>
#include <future>
#include <condition_variable>
#include <thread>
#include <functional>
#include <stdexcept>

namespace std
{
#define THREADPOOL_MAX_NUM 16

class threadpool
{
private:
    using Task = function<void()>; // task type, must be a std::function taking `void()`
    vector<thread> _pool;          // thread pool
    queue<Task> _tasks;            // task queue
    mutex _lock;                   // mutex for lock
    condition_variable _task_cv;   // block by some condition
    atomic<bool> _run{true};       // thread pool is running
    atomic<int> _idlThrNum{0};     // idle thread number

public:
    inline threadpool(unsigned short size = 4) { addThread(size); }

    inline ~threadpool()
    {
        _run = false;
        _task_cv.notify_all(); // wake up all threads
        for (thread &thread : _pool)
        {
            if (thread.joinable())
                thread.join(); // wait for all threads to finish and join
        }
    }

public:
    // submit a task; use .get() returns when task finishes,
    // implementation either by bind：.commit(std::bind(&Dog::sayHello, &dog));
    // or mem_fn：.commit(std::mem_fn(&Dog::sayHello), this);
    template <class F, class... Args>
    auto commit(F &&f, Args &&...args) -> future<decltype(f(args...))>
    {
        if (!_run) // stoped ??
            throw runtime_error("commit on ThreadPool is stopped.");

        using RetType = decltype(f(args...)); // typename std::result_of<F(Args...)>::type, function return type
        auto task = make_shared<packaged_task<RetType()>>(
            bind(forward<F>(f), forward<Args>(args)...)); // bind function entry and params
        future<RetType> future = task->get_future();
        {   // add to task queue
            // lock_guard auto locks the task queue emplace work; unlocks when out of scope
            lock_guard<mutex> lock{_lock}; 
            _tasks.emplace([task]() {      
                (*task)();
            });
        }
#ifdef THREADPOOL_AUTO_GROW
        if (_idlThrNum < 1 && _pool.size() < THREADPOOL_MAX_NUM)
            addThread(1);
#endif                             // !THREADPOOL_AUTO_GROW
        _task_cv.notify_one();     // wakeup a thread

        return future;
    }

    int idlCount() { return _idlThrNum; }

    int thrCount() { return _pool.size(); }

#ifndef THREADPOOL_AUTO_GROW
private:
#endif // !THREADPOOL_AUTO_GROW

    // add a number of `size` threads
    void addThread(unsigned short size)
    {
        for (; _pool.size() < THREADPOOL_MAX_NUM && size > 0; --size)
        {                               
            _pool.emplace_back([this] { 
                while (_run)
                {
                    Task task; // get a task coming to work
                    {
                        unique_lock<mutex> lock{_lock};
                        // stuck here, wait till got a task
                        _task_cv.wait(lock, [this]
                                        { return !_run || !_tasks.empty(); }); 
                        if (!_run && _tasks.empty())
                            return;
                        task = move(_tasks.front()); // get a task
                        _tasks.pop();
                    }
                    _idlThrNum--;
                    task(); // task execution
                    _idlThrNum++;
                }
            });
            _idlThrNum++;
        }
    }
};

}

#endif // !THREAD_POOL_H