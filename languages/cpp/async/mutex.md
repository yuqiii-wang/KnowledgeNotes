# Mutex Under the hood

## `mutex` in Linux

Mutex can be used in an intra- or inter- process context, depending on how you init the mutex.

## `mutex` and `spin_lock`

`Spinlock` is a lock which causes a thread trying to acquire it to simply wait in the loop and repeatedly check for its availability. In contrast, a `mutex` is a program object that is created so that multiple processes can take turns sharing the same resource. 

A thread reached `mutex` immediately goes to sleep, until waken by `mutex.unlock()`; while for `spinlock`, a thread periodically checks it.