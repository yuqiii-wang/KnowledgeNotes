# Summary

## Operations

One socket establishment means that one `fd` gets created, 
and need to monitor this `fd` to retrieve data.

* When a process calls `select`, `select` iterates and copies all `fd` (file descriptors) to kernel space; then uses `socket`'s poll to see if any of the `fd` has new data arrival.
If none, the process goes to sleep until next  time wake up.

* `poll` has similar operations as `select`'s, but uses a new data structure to `pollfd` instead of `fd_set`.
The new data structure removes the limit of max number of `fd` can be monitored specified in `fd_set`.

* `epoll`'s improvement is removal of iterating all `fd`s.
First, it uses red-black tree to store events, supporting fast CRUD operations.
Second, it passively receives notification about data arrival at `fd` via maintained a ready list `rdlist` that has info about which sockets have new data.

## Performance

* Upon ready IO, select/poll are $O(n)$, epoll is $O(n_{ready})$, where $n$ is the total number of all `fd`s, $n_{ready}$ is the number of `fd` with buffers arrived of data.

* `epoll` has the best performance results since it passively received data notification then run data retrieval, rather than the other two scanning through all registered `fd`

* If the number of listening `fd`s is small, `select` might actually faster than `epoll` since `epoll` requires many kernel function invocation.

![poll-times](imgs/poll-times.png "poll-times")
