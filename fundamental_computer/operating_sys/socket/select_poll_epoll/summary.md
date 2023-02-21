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

## `epoll` Operation Summary

Below is a server socket programming by `epoll` for data reception handling.

1. The server creates a socket and the associated epoll.
2. In an indefinite loop, `epoll_wait` is used to wait for any data arrival for every connected socket, the user process is blocked at this step
3. If event is of listening, the server `accept` and `epoll_ctl_add` this new connection
4. If event is of normal data reception, the server just `read` the data
5. If event is of close, the server removes out this connection and close `df`


```cpp
// set socket
set_sockaddr(&srv_addr);
bind(listen_sock, (struct sockaddr *)&srv_addr, sizeof(srv_addr));

setnonblocking(listen_sock);
listen(listen_sock, MAX_CONN);

// create an epoll and add one sock
epfd = epoll_create(1);
epoll_ctl_add(epfd, listen_sock, EPOLLIN | EPOLLOUT | EPOLLET);

while (true) {
    nfds = epoll_wait(epfd, events, MAX_EVENTS, -1);
    for (i = 0; i < nfds; i++) {
    	if (events[i].data.fd == listen_sock) {
            /* handle new connection */
    		conn_sock = accept(listen_sock,
    			   (struct sockaddr *)&cli_addr,
    			   &socklen);
            ...
            epoll_ctl_add(epfd, conn_sock,
					      EPOLLIN | EPOLLET | EPOLLRDHUP |
					      EPOLLHUP);
        }
        else if (events[i].events & EPOLLIN) {
            n = read(events[i].data.fd, buf,
						 sizeof(buf));
        }
    	if (events[i].events & (EPOLLRDHUP | EPOLLHUP)) {
            epoll_ctl(epfd, EPOLL_CTL_DEL,
					  events[i].data.fd, NULL);
    		close(events[i].data.fd);
    		continue;
        }
    }
}
```

### High Performance Reason

`epoll_wait` is blocking user process, but internally it releases CPU resources.

When new data comes to `sock`, it has already registered a callback `ep_poll_callback` that wakes up `epoll` and sends data to user process.
