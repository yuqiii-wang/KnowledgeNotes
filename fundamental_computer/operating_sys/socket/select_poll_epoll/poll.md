# Socket Poll

`poll` draws similarities with `select`, except for it using `pollfd` replacing `fd_set`:

```cpp
int poll(struct pollfd *fds, 
        nfds_t nfds, 
        int timeout);
```

### `struct pollfd *fds`
```cpp
struct pollfd {
    int   fd;         /* file descriptor */
    short events;     /* requested events */
    short revents;    /* returned events */
};
```

`events`: 
a bit mask specifying the events the application is interested in for the file descriptor `fd`.
This field may be specified as zero, in which case the only
events that can be returned in revents are `POLLHUP`, `POLLERR`, and
`POLLNVAL` 

Some typical events are (defined in `<poll.h>`)
* `POLLIN` There is data to read.
* `POLLOUT` Writing is now possible
* `POLLNVAL` Invalid request: fd not open


### `nfds_t nfds`

The number of `pollfd` items.

### `timeout`

The time by which `poll` must return.

## Use Example

```cpp
int nfds, num_open_fds;
struct pollfd *pfds;

char file_list[3][9] = {"file1.txt", "file2.txt", "file3.txt"}

// allocate some memory pfds
num_open_fds = nfds = argc - 1;
pfds = calloc(nfds, sizeof(struct pollfd));

// get fd ready
for (int j = 0; j < nfds; j++) {
    pfds[j].fd = open(file_list[j + 1], O_RDONLY);
    if (pfds[j].fd == -1)
        errExit("open");

    pfds[j].events = POLLIN;
}

while (num_open_fds > 0) {

    int ready = -1;
    // poll blocking, wait till there are data arrived
    ready = poll(pfds, nfds, -1);
    if (ready == -1)
        errExit("poll");

    for (int j = 0; j < nfds; j++) {
        char buf[10];

        // return event type should be `POLLIN` and read data
        if (pfds[j].revents != 0) {
            if (pfds[j].revents & POLLIN) {
                ssize_t s = read(pfds[j].fd, buf, sizeof(buf));
                if (s == -1)
                    errExit("read");
                printf("read %zd bytes: %.*s\n", s, (int) s, buf);
            }
        }
    }
}
```