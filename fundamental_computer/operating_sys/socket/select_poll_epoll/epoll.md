# EPoll

The major change of `epoll` from `poll` is passive data reception rather than active polling on OS signal by `poll` if there is data arrived.

```cpp
#include <sys/epoll.h>

int epoll_create(int size);
int epoll_create1(int flags);

int epoll_ctl(int epfd, int op, int fd, 
                struct epoll_event *event);

int epoll_wait(int epfd, struct epoll_event *events,
               int maxevents, int timeout);
```

### `epoll_create`

`epoll_create()` creates a new `epoll` instance.

When `flags` is zero, `epoll_create1()` is same as `epoll_create()`

### `epoll_ctl`

Provide interface to `epoll` system by file descriptor `epfd`. It requests that the operation `op` be performed for the target file descriptor, `fd`. 

Some typical `op`s are 
* `EPOLL_CTL_ADD`
Add `fd` to the interest list and associate the settings specified in event with the internal file linked to `fd`. 
* `EPOLLIN`
The associated file is available for read operations. 
* `EPOLLOUT`
The associated file is available for write operations.

### `epoll_wait`

The `epoll_wait` system call waits for events on the `epoll` instance referred to by the file descriptor epfd. 

## Process

File descriptor is used for all the subsequent calls to the `epoll` interface. 

When no longer required, the file descriptor returned by `epoll_create()` should be closed by using `close()`. 

When all file descriptors referring to an `epoll` instance have been closed, OS kernel destroys the instance and releases the associated resources for reuse.  

```cpp
// create an epoll inst
int epoll_fd = epoll_create1(0);
if (epoll_fd == -1) {
    fprintf(stderr, "Failed to create epoll file descriptor\n");
    return 1;
}

const int MAX_EVENTS = 10;
struct epoll_event event, events[MAX_EVENTS];
event.events = EPOLLIN;
event.data.fd = 0;
// one event test if the epoll inst is connectable
if(epoll_ctl(epoll_fd, EPOLL_CTL_ADD, 0, &event))
{
    fprintf(stderr, "Failed to add file descriptor to epoll\n");
    close(epoll_fd);
    return 1;
}

const int READ_SIZE = 100;
char read_buffer[READ_SIZE + 1];
while (true) {
    // number of events arrived at the time when epoll_wait is invoked
    int event_count = epoll_wait(epoll_fd, events, MAX_EVENTS, 30000);
    for (int i = 0; i < event_count; i++) {
        // read from events[i].data.fd
        // read_buffer store data into read_buffer
        size_t bytes_read = read(events[i].data.fd, read_buffer, READ_SIZE);
    }
}

// must close fd
if (close(epoll_fd)) {
    fprintf(stderr, "Failed to close epoll file descriptor\n");
    return 1;
}
```

## Insight Operation Detail

Inside `include/linux/eventpool.h`, there are 

### `eventpoll`

* Instance from `epoll_create`

* Register an OS interrupt callback to kernel by `epoll_ctl` that when there is new data arrival, kernel copies the data into the interest list.

* Provide an interface to an interest list of `epitem` nodes

```cpp
/*
 * This structure is stored inside the "private_data" member of the file
 * structure and represents the main data structure for the eventpoll
 * interface.
 */
struct eventpoll {
    /* Protect the access to this structure */
    spinlock_t lock;

    /*
     * This mutex is used to ensure that files are not removed
     * while epoll is using them. This is held during the event
     * collection loop, the file cleanup path, the epoll file exit
     * code and the ctl operations.
     */
    struct mutex mtx;

    /* Wait queue used by sys_epoll_wait() */
    wait_queue_head_t wq;

    /* Wait queue used by file->poll() */
    wait_queue_head_t poll_wait;

    /* List of ready file descriptors */
    struct list_head rdllist;

    /* RB tree root used to store monitored fd structs */
    struct rb_root rbr;

    /*
     * This is a single linked list that chains all the "struct epitem" that
     * happened while transferring ready events to userspace w/out
     * holding ->lock.
     */
    struct epitem *ovflist;

    /* wakeup_source used when ep_scan_ready_list is running */
    struct wakeup_source *ws;

    /* The user that created the eventpoll descriptor */
    struct user_struct *user;

    struct file *file;

    /* used to optimize loop detection check */
    int visited;
    struct list_head visited_list_link;
};
```

### `epitem`

* `fd`s as nodes in the interested-list which is created by `epoll_ctl`

* Every `fd` as a node referenced by a Red-Black Tree

```cpp
/*
 * Each file descriptor added to the eventpoll interface will
 * have an entry of this type linked to the "rbr" RB tree.
 * Avoid increasing the size of this struct, there can be many thousands
 * of these on a server and we do not want this to take another cache line.
 */
struct epitem {
    union {
        /* RB tree node links this structure to the eventpoll RB tree */
        struct rb_node rbn;
        /* Used to free the struct epitem */
        struct rcu_head rcu;
    };

    /* List header used to link this structure to the eventpoll ready list */
    struct list_head rdllink;

    /*
     * Works together "struct eventpoll"->ovflist in keeping the
     * single linked chain of items.
     */
    struct epitem *next;

    /* The file descriptor information this item refers to */
    struct epoll_filefd ffd;

    /* Number of active wait queue attached to poll operations */
    int nwait;

    /* List containing poll wait queues */
    struct list_head pwqlist;

    /* The "container" of this item */
    struct eventpoll *ep;

    /* List header used to link this item to the "struct file" items list */
    struct list_head fllink;

    /* wakeup_source used when EPOLLWAKEUP is set */
    struct wakeup_source __rcu *ws;

    /* The structure that describe the interested events and the source fd */
    struct epoll_event event;
};
```

## Concurrency Issues

`epoll` is subject to broken reads if two threads are racing to read data buffer from a `fd`.

Solution is to use `EPOLLONETSHOT` by `epoll_ctl` that allows only one trigger.