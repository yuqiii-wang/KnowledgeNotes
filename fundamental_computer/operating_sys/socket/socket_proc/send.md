# Connect

The `connect()` system call connects the socket referred to by the
file descriptor `sockfd` to the address specified by `addr`.  The
`addrlen` argument specifies the size of `addr`.  The format of the
address in `addr` is determined by the address space of the socket `sockfd`;

```cpp
#include <sys/socket.h>

int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen);
```