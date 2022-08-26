# Bind

The `bind()` function binds a unique local name (typically an IP addr and a port for TCP/IP communication) to the socket with a file descriptor.

Example usage shown as below
```cpp
#define _OE_SOCKETS
#include <sys/types.h>
#include <sys/socket.h>

struct sockaddr_in myname;
/* Bind to a specific interface in the Internet domain */
/* make sure the sin_zero field is cleared */
memset(&myname, 0, sizeof(myname));
myname.sin_family = AF_INET;
myname.sin_addr.s_addr = inet_addr("129.5.24.1"); 
/* specific interface */
myname.sin_port = htons(1024);

rc = bind(s, (struct sockaddr *) &myname, sizeof(myname));
```

Declaration:
```cpp
int bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen);
```
* `socket`
The socket descriptor returned by a previous `socket()` call.
* `address`
The pointer to a sockaddr structure containing the name that is to be bound to socket.

Below is an example for a socket descriptor created in the AF_INET domain.
```cpp
struct in_addr {
    ip_addr_t s_addr;
};

struct sockaddr_in {
    unsigned char  sin_len;
    unsigned char  sin_family;
    unsigned short sin_port;
    struct in_addr sin_addr;
    unsigned char  sin_zero[8];

};
```
* `address_len`
The size of address in bytes.

## `__sys_bind`

```cpp
int __sys_bind(int fd, struct sockaddr __user *umyaddr, int addrlen)
{
    struct socket *sock;
    struct sockaddr_storage address;
    int err, fput_needed;

    sock = sockfd_lookup_light(fd, &err, &fput_needed);

    sock->ops->bind(sock,
                    (struct sockaddr *)
                    &address, addrlen);
}
```

## `inet_bind`

For `AF_INET`, `bind()` calls `inet_bind` where `__inet_bind` is invoked. 
```cpp
int inet_bind(struct socket *sock, struct sockaddr *uaddr, int addr_len)
{
	struct sock *sk = sock->sk;
	
	return __inet_bind(sk, uaddr, addr_len, false, true);
}
```

```cpp
int __inet_bind(struct sock *sk, struct sockaddr *uaddr, int addr_len,
		bool force_bind_address_no_port, bool with_lock)
```