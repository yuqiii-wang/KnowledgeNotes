# Socket

```cpp
// create a socket by OS
int s = socket(AF_INET, SOCK_STREAM, 0);   
// bind a socket/port
bind(s, ...);
// listen to a port
listen(s, ...);
// accept client connection
int c = accept(s, ...);
// receive client data
recv(c, ...);
// printout the data
printf(...);
```

Code above lists the typical work of socket communication.

1. OS create a tmp socket obj when running `socket(AF_INET, SOCK_STREAM, 0);`

The socket object is a file assigned with an `fd` (file descriptor), in which there are separate areas: send_buffer, recv_buffer, queue_list, etc.

2. When program runs at `recv`, this program process is added to the socket obj waiting list by OS.

3. When there is data coming into buffer, `recv` returns.

### `recv` Is A Blocking Service

`recv` eats up the precious CPU time checking for constantly checking if there is data coming into socket buffer.
