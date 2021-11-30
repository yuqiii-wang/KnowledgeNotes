# Data Communication

## IPC socket

A Unix domain socket or IPC socket (inter-process communication socket) is a data communications endpoint for exchanging data between processes executing on the same host operating system. All communication occurs entirely within the operating system kernel, rather than via network protocol.

Usually, there are two types:

* SOCK_STREAM (compare to TCP) – for a stream-oriented socket
* SOCK_DGRAM (compare to UDP) – for a datagram-oriented socket that preserves message boundaries (as on most UNIX implementations, UNIX domain datagram sockets are always reliable and don't reorder datagrams)