# Data Communication

## IPC socket

A Unix domain socket or IPC socket (inter-process communication socket) is a data communications endpoint for exchanging data between processes executing on the same host operating system. All communication occurs entirely within the operating system kernel, rather than via network protocol.

Usually, there are two types:

* SOCK_STREAM (compare to TCP) – for a stream-oriented socket
* SOCK_DGRAM (compare to UDP) – for a datagram-oriented socket that preserves message boundaries (as on most UNIX implementations, UNIX domain datagram sockets are always reliable and don't reorder datagrams)

### An Architectural Overview of TCP Unix Socket

network interface (NIC) receives TCP datagrams/packets and transmits the data through Linux kernel (kernel is waken up by interupt or polling NIC services). A typical packet contains about 1000 - 1500 bytes of data, including header.

When kernel gets ONE packet from the NIC it decodes the packet and finds the associated memory and copies the packet to the socket associated buffer. User can run `ssize_t read(int fd, void *buf, size_t count);` to get the payload (or other system calls of same purposes). If user not yet calls `read` while new packets keep coming in, the new packets' payloads appends to the socket's buffer. Once user calls `read`, new packets' payloads reside from the start of buffer memory (however, the socket buffer is not `memset(buf, 0, sizeof(buf))` but overwritten).

### Flow Control

Flow control is the process of managing the rate of data transmission between two nodes to prevent a fast sender from overwhelming a slow receiver. 

* For UDP socket, the server side will just start dropping packets after the receive buffer is filled.

* For TCP, server's socket decreases kernel window size when receiving data exceeds server processing capacity, and rejects incoming TCP Datagrams when window size is zero.

## QUIC

QUIC (Quick UDP Internet Connections) is tailored to facilitate http communication with a number of improvements based on http characteristics.

### TCP disadvantages and QUIC's remediations

Some major comparisons are listed below

TCP: 
1. breaks up the data into small network packets
2. uses checksum to detect corrupt packets
3. sends automatic repeat request (ARQ) for corrupt packets, until proven validity
4. communicates in a sequential/sync manner

QUIC:
1. prepares key exchange for TLS during initial handshake, hence reducing connection overhead
2. uses multiplexing UDP communication so that different protocols can continue receiving data despite some of data streams might be broken (paralellism).
3. includes connection identifier that reduces re-connection time when an end user changes local network (such as jump between different wifi hotspots)