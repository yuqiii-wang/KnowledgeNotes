# TCP/IP

A tcp packet for IPv4 is shown as below.

![tcp_packet](imgs/tcp_packet.png "tcp_packet")

## TCP windows

The throughput of a TCP communication is limited by two windows: the *congestion window* and the *receive window*. 

The congestion window tries not to exceed the capacity of the network (congestion control); the receive window tries not to exceed the capacity of the receiver to process data (flow control). 

### Tcp data transmission (after handshake)

|Sender||Receiver|Comments|
|-|-|-|-|
|TCP 1| $\rightarrow$ ||A sender sends a tcp packet of a sequence no.1 to a receiver|
||$\leftarrow$|ACK 2|The receiver acknowledges the TCP 1 by replying an ACK asking for next tcp packet of a sequence no.2|
|TCP 2| $\rightarrow$ |||
|TCP 3| $\rightarrow$ ||A tcp sliding window is dynamic, growing by a power of $2$, so that this time the sender sends 2 packets|
||$\leftarrow$|ACK 4||
|TCP 4| $\rightarrow$ |||
|TCP 5| $\rightarrow$ |$\times$ (packet lost, not received) |Probably there was traffic congestion that this packet was lost, and the receiver did not receive this tcp of a sequence no.5 |
|TCP 6| $\rightarrow$ |||
|TCP 7| $\rightarrow$ |||
||$\leftarrow$|ACK 5|The receiver replies to the sender, it is asking for the tcp packet of a sequence no.5|
|TCP 5| $\rightarrow$ ||The sender might see there is possible traffic congestion, only sends one tcp packet on demand|
||$\leftarrow$|ACK 8|Since there was only the no.5 tcp packet lost, others were received (the max sequence no is $7$), the receiver is now asking for the 8-th tcp packet|
|TCP 8| $\rightarrow$ |||
|TCP 9| $\rightarrow$ |||
||...|||

### Specifications

In rfc7323 - TCP Extensions for High Performance, TCP header uses a 16-bit field to report the receive window
size to the sender.  Therefore, the largest window that can be
used is $2^{16} = 64 kb$.

## Recovery

TCP may experience poor performance when multiple packets are lost from one window of data. With the limited information available from cumulative acknowledgments, a TCP sender can only learn about a single lost packet per round trip time. 

* Normal recovery

Just send `ACK` for the missing tcp packets.

* Selective Acknowledgments (SACK)

SACK is an optional/extension field in TCP header that holds information about "gaps" between received tcp packet sequence numbers.

Other enhancements to SACK 
1. Forward Acknowledgment (FACK)
2. Duplicate Selective Acknowledgment (DSACK)
3. Recent Acknowledgment (RACK)