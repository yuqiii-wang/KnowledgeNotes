# TCP queues 

The below four queues are used to bridge the interactions between NIC initiated interrupt handlers `tcp_v4_rcv()` and user initiated `recv()`. 

* Receive queue
* Prequeue
* Backlog
* Out of order queue

When NIC receives a packet, it sends the packet in `skb` to kernel in an interrupt request context, where kernel depending on different scenarios puts the `skb` to different queues. 

User `recv()` depending on blocking/non-blocking mode scans these queues and copies data from the queues to user space `iovec`. 

When user calls `recv()`, the process follows the below order: Receive queue -> Pprequeue -> Backlog. Out of order queue is processed when buffer resource is scarce.

## Receive Queue 

```cpp
struct sock *sk; 
sk->receive_queue
```

Prior to put data into receive queue, `tcp_copy_to_iovec` in `tcp_rcv_established` from softIRQ init by NIC is invoked to copy data directly to user space. If failed, softIRQ puts data to receive queue. 

Receive queues are finished processing (packets are all in sequence no. order), removed protocol header info, only data inside `sk_buff` is copied to user space by `skb_copy_datagram_msg(skb, offset, msg, used)` (for user process's `recv`).

Should read at least `target = sock_rcvlowat(sk, flags & MSG_WAITALL, len);` or `len` bytes of data, otherwise, it either is blocked at `sk_busy_loop(sk, nonblock);` or running on prequeue.

## Prequeue  

Prequeue of the given definition,
```cpp
struct tcp_sock *tp = tcp_sk(sk);
tp->ucopy.prequeue;
```
can start processing `skb`s, when user is currently reading a socket (set a lock to the socket), and set `tp.ucopy.task` but not set `net.ipv4.tcp_low_latency`, packets are put into prequeue.

```cpp
// in tcp_v4_rcv
if (!sock_owned_by_user(sk)) {
    tcp_prequeue(sk, skb);
}

// in tcp_prequeue
if (sysctl_tcp_low_latency || !tp->ucopy.task)
    return false;
else
    __skb_queue_tail(&tp->ucopy.prequeue, skb);    
```

`skb` retains header info but not processed (this means packets might stored out of order by sequence no). User process is responsible for processing the raw skb packets.

If `sk_rcvbuf` is full, the kernel starts processing prequeue packets and put them into receive queue. 

### Discussion of using prequeue

Prequeue is optional and can be disabled. If disabled, when socket is not used/locked by user, `skb`s are handled by `tcp_v4_do_rcv` in a softIRQ context, directly copied to user space via `tcp_copy_to_iovec` or to receive queue awaiting user process.

When dat is put into receive queue, server by `tcp_v4_do_rcv` replies `ack` to client socket. This is misleading since data has not yet reached to server's user space buffer, and user space process might be busy. By sending `ack` to client socket, the client might think that server has finished handling all sent tcp packets (actually not), then next time it sends much more data that might cause traffic jam.  

Besides, fast path is only conditional on tcp packet sequence number being in order, otherwise slow path is selected. Since receive queue only permits packets in order, prequeue is a good alternative to store out of order packets.

However, this does not necessarily improve tcp processing performance, and is removed in IPv6.

## Backlog Queue

```cpp
struct sock *sk; 
sk->backlog;
```

If user is reading from receive queue or prequeue queue, new packets come into backlog queue.

## Out of order queue  


