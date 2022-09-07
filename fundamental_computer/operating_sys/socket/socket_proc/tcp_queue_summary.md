# TCP queue info

Prequeue, receive queue and backlog are the three queues that bridge NIC initiated interrupt handlers `tcp_v4_rcv()` and user initiated `recv()`. 

When `tcp_v4_rcv()` got new packets, it allocates the packets to any of the three queues (prequeue, receive queue and backlog) depending on different scenarios.

1. If socket is on lock, to prevent concurrency of operating socket, data packets are put to backlog.
2. If socket is not on lock, data packets are put to prequeue.
3. If in `tcp_rcv_established`, data packets are put to receive queue.

From `tcp_v4_rcv`
```cpp
// lock the sock
bh_lock_sock_nested(sk);
ret = 0;
// check if user is not currently occupied 
if (!sock_owned_by_user(sk)) {
    // check is prequeue is safe to operate
    if (!tcp_prequeue(sk, skb))
        ret = tcp_v4_do_rcv(sk, skb);
// user is currently operating on the sock
// temporarily put data to backlog
} else if (unlikely(sk_add_backlog(sk, skb,
                   sk->sk_rcvbuf + sk->sk_sndbuf))) {
    bh_unlock_sock(sk);
    NET_INC_STATS_BH(net, LINUX_MIB_TCPBACKLOGDROP);
    goto discard_and_relse;
}
bh_unlock_sock(sk);
```

* Prequeue Queue 

```cpp
struct tcp_sock *tp = tcp_sk(sk);
tp->ucopy.prequeue;
```

When user is currently reading a socket, and set `tp.ucopy.task` but not set `net.ipv4.tcp_low_latency`, packets are put into prequeue.

`skb` retains header info but not processed (this means packets might stored out of order by sequence no). User process is responsible for processing the raw skb packets.

If `sk_rcvbuf` is full, the kernel starts processing prequeue packets and put them into receive queue.

* Receive Queue 

```cpp
struct sock *sk; 
sk->receive_queue
```

Receive queues are finished processing (packets are all in sequence no. order), removed protocol header info, only data inside `sk_buff` is copied to user space's receive queue.

* Backlog Queue

```cpp
struct sock *sk; 
sk->backlog;
```

If user is reading from receive queue or prequeue queue, new packets come into backlog queue.

## Advantages of prequeue

