# Recv from user

The `recv()`, `recvfrom()`, and `recvmsg()` calls are used to receive
messages from a socket.

```cpp
#include <sys/socket.h>

ssize_t recv(int sockfd, void *buf, size_t len, int flags);
ssize_t recvfrom(int sockfd, void *restrict buf, size_t len, int flags,
                struct sockaddr *restrict src_addr,
                socklen_t *restrict addrlen);
ssize_t recvmsg(int sockfd, struct msghdr *msg, int flags);
```

## Entry of `recv`

It first finds the corresponding `sock` by `fd`, and pends on `sock_recvmsg` waiting for data arrival.

`import_single_range` associates user buffer with user space cache `struct iovec iov`.

```cpp
SYSCALL_DEFINE4(recv, int, fd, void __user *, ubuf, size_t, size,
		unsigned int, flags)
{
	return __sys_recvfrom(fd, ubuf, size, flags, NULL, NULL);
}

int __sys_recvfrom(int fd, void __user *ubuf, size_t size, unsigned int flags,
		   struct sockaddr __user *addr, int __user *addr_len)
{
	struct socket *sock;
	struct iovec iov;
	struct msghdr msg;
	struct sockaddr_storage address;
	int err, err2;
	int fput_needed;

    // init struct msghdr to store data
	err = import_single_range(READ, ubuf, size, &iov, &msg.msg_iter);
	if (unlikely(err))
		return err;

    // find sock by fd    
	sock = sockfd_lookup_light(fd, &err, &fput_needed);  
	if (!sock)
		goto out;

	msg.msg_control = NULL;
	msg.msg_controllen = 0;
	/* Save some cycles and don't copy the address if not needed */
	msg.msg_name = addr ? (struct sockaddr *)&address : NULL;
	/* We assume all kernel code knows the size of sockaddr_storage */
	msg.msg_namelen = 0;
	msg.msg_iocb = NULL;
	msg.msg_flags = 0;
	if (sock->file->f_flags & O_NONBLOCK)
		flags |= MSG_DONTWAIT;

    // read data from socket, blocking mode by default, if no data arrived, it suspends     
	err = sock_recvmsg(sock, &msg, flags);

    // move data from kernel space to user space
	if (err >= 0 && addr != NULL) {
		err2 = move_addr_to_user(&address,
					 msg.msg_namelen, addr, addr_len);
		if (err2 < 0)
			err = err2;
	}

	fput_light(sock->file, fput_needed);
out:
	return err;
}
```

## Sock msg receive

In the earlier setup during `accept` creating a new socket, `sock` is defined using `inet_recvmsg` for its `recvmsg` method.

```cpp
const struct proto_ops inet_stream_ops = {
	.family		   = PF_INET,
	.owner		   = THIS_MODULE,
	.release	   = inet_release,
	.setsockopt	   = sock_common_setsockopt,
	.getsockopt	   = sock_common_getsockopt,
	.sendmsg	   = inet_sendmsg,
	.recvmsg	   = inet_recvmsg,  // `inet_recvmsg` is register as the recv method for `sock_recvmsg`
#ifdef CONFIG_MMU
	.mmap		   = tcp_mmap,
    // .......
}

int sock_recvmsg(struct socket *sock, struct msghdr *msg, int flags)
{
	int err = security_socket_recvmsg(sock, msg, msg_data_left(msg), flags);

	return err ?: sock_recvmsg_nosec(sock, msg, flags);
}

static inline int sock_recvmsg_nosec(struct socket *sock, struct msghdr *msg,
				     int flags)
{
    // ops = inet_stream_ops
    // recvmsg = inet_recvmsg
	return sock->ops->recvmsg(sock, msg, msg_data_left(msg), flags);
}
```

## TCP msg receive

`inet_recvmsg` uses `tcp_recvmsg` to process data. Packets are 
1. packets in flight
2. backlog
3. prequeue
4. receive_queue

```cpp
struct proto tcp_prot = {
	.name			= "TCP",
	.owner			= THIS_MODULE,
	.close			= tcp_close,
	.pre_connect	= tcp_v4_pre_connect,
	.connect		= tcp_v4_connect,
	.disconnect		= tcp_disconnect,
	.accept			= inet_csk_accept,
	.ioctl			= tcp_ioctl,
	.init			= tcp_v4_init_sock,
	.destroy		= tcp_v4_destroy_sock,
	.shutdown		= tcp_shutdown,
	.setsockopt		= tcp_setsockopt,
	.getsockopt		= tcp_getsockopt,
	.keepalive		= tcp_set_keepalive,
	.recvmsg		= tcp_recvmsg,    // `tcp_recvmsg` is registered/hooked as the recv method for `inet_recvmsg`
	.sendmsg		= tcp_sendmsg,
	.sendpage		= tcp_sendpage,
    // ........
}

int inet_recvmsg(struct socket *sock, struct msghdr *msg, size_t size,
		 int flags)
{
	struct sock *sk = sock->sk;
	int addr_len = 0;
	int err;

	if (likely(!(flags & MSG_ERRQUEUE)))
		sock_rps_record_flow(sk);

    // sk_prot = tcp_prot
    // recvmsg = tcp_recvmsg
	err = sk->sk_prot->recvmsg(sk, msg, size, flags & MSG_DONTWAIT,
				   flags & ~MSG_DONTWAIT, &addr_len);
	if (err >= 0)
		msg->msg_namelen = addr_len;
	return err;
}
```

## TCP recv

* Receive queue path
1. Find a good skb by `skb_queue_walk(&sk->sk_receive_queue, skb)` which is true
2. If `if (offset < skb->len) goto found_ok_skb;`
3. Copy data to msg by `skb_copy_datagram_msg(skb, offset, msg, used);`

* Prequeue path
1. `if (!sysctl_tcp_low_latency && tp->ucopy.task == user_recv)` is true, handle prequeue
2. `tcp_prequeue_process` handles the prequeue

* Backlog path
1. `if (copied >= target && !READ_ONCE(sk->sk_backlog.tail))` is true (indicating `sk->sk_backlog.tail` got data) and the above receive queue path is false
2. `__sk_flush_backlog(sk);` processes sk's backlog

```cpp
int tcp_recvmsg(struct sock *sk, struct msghdr *msg, size_t len, int nonblock,
        int flags, int *addr_len)
{
    struct tcp_sock *tp = tcp_sk(sk);
    int copied = 0;
    u32 peek_seq;
    u32 *seq;
    unsigned long used;
    int err;
    int target;        /* Read at least this many bytes */
    long timeo;
    struct task_struct *user_recv = NULL;
    struct sk_buff *skb, *last;
    u32 urg_hole = 0;

    if (unlikely(flags & MSG_ERRQUEUE))
        return inet_recv_error(sk, msg, len, addr_len);

	// if busy, keep looping/polling on skb queue to see if any data comes in.
    if (sk_can_busy_loop(sk) && skb_queue_empty(&sk->sk_receive_queue) &&
        (sk->sk_state == TCP_ESTABLISHED))
        sk_busy_loop(sk, nonblock);

    // avoid software interrupt
    lock_sock(sk);

    err = -ENOTCONN;
    // if socket on tcp listen, not ready and out
    if (sk->sk_state == TCP_LISTEN)
        goto out;

	// read timeout if on blocking mode
	// timeo = 1 when on non-blocking mode
    timeo = sock_rcvtimeo(sk, nonblock);

    /* Urgent data needs to be handled specially. */
    if (flags & MSG_OOB)
        goto recv_urg;

    if (unlikely(tp->repair)) {
        // ...... some repairs
    }

    /* read sequence number */
    seq = &tp->copied_seq;

    /* just check data and copy a sequence number */
    if (flags & MSG_PEEK) {
        peek_seq = tp->copied_seq;
        seq = &peek_seq;
    }

    // get data length
    target = sock_rcvlowat(sk, flags & MSG_WAITALL, len);

    do {
        u32 offset;

        /* Are we at urgent data? Stop if we have read anything or have SIGURG pending. */
        if (tp->urg_data && tp->urg_seq == *seq) {
            // user process handling signal, break
            // ...
            break;
        }

        /* Next get a buffer. */

        /* get the last element from receive queue */
        last = skb_peek_tail(&sk->sk_receive_queue);

        /* iterate receive queue, until find a good skb */
        // the next good skb is marked last/tail
        skb_queue_walk(&sk->sk_receive_queue, skb) {
            last = skb;

            /* Now that we have two receive queues this
             * shouldn't happen.
             *
             * This gives a warning when the element sequence number of the receive queue 
             * is greater than the to-be-retrieved element's sequence number
             */
            if (WARN(before(*seq, TCP_SKB_CB(skb)->seq),
                 "recvmsg bug: copied %X seq %X rcvnxt %X fl %X\n",
                 *seq, TCP_SKB_CB(skb)->seq, tp->rcv_nxt,
                 flags))
                break;

            /* get sequence number offset */
            offset = *seq - TCP_SKB_CB(skb)->seq;

            /* found SYN, do `offset--` */
            if (unlikely(TCP_SKB_CB(skb)->tcp_flags & TCPHDR_SYN)) {
                pr_err_once("%s: found a SYN, please report !\n", __func__);
                offset--;
            }
            /* when offset is smaller than skb data length, `goto found_ok_skb;` */
            if (offset < skb->len)
                goto found_ok_skb;

            /* handle FIN when received FIN */
            if (TCP_SKB_CB(skb)->tcp_flags & TCPHDR_FIN)
                goto found_fin_ok;
            WARN(!(flags & MSG_PEEK),
                 "recvmsg bug 2: copied %X seq %X rcvnxt %X fl %X\n",
                 *seq, TCP_SKB_CB(skb)->seq, tp->rcv_nxt, flags);
        }

        /* Well, if we have backlog, try to process it now yet. */

        /* data all read && backlog is empty */
        if (copied >= target && !sk->sk_backlog.tail)
            break;

        /* otherwise, data not all read; or after reading, backlog is not empty */
        /* already read */
        if (copied) {
            if (sk->sk_err ||
                sk->sk_state == TCP_CLOSE ||
                (sk->sk_shutdown & RCV_SHUTDOWN) ||
                !timeo ||
                signal_pending(current))
                break;
        } else {
            if (sock_flag(sk, SOCK_DONE))
                break;

            if (sk->sk_err) {
                copied = sock_error(sk);
                break;
            }

            if (sk->sk_shutdown & RCV_SHUTDOWN)
                break;

            if (sk->sk_state == TCP_CLOSE) {
                if (!sock_flag(sk, SOCK_DONE)) {
                    /* This occurs when user tries to read
                     * from never connected socket.
                     */
                    copied = -ENOTCONN;
                    break;
                }
                break;
            }

            /* non-blocking wait */
            if (!timeo) {
                copied = -EAGAIN;
                break;
            }

            /* got signal to proccess */
            if (signal_pending(current)) {
                copied = sock_intr_errno(timeo);
                break;
            }
        }

        /* cleanup, to see if ack needs to send */
        tcp_cleanup_rbuf(sk, copied);

        /* no low latency && tp task is empty or is the current process */
        if (!sysctl_tcp_low_latency && tp->ucopy.task == user_recv) {
            /* Install new reader */
            if (!user_recv && !(flags & (MSG_TRUNC | MSG_PEEK))) {
                user_recv = current;
                tp->ucopy.task = user_recv;
                tp->ucopy.msg = msg;
            }

            /* this user cache length */
            tp->ucopy.len = len;

            WARN_ON(tp->copied_seq != tp->rcv_nxt &&
                !(flags & (MSG_PEEK | MSG_TRUNC)));

            /* Ugly... If prequeue is not empty, we have to
             * process it before releasing socket, otherwise
             * order will be broken at second iteration.
             * More elegant solution is required!!!
             *
             * Look: we have the following (pseudo)queues:
             *
             * 1. packets in flight
             * 2. backlog
             * 3. prequeue
             * 4. receive_queue
             *
             * Each queue can be processed only if the next ones
             * are empty. At this point we have empty receive_queue.
             * But prequeue _can_ be not empty after 2nd iteration,
             * when we jumped to start of loop because backlog
             * processing added something to receive_queue.
             * We cannot release_sock(), because backlog contains
             * packets arrived _after_ prequeued ones.
             *
             * Shortly, algorithm is clear --- to process all
             * the queues in order. We could make it more directly,
             * requeueing packets from backlog to prequeue, if
             * is not empty. It is more elegant, but eats cycles,
             * unfortunately.
             */
            if (!skb_queue_empty(&tp->ucopy.prequeue))
                goto do_prequeue;

            /* __ Set realtime policy in scheduler __ */
        }

        /* finished reading, now start processing backlog */
        if (copied >= target) {
            /* Do not sleep, just process backlog. */
            release_sock(sk);
            lock_sock(sk);
        }
        else {
            sk_wait_data(sk, &timeo, last);
        }

        /* user space receiving data */
        if (user_recv) {
            int chunk;

            /* __ Restore normal policy in scheduler __ */

            chunk = len - tp->ucopy.len;

            /* compute the remaining length and to-be-read length */
            if (chunk != 0) {
                NET_ADD_STATS(sock_net(sk), LINUX_MIB_TCPDIRECTCOPYFROMBACKLOG, chunk);
                len -= chunk;
                copied += chunk;
            }

            /*
                all received data finished copied to user space
                && prequeue is not empty
            */
            if (tp->rcv_nxt == tp->copied_seq &&
                !skb_queue_empty(&tp->ucopy.prequeue)) {
do_prequeue:
                tcp_prequeue_process(sk);


                /* read_length - the remaining_length */
                chunk = len - tp->ucopy.len;
                if (chunk != 0) {
                    NET_ADD_STATS(sock_net(sk), LINUX_MIB_TCPDIRECTCOPYFROMPREQUEUE, chunk);
                    len -= chunk;
                    copied += chunk;
                }
            }
        }

        if ((flags & MSG_PEEK) &&
            (peek_seq - copied - urg_hole != tp->copied_seq)) {
            net_dbg_ratelimited("TCP(%s:%d): Application bug, race in MSG_PEEK\n",
                        current->comm,
                        task_pid_nr(current));
            peek_seq = tp->copied_seq;
        }
        continue;

found_ok_skb:
        /* Ok so how much can we use? */

        used = skb->len - offset;

        if (len < used)
            used = len;

        /* Do we have urgent data here? */
        if (tp->urg_data) {
            u32 urg_offset = tp->urg_seq - *seq;

            if (urg_offset < used) {
                if (!urg_offset) {
                    if (!sock_flag(sk, SOCK_URGINLINE)) {
                        /* adjust sequence no and read offset */
                        ++*seq;
                        urg_hole++;
                        offset++;
                        used--;

                        // no data to read
                        if (!used)
                            goto skip_copy;
                    }
                }
                else
                    used = urg_offset;
            }
        }

        /* copy data from skb to msg, copied length is recorded in `used`  */
        if (!(flags & MSG_TRUNC)) {
            err = skb_copy_datagram_msg(skb, offset, msg, used);
            if (err) {
                /* Exception. Bailout! */
                if (!copied)
                    copied = -EFAULT;
                break;
            }
        }

        /* compute the copied and to-be-copied data length */
        *seq += used;
        copied += used;
        len -= used;

        tcp_rcv_space_adjust(sk);

skip_copy:
        if (tp->urg_data && after(tp->copied_seq, tp->urg_seq)) {
            tp->urg_data = 0;
            tcp_fast_path_check(sk);
        }

        /* continue reading  */
        if (used + offset < skb->len)
            continue;
        /* process FIN */
        if (TCP_SKB_CB(skb)->tcp_flags & TCPHDR_FIN)
            goto found_fin_ok;

        /* relase skb if not peeking */
        // will call `__kfree_skb`, `skb_release_all`, `skb_release_head_state`,
        // then `skb->destructor` and finally `sock_rfree`
        if (!(flags & MSG_PEEK))
            sk_eat_skb(sk, skb);
        continue;

found_fin_ok:
        /* Process the FIN. */
        /* increment sequence no */
        ++*seq;
        if (!(flags & MSG_PEEK))
            sk_eat_skb(sk, skb);
        break;
    } while (len > 0);

    /* user space receives data */
    if (user_recv) {
        if (!skb_queue_empty(&tp->ucopy.prequeue)) {
            int chunk;

            tp->ucopy.len = copied > 0 ? len : 0;

            /* prequeue process */
            tcp_prequeue_process(sk);

            /* recalculate copied/not-copied length */
            if (copied > 0 && (chunk = len - tp->ucopy.len) != 0) {
                NET_ADD_STATS(sock_net(sk), LINUX_MIB_TCPDIRECTCOPYFROMPREQUEUE, chunk);
                len -= chunk;
                copied += chunk;
            }
        }

        /* user space task finishes */
        tp->ucopy.task = NULL;
        tp->ucopy.len = 0;
    }

    /* According to UNIX98, msg_name/msg_namelen are ignored
     * on connected socket. I was just happy when found this 8) --ANK
     */

    /* Clean up data we have read: This will do ACK frames. */
    tcp_cleanup_rbuf(sk, copied);

    release_sock(sk);
    return copied;

out:
    release_sock(sk);
    return err;

recv_urg:
    err = tcp_recv_urg(sk, msg, len, flags);
    goto out;

recv_sndq:
    err = tcp_peek_sndq(sk, msg, len);
    goto out;
}
```

## Appendix

`tcp_recvmsg` uses `sk_busy_loop` to wait for data to come in.

```cpp
static inline void sk_busy_loop(struct sock *sk, int nonblock)
{
#ifdef CONFIG_NET_RX_BUSY_POLL
	unsigned int napi_id = READ_ONCE(sk->sk_napi_id);

	if (napi_id >= MIN_NAPI_ID)
		napi_busy_loop(napi_id, nonblock ? NULL : sk_busy_loop_end, sk,
			       READ_ONCE(sk->sk_prefer_busy_poll),
			       READ_ONCE(sk->sk_busy_poll_budget) ?: BUSY_POLL_BUDGET);
#endif
}

void napi_busy_loop(unsigned int napi_id,
		    bool (*loop_end)(void *, unsigned long),
		    void *loop_end_arg, bool prefer_busy_poll, u16 budget)
{
	unsigned long start_time = loop_end ? busy_loop_current_time() : 0;
	int (*napi_poll)(struct napi_struct *napi, int budget);
	void *have_poll_lock = NULL;
	struct napi_struct *napi;

restart:
	napi_poll = NULL;

	rcu_read_lock();

	napi = napi_by_id(napi_id);
	if (!napi)
		goto out;

	preempt_disable();
	for (;;) {
		int work = 0;

		local_bh_disable();
		if (!napi_poll) {
			unsigned long val = READ_ONCE(napi->state);

			/* If multiple threads are competing for this napi,
			 * we avoid dirtying napi->state as much as we can.
			 */
			if (val & (NAPIF_STATE_DISABLE | NAPIF_STATE_SCHED |
				   NAPIF_STATE_IN_BUSY_POLL)) {
				if (prefer_busy_poll)
					set_bit(NAPI_STATE_PREFER_BUSY_POLL, &napi->state);
				goto count;
			}
			if (cmpxchg(&napi->state, val,
				    val | NAPIF_STATE_IN_BUSY_POLL |
					  NAPIF_STATE_SCHED) != val) {
				if (prefer_busy_poll)
					set_bit(NAPI_STATE_PREFER_BUSY_POLL, &napi->state);
				goto count;
			}
			have_poll_lock = netpoll_poll_lock(napi);
			napi_poll = napi->poll;
		}
		work = napi_poll(napi, budget);
		trace_napi_poll(napi, work, budget);
		gro_normal_list(napi);
count:
		if (work > 0)
			__NET_ADD_STATS(dev_net(napi->dev),
					LINUX_MIB_BUSYPOLLRXPACKETS, work);
		local_bh_enable();

		if (!loop_end || loop_end(loop_end_arg, start_time))
			break;

		if (unlikely(need_resched())) {
			if (napi_poll)
				busy_poll_stop(napi, have_poll_lock, prefer_busy_poll, budget);
			preempt_enable();
			rcu_read_unlock();
			cond_resched();
			if (loop_end(loop_end_arg, start_time))
				return;
			goto restart;
		}
		cpu_relax();
	}
	if (napi_poll)
		busy_poll_stop(napi, have_poll_lock, prefer_busy_poll, budget);
	preempt_enable();
out:
	rcu_read_unlock();
}
```

`sock_rcvlowat` returns the min of data length from either the actual socket data length or user input length.

* `SO_RCVLOWAT` and `SO_SNDLOWAT`

Specify the minimum number of bytes in the buffer until the socket layer will pass the data to the protocol (`SO_SNDLOWAT`) or the user on receiving (`SO_RCVLOWAT`).
These two values are initialized to 1. 
```cpp
static inline int sock_rcvlowat(const struct sock *sk, int waitall, int len)
{
    // sk->sk_rcvlowat set to SO_RCVLOWAT
	int v = waitall ? len : min_t(int, READ_ONCE(sk->sk_rcvlowat), len);

	return v ?: 1;
}

#define min_t(type, x, y) ({	\
	type __min1 = (x);			\
	type __min2 = (y);			\
	__min1 < __min2 ? __min1 : __min2; })
```

`import_single_range` associates user buffer `void __user *buf` with user space cache `struct iovec *iov`, where packet data is extracted into.
```cpp
int import_single_range(int rw, void __user *buf, size_t len,
		 struct iovec *iov, struct iov_iter *i)
{
	if (len > MAX_RW_COUNT)
		len = MAX_RW_COUNT;
	if (unlikely(!access_ok(buf, len)))
		return -EFAULT;

	iov->iov_base = buf;
	iov->iov_len = len;
	iov_iter_init(i, rw, iov, 1, len);
	return 0;
}
```

Prequeue operations:
```cpp
static void tcp_prequeue_process(struct sock *sk)
{
    struct sk_buff *skb;
    struct tcp_sock *tp = tcp_sk(sk);

    NET_INC_STATS_USER(sock_net(sk), LINUX_MIB_TCPPREQUEUED);

    /* RX process wants to run with disabled BHs, though it is not
     * necessary */
    local_bh_disable();
    while ((skb = __skb_dequeue(&tp->ucopy.prequeue)) != NULL)
        // 
        sk_backlog_rcv(sk, skb);
     local_bh_enable();   

    /* Clear memory counter. */    
    tp->ucopy.memory = 0;
}

static inline int sk_backlog_rcv(struct sock *sk, struct sk_buff *skb)
{
	if (sk_memalloc_socks() && skb_pfmemalloc(skb))
		return __sk_backlog_rcv(sk, skb);

	return INDIRECT_CALL_INET(sk->sk_backlog_rcv,
				  tcp_v6_do_rcv,
				  tcp_v4_do_rcv,
				  sk, skb);
}
```

backlog operations:
```cpp
void __sk_flush_backlog(struct sock *sk)
{
	spin_lock_bh(&sk->sk_lock.slock);
	__release_sock(sk);
	spin_unlock_bh(&sk->sk_lock.slock);
}

void __release_sock(struct sock *sk)
	__releases(&sk->sk_lock.slock)
	__acquires(&sk->sk_lock.slock)
{
	struct sk_buff *skb, *next;

	while ((skb = sk->sk_backlog.head) != NULL) {
		sk->sk_backlog.head = sk->sk_backlog.tail = NULL;

		spin_unlock_bh(&sk->sk_lock.slock);

		do {
			next = skb->next;
			prefetch(next);
			WARN_ON_ONCE(skb_dst_is_noref(skb));
			skb_mark_not_on_list(skb);
			sk_backlog_rcv(sk, skb);

			cond_resched();

			skb = next;
		} while (skb != NULL);

		spin_lock_bh(&sk->sk_lock.slock);
	}

	/*
	 * Doing the zeroing here guarantee we can not loop forever
	 * while a wild producer attempts to flood us.
	 */
	sk->sk_backlog.len = 0;
}

static inline int sk_backlog_rcv(struct sock *sk, struct sk_buff *skb)
{
	if (sk_memalloc_socks() && skb_pfmemalloc(skb))
		return __sk_backlog_rcv(sk, skb);

	return INDIRECT_CALL_INET(sk->sk_backlog_rcv,
				  tcp_v6_do_rcv,
				  tcp_v4_do_rcv,
				  sk, skb);
}

int __sk_backlog_rcv(struct sock *sk, struct sk_buff *skb)
{
	int ret;
	unsigned int noreclaim_flag;

	/* these should have been dropped before queueing */
	BUG_ON(!sock_flag(sk, SOCK_MEMALLOC));

	noreclaim_flag = memalloc_noreclaim_save();
	ret = INDIRECT_CALL_INET(sk->sk_backlog_rcv,
				 tcp_v6_do_rcv,
				 tcp_v4_do_rcv,
				 sk, skb);
	memalloc_noreclaim_restore(noreclaim_flag);

	return ret;
}
```