# Accept

`accept` extracts the first
connection request on the queue of pending connections for the
listening socket, `sockfd`, creates a new connected socket, and
returns a new file descriptor referring to that socket.

After the successful creation of the new socket, user can use `recv()` to retrieve data packets.

```cpp
#include <sys/socket.h>

int accept(int sockfd, struct sockaddr *restrict addr, socklen_t *restrict addrlen);
```

## Entry of `accept`

`accept` creates a new sock with references by local host + port and client host + port. Since clients must have different hosts and ports, `accept` can have multiple connections to the same local host and port (typically `127.0.0.1:80` for http).

`accept` returns a new `fd` for the new socket dedicated to receiving one client data.

```cpp
/*
 *	For accept, we attempt to create a new socket, set up the link
 *	with the client, wake up the client, then return the new
 *	connected fd. We collect the address of the connector in kernel
 *	space and move it to user at the very end. This is unclean because
 *	we open the socket then return an error.
 *
 *	1003.1g adds the ability to recvmsg() to query connection pending
 *	status to recvmsg. We need to add that support in a way thats
 *	clean when we restucture accept also.
 */ 
asmlinkage long sys_accept(int fd, struct sockaddr *upeer_sockaddr, int *upeer_addrlen)
{
	struct socket *sock, *newsock;
	int err, len;
	char address[MAX_SOCK_ADDR];
 
    // find sock by fd
	sock = sockfd_lookup(fd, &err);
	if (!sock)
		goto out;
 
    // allocate a new sock dedicated to client 
	err = -EMFILE;
	if (!(newsock = sock_alloc())) 
		goto out_put;
 
    // init the new sock
	newsock->type = sock->type;
	newsock->ops = sock->ops;

    // ......
 
    // this invokes inet_accept for TCP connection
	err = sock->ops->accept(sock, newsock, sock->file->f_flags);
	if (err < 0)
		goto out_release;
 
    // need to get client socket addr and port
    // move data from kernel space to user space 
	if (upeer_sockaddr) {
		if (newsock->ops->getname(newsock, (struct sockaddr *)address, &len, 2) < 0) {
			err = -ECONNABORTED;
			goto out_release;
		}
		err = move_addr_to_user(address, len, upeer_sockaddr, upeer_addrlen);
		if (err < 0)
			goto out_release;
	}
 
	/* File flags are not inherited via accept() unlike another OSes. */
	if ((err = sock_map_fd(newsock)) < 0)
		goto out_release;
 
out_put:
	sockfd_put(sock);
out:
	return err;
 
// got err, discharge the new sock and goto `sockfd_put(sock)` 
out_release:
	sock_release(newsock);
	goto out_put;
}
```

`sock->ops->accept(sock, newsock, sock->file->f_flags);` from `sys_accept` invokes `unix_accept`.

`sk_buff *skb;` is where `recv()` gets the data from.
```cpp
//net/unix/af_unix.c 
static int unix_accept(struct socket *sock, struct socket *newsock, int flags)
{
	unix_socket *sk = sock->sk;
	unix_socket *tsk;
	struct sk_buff *skb;
	int err;
 
	err = -EOPNOTSUPP;
	if (sock->type!=SOCK_STREAM)
		goto out;
 
	err = -EINVAL;
	if (sk->state!=TCP_LISTEN)
		goto out;
 
	/* If socket state is TCP_LISTEN it cannot change (for now...),
	 * so that no locks are necessary.
	 */
 
	skb = skb_recv_datagram(sk, 0, flags&O_NONBLOCK, &err);
	if (!skb)
		goto out;
 
	tsk = skb->sk;
	skb_free_datagram(sk, skb);
	wake_up_interruptible(&sk->protinfo.af_unix.peer_wait);
 
	/* attach accepted sock to socket */
	unix_state_wlock(tsk);
	newsock->state = SS_CONNECTED;
	sock_graft(tsk, newsock);
	unix_state_wunlock(tsk);
	return 0;
 
out:
	return err;
}
```

`__skb_recv_datagram` peeks/retrieves date packet depending on blocking or non-blocking mode. Peeking means just to check if skb receive queue has data and return immediately (this is fast).
```cpp
struct sk_buff *__skb_recv_datagram(struct sock *sk, unsigned flags,
                 int *peeked, int *err)
{
    struct sk_buff *skb;
    long timeo;

    int error = sock_error(sk);

    if (error)
        goto no_packet;
     /* when socket is on blocking mode, get the timeout value */
    timeo = sock_rcvtimeo(sk, flags & MSG_DONTWAIT);

    do {
        unsigned long cpu_flags;
        // lock the socket to avoid other thread access/retrieving data  
        spin_lock_irqsave(&sk->sk_receive_queue.lock, cpu_flags);
        // check socket receive queue
        skb = skb_peek(&sk->sk_receive_queue);
        if (skb) {
            *peeked = skb->peeked;

            // MSG_PEEK means just have a look, not popping out the element
            if (flags & MSG_PEEK) {
                skb->peeked = 1;
                atomic_inc(&skb->users);
            } else 
                // get the data element from receive queue
                __skb_unlink(skb, &sk->sk_receive_queue); 
        }
        spin_unlock_irqrestore(&sk->sk_receive_queue.lock, cpu_flags);

         // skb got data, returned
        if (skb)
            return skb;

        /* User doesn't want to wait */
        error = -EAGAIN;
        /*
        two scenarios for time0 == 0:
        1. socket is non-blocking
        2. socket blocking is timeout -> jump to no_packet
        */
        if (!timeo)
            goto no_packet;

    } while (!wait_for_packet(sk, err, &timeo)); // blocking for waiting for a packet to come in

    return NULL;

no_packet:
    *err = error;
    return NULL;
}
```

Sleep the running thread until conditions are met.
```cpp
static int wait_for_packet(struct sock *sk, int *err, long *timeo_p)
{
    int error;
     /* 
     Some process scheduling work to get socket init for wait
     */
    DEFINE_WAIT_FUNC(wait, receiver_wake_function);

    prepare_to_wait_exclusive(sk_sleep(sk), &wait, TASK_INTERRUPTIBLE);

    /* Socket errors? */
    error = sock_error(sk);
    if (error)
        goto out_err;

     /* if skb receive queue is not empty, no more work should do, just return  */
    if (!skb_queue_empty(&sk->sk_receive_queue))
        goto out;

    
    /* Socket shut down? */
    if (sk->sk_shutdown & RCV_SHUTDOWN)
        goto out_noerr;

    /* Sequenced packets can come disconnected.
     * If so we report the problem
     */
    error = -ENOTCONN;
    if (connection_based(sk) &&
     !(sk->sk_state == TCP_ESTABLISHED || sk->sk_state == TCP_LISTEN))
        goto out_err;

    /* handle signals */
    // Make sure that a pending signal can wake a blocking process
    if (signal_pending(current))
        goto interrupted;

    error = 0;
    /* By already set TASK_INTERRUPTIBLE, sleep this process until waken by signal when conditions are met*/
    *timeo_p = schedule_timeout(*timeo_p);
out:
     /* Some cleanup work */
    finish_wait(sk_sleep(sk), &wait);
    return error;
interrupted:
    error = sock_intr_errno(*timeo_p);
out_err:
    *err = error;
    goto out;
out_noerr:
    *err = 0;
    error = 1;
    goto out;
}
```