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

## The bind process

### `__sys_bind`

`sockfd_lookup_light` checks if the used `fd` valid, and `sock->ops->bind` associates the sock with the addr. For TCP/IP, `inet_stream_ops` is set and `bind` eventually invokes `inet_bind`. 

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

### `inet_bind`

For `AF_INET`, `bind()` calls `inet_bind` where `__inet_bind` is invoked. 
```cpp
int inet_bind(struct socket *sock, struct sockaddr *uaddr, int addr_len)
{
	struct sock *sk = sock->sk;
	
	return __inet_bind(sk, uaddr, addr_len, false, true);
}
```

The main function of interest is `sk->sk_prot->get_port(sk, snum)` that invokes `tcp_port.inet_csk_get_port` since in socket creation, sk_prot is defined to be a tcp_port. Besides, `__inet_bind` provides other services as listed below.
```cpp
int __inet_bind(struct sock *sk, struct sockaddr *uaddr, int addr_len,
		bool force_bind_address_no_port, bool with_lock)
{
    struct sockaddr_in *addr = (struct sockaddr_in *)uaddr;
    struct inet_sock *inet = inet_sk(sk);
    struct net *net = sock_net(sk);
    int err;

    // about router, such as 0.0.0.0 defining broadcast and others for one host forwarding 
    tb_id = l3mdev_fib_table_by_index(net, sk->sk_bound_dev_if) ? : tb_id;
    chk_addr_ret = inet_addr_type_table(net, addr->sin_addr.s_addr, tb_id);

    // ntohs big-endian operation to get the port num, do checking such as non-privilege/forbidden use of port lower than 1024 by an ordinary user
    int snum = ntohs(addr->sin_port);
    err = -EACCES;
    if (snum && snum < inet_prot_sock(net) &&
        !ns_capable(net->user_ns, CAP_NET_BIND_SERVICE))
        goto out; 

    // sock is valid and the ip is available for use (not already assigned/occupied)
    err = -EINVAL;
    if (sk->sk_state != TCP_CLOSE || inet->inet_num)
        goto out_release_sock;   

    // for 0.0.0.0 connection, should handover to router to decide what IP to connect
    inet->inet_rcv_saddr = inet->inet_saddr = addr->sin_addr.s_addr;
    if (chk_addr_ret == RTN_MULTICAST || chk_addr_ret == RTN_BROADCAST)
        inet->inet_saddr = 0;  /* Use device */     

    // `sk->sk_prot->get_port(sk, snum)` invokes `tcp_port.inet_csk_get_port` since in socket creation, sk_prot is defined to be a tcp_port
    if (snum || !(inet->bind_address_no_port ||
              force_bind_address_no_port)) {
        if (sk->sk_prot->get_port(sk, snum)) {
            inet->inet_saddr = inet->inet_rcv_saddr = 0;
            err = -EADDRINUSE;
            goto out_release_sock;
        }
        err = BPF_CGROUP_RUN_PROG_INET4_POST_BIND(sk);
        if (err) {
            inet->inet_saddr = inet->inet_rcv_saddr = 0;
            goto out_release_sock;
        }
    }    
}
```

### `inet_csk_get_port`

```cpp
int inet_csk_get_port(struct sock *sk, unsigned short snum)
{
    bool reuse = sk->sk_reuse && sk->sk_state != TCP_LISTEN;
    struct inet_hashinfo *hinfo = sk->sk_prot->h.hashinfo;
    int ret = 1, port = snum;
    struct inet_bind_hashbucket *head;
    struct net *net = sock_net(sk);
    struct inet_bind_bucket *tb = NULL;
    kuid_t uid = sock_i_uid(sk);

    // inet_csk_find_open_port is used when there is no user defined port
    // inet_csk_find_open_port assigns an available port to user
    if (!port) {
        head = inet_csk_find_open_port(sk, &tb, &port);
        if (!head)
            return ret;
        if (!tb)
            goto tb_not_found;
        goto success;
    }

    // inet_bhashfn is an hash algo to locate a net and port
    // inet_bhashfn returns a key to hinfo->bhash to find `inet_bind_hashbucket *head`
    head = &hinfo->bhash[inet_bhashfn(net, port, hinfo->bhash_size)];
    spin_lock_bh(&head->lock);

    // `inet_bind_bucket_for_each` iterates inet_bind_hashbucket->chain to see if bound bucket is used
    inet_bind_bucket_for_each(tb, &head->chain)
        if (net_eq(ib_net(tb), net) && tb->port == port)
            goto tb_found;

    // if found, check if it can be reused
    // if not found/used, create a new
tb_not_found:
    tb = inet_bind_bucket_create(hinfo->bind_bucket_cachep,
                     net, head, port);
    if (!tb)
        goto fail_unlock;

tb_found:
    if (!hlist_empty(&tb->owners)) {
        if (sk->sk_reuse == SK_FORCE_REUSE)
            goto success;

        if ((tb->fastreuse > 0 && reuse) ||
            sk_reuseport_match(tb, sk))
            goto success;
        if (inet_csk_bind_conflict(sk, tb, true, true))
            goto fail_unlock;
    }        
}
```

`inet_hashinfo` is the most important part. `inet_hashinfo` is a global singleton that has `inet_ehash_bucket` and `inet_bind_hashbucket` for established tcp sock and bound tcp sock, respectively. 

```cpp
struct inet_hashinfo {

    // established tcp sock hash bucket
    // key: local host, local port, remote host, remote port
    // value: an establishing or already established socket connection
    // when a tcp request comes in, it first reads addr and port info from header
    // then use the addr and port to locate a bound socket from `inet_ehash_bucket*ehash` 
    // the rest data is loaded into socket's recv buffer
	struct inet_ehash_bucket	*ehash;
	spinlock_t			*ehash_locks;
	unsigned int			ehash_mask;
	unsigned int			ehash_locks_mask;

    // bound hash bucket
    // key: local port
    // value: all sockets that are bound to this port
    // will be added to `inet_bind_hashbucket->chain`
    struct inet_bind_hashbucket	*bhash;

    // slab for bound tcp_sock
	struct kmem_cache		*bind_bucket_cachep;
	unsigned int			bhash_size;
	unsigned int			lhash2_mask;

    // key: local host and port
    // value: sockets that are listening to a port
	struct inet_listen_hashbucket	*lhash2;

    // key: local port
    // value: sockets that are listening to a port
	struct inet_listen_hashbucket	listening_hash[INET_LHTABLE_SIZE]
					____cacheline_aligned_in_smp;
};
```

When an OS boots, `tcp_init` inits `inet_hashinfo`.
```cpp
// net/ipv4/tcp.c
void __init tcp_init(void)
{
        // init inet_hashinfo
}
```

All `inet_bind_bucket` instances connect to each other through its member nodes, added to `chain` of `inet_bind_hashbucket`, finally the `chain` is added to `bhash` of `inet_hashinfo`.

Meanwhile, when client side on `connect` or server side on `bind` status, `inet_bind_bucket` is added to `struct sock`'s `sk_bind_node`, and `inet_csk(sk)->icsk_bind_hash` points to this `inet_bind_bucket`. Hence, `inet_bind_bucket` is bound to `struct sock`.
```cpp
struct inet_bind_hashbucket {

    // lock to control read/write
	spinlock_t		lock;
	
    // a chain of `inet_bind_bucket`s
	struct hlist_head	chain;
};

struct inet_bind_bucket {
    unsigned short		port;
	
	/* port reusability 
	 * 0: bound by bind, not available in use 
	 * 1: bound by bind, not available in use if `sk->sk_reuse` is true
	 * -1: Dynamic binding by client (by `inet_hash_connection()`)
	 */
	signed short		fastreuse;

    // The number of being referenced by `inet_bind_hash`
	int			num_owners;
	
    // To store bhash list nodes
    // `inet_bind_bucket` added to `inet_bind_hashbucket`'s chain through this node.
	struct hlist_node	node;
	
    // `inet_bind_bucket` is added to `struct sock`'s sk_bind_node
    // owners is first bound to `sk->sk_bind_node`, when on timewait, is bound to `tw->tw_bind_node`
	struct hlist_head	owners; 
};
```

```cpp
struct inet_ehash_bucket {

    // After three-way handshake, a tcp connection is formally established, `struct tcp_sock` is added to this hash head.
    // Client can check tcp connection info, such as port,   
	struct hlist_nulls_head chain;
	
    // Used to store time wait sock: inet_timewait_sock
	struct hlist_nulls_head twchain;
};
```

```cpp
// net/ipv4/inet_hashtables.c
void inet_bind_hash(struct sock *sk, struct inet_bind_bucket *tb,
                    const unsigned short snum)
{
    // Saved the bound port
    inet_sk(sk)->inet_num = snum;

    // tb's owners is used to store all bound sockets
    sk_add_bind_node(sk, &tb->owners);

    // tb's memory addr is assigned to icsk_bind_hash
    // user can access tb (has info about socket bhash) through this socket method `inet_csk`
    inet_csk(sk)->icsk_bind_hash = tb;
}
```