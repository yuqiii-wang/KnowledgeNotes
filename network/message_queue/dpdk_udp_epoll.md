# Data Plane Development Kit (DPDK), UDP and EPoll

## DPDK Interrupts

DPDK uses epoll to listen to UIO devices to mimic Linux handling interrupts for arrived packets.

In `rte_eal_intr_init`, all UIO devices are registered in `intr_sources` that are init.

Then `intr_pipe` is used to control `epoll`, that the passed `fd` should be used for the interrupts.

Finally, set up a thread running `eal_intr_thread_main`.

```cpp
int rte_eal_intr_init(void)
{
  int ret = 0;

  /* init the global interrupt source head */
  TAILQ_INIT(&intr_sources);

  /*
   * create a pipe which will be waited by epoll and notified to
   * rebuild the wait list of epoll.
  */
  if (pipe(intr_pipe.pipefd) < 0)
    return -1;

  /* create the host thread to wait/handle the interrupt */
  ret = pthread_create(&intr_thread,NULL, eal_intr_thread_main, NULL);

    if (ret != 0)
      RTE_LOG(ERR, EAL, "Failed to create thread for interrupt handling\n");

  return ret;
}
```

`eal_intr_thread_main` runs in a loop, where `int pfd = epoll_create(1);` launches an epoll,
then `epoll_ctl(pfd, EPOLL_CTL_ADD, intr_pipe.readfd, ...)` binds `fd`,
finally `eal_intr_handle_interrupts(pfd, numfds);` performs `epoll_wait` processing arrived packets.


```cpp
/* Manipulate an epoll instance "epfd". Returns 0 in case of success,
   -1 in case of error ( the "errno" variable will contain the
   specific error code ) The "op" parameter is one of the EPOLL_CTL_*
   constants defined above. The "fd" parameter is the target of the
   operation. The "event" parameter describes which events the caller
   is interested in and any associated user data.  */
extern int epoll_ctl (int __epfd, int __op, int __fd,
		      struct epoll_event *__event) __THROW;

/**
 * It builds/rebuilds up the epoll file descriptor with all the
 * file descriptors being waited on. Then handles the interrupts.
 *
 * @param arg
 *  pointer. (unused)
 *
 * @return
 *  never return;
 */
static __rte_noreturn void *
eal_intr_thread_main(__rte_unused void *arg)
{
	/* host thread, never break out */
	for (;;) {
		/* build up the epoll fd with all descriptors we are to
		 * wait on then pass it to the handle_interrupts function
		 */
		static struct epoll_event pipe_event = {
			.events = EPOLLIN | EPOLLPRI,
		};
		struct rte_intr_source *src;
		unsigned numfds = 0;

		/* create epoll fd */
		int pfd = epoll_create(1);
		if (pfd < 0)
			rte_panic("Cannot create epoll instance\n");

		pipe_event.data.fd = intr_pipe.readfd;
		/**
		 * add pipe fd into wait list, this pipe is used to
		 * rebuild the wait list.
		 */
		if (epoll_ctl(pfd, EPOLL_CTL_ADD, intr_pipe.readfd,
						&pipe_event) < 0) {
			rte_panic("Error adding fd to %d epoll_ctl, %s\n",
					intr_pipe.readfd, strerror(errno));
		}
		numfds++;

		rte_spinlock_lock(&intr_lock);

		TAILQ_FOREACH(src, &intr_sources, next) {
			struct epoll_event ev;

			if (src->callbacks.tqh_first == NULL)
				continue; /* skip those with no callbacks */
			memset(&ev, 0, sizeof(ev));
			ev.events = EPOLLIN | EPOLLPRI | EPOLLRDHUP | EPOLLHUP;
			ev.data.fd = rte_intr_fd_get(src->intr_handle);

			/**
			 * add all the uio device file descriptor
			 * into wait list.
			 */
			if (epoll_ctl(pfd, EPOLL_CTL_ADD,
					rte_intr_fd_get(src->intr_handle), &ev) < 0) {
				rte_panic("Error adding fd %d epoll_ctl, %s\n",
					rte_intr_fd_get(src->intr_handle),
					strerror(errno));
			}
			else
				numfds++;
		}
		rte_spinlock_unlock(&intr_lock);
		/* serve the interrupt */
		eal_intr_handle_interrupts(pfd, numfds);

		/**
		 * when we return, we need to rebuild the
		 * list of fds to monitor.
		 */
		close(pfd);
	}
}
```

`eal_intr_handle_interrupts` pends on `epoll_wait` until new packets come in.
Given the epoll event, it runs `eal_intr_process_interrupts` to read data.
The reading behavior is a typical read from a file descriptor `read(intr_pipe.readfd, buf.charbuf, sizeof(buf.charbuf))`

For successfully read bytes, invoke the callback function by `active_cb.cb_fn(active_cb.cb_arg);`.

```cpp
struct rte_intr_source {
	TAILQ_ENTRY(rte_intr_source) next;
	struct rte_intr_handle *intr_handle; /**< interrupt handle */
	struct rte_intr_cb_list callbacks;  /**< user callbacks */
	uint32_t active;
};


/**
 * It handles all the interrupts.
 *
 * @param pfd
 *  epoll file descriptor.
 * @param totalfds
 *  The number of file descriptors added in epoll.
 *
 * @return
 *  void
 */
static void
eal_intr_handle_interrupts(int pfd, unsigned totalfds)
{
	struct epoll_event events[totalfds];
	int nfds = 0;

	for(;;) {
		nfds = epoll_wait(pfd, events, totalfds,
			EAL_INTR_EPOLL_WAIT_FOREVER);
		/* epoll_wait fail */
		if (nfds < 0) {
			if (errno == EINTR)
				continue;
			RTE_LOG(ERR, EAL,
				"epoll_wait returns with fail\n");
			return;
		}
		/* epoll_wait timeout, will never happens here */
		else if (nfds == 0)
			continue;
		/* epoll_wait has at least one fd ready to read */
		if (eal_intr_process_interrupts(events, nfds) < 0)
			return;
	}
}

static int
eal_intr_process_interrupts(struct epoll_event *events, int nfds)
{
	bool call = false;
	int n, bytes_read, rv;
	struct rte_intr_source *src;
	struct rte_intr_callback *cb, *next;
	union rte_intr_read_buffer buf;
	struct rte_intr_callback active_cb;

	for (n = 0; n < nfds; n++) {

		/**
		 * if the pipe fd is ready to read, return out to
		 * rebuild the wait list.
		 */
		if (events[n].data.fd == intr_pipe.readfd){
			int r = read(intr_pipe.readfd, buf.charbuf,
					sizeof(buf.charbuf));
			RTE_SET_USED(r);
			return -1;
		}
		rte_spinlock_lock(&intr_lock);
		TAILQ_FOREACH(src, &intr_sources, next)
			if (rte_intr_fd_get(src->intr_handle) == events[n].data.fd)
				break;
		if (src == NULL){
			rte_spinlock_unlock(&intr_lock);
			continue;
		}

		/* mark this interrupt source as active and release the lock. */
		src->active = 1;
		rte_spinlock_unlock(&intr_lock);

		/* set the length to be read dor different handle type */
		switch (rte_intr_type_get(src->intr_handle)) {
		case RTE_INTR_HANDLE_UIO:
		case RTE_INTR_HANDLE_UIO_INTX:
			bytes_read = sizeof(buf.uio_intr_count);
			break;
		case RTE_INTR_HANDLE_ALARM:
			bytes_read = sizeof(buf.timerfd_num);
			break;
#ifdef VFIO_PRESENT
#ifdef HAVE_VFIO_DEV_REQ_INTERFACE
		case RTE_INTR_HANDLE_VFIO_REQ:
#endif
		case RTE_INTR_HANDLE_VFIO_MSIX:
		case RTE_INTR_HANDLE_VFIO_MSI:
		case RTE_INTR_HANDLE_VFIO_LEGACY:
			bytes_read = sizeof(buf.vfio_intr_count);
			break;
#endif
		case RTE_INTR_HANDLE_VDEV:
		case RTE_INTR_HANDLE_EXT:
			bytes_read = 0;
			call = true;
			break;
		case RTE_INTR_HANDLE_DEV_EVENT:
			bytes_read = 0;
			call = true;
			break;
		default:
			bytes_read = 1;
			break;
		}

		if (bytes_read > 0) {
			/**
			 * read out to clear the ready-to-be-read flag
			 * for epoll_wait.
			 */
			bytes_read = read(events[n].data.fd, &buf, bytes_read);
			if (bytes_read < 0) {
				if (errno == EINTR || errno == EWOULDBLOCK)
					continue;

				RTE_LOG(ERR, EAL, "Error reading from file "
					"descriptor %d: %s\n",
					events[n].data.fd,
					strerror(errno));
				/*
				 * The device is unplugged or buggy, remove
				 * it as an interrupt source and return to
				 * force the wait list to be rebuilt.
				 */
				rte_spinlock_lock(&intr_lock);
				TAILQ_REMOVE(&intr_sources, src, next);
				rte_spinlock_unlock(&intr_lock);

				for (cb = TAILQ_FIRST(&src->callbacks); cb;
							cb = next) {
					next = TAILQ_NEXT(cb, next);
					TAILQ_REMOVE(&src->callbacks, cb, next);
					free(cb);
				}
				rte_intr_instance_free(src->intr_handle);
				free(src);
				return -1;
			} else if (bytes_read == 0)
				RTE_LOG(ERR, EAL, "Read nothing from file "
					"descriptor %d\n", events[n].data.fd);
			else
				call = true;
		}

		/* grab a lock, again to call callbacks and update status. */
		rte_spinlock_lock(&intr_lock);

		if (call) {

			/* Finally, call all callbacks. */
			TAILQ_FOREACH(cb, &src->callbacks, next) {

				/* make a copy and unlock. */
				active_cb = *cb;
				rte_spinlock_unlock(&intr_lock);

				/* call the actual callback */
				active_cb.cb_fn(active_cb.cb_arg);

				/*get the lock back. */
				rte_spinlock_lock(&intr_lock);
			}
		}
		/* we done with that interrupt source, release it. */
		src->active = 0;

		rv = 0;

		/* check if any callback are supposed to be removed */
		for (cb = TAILQ_FIRST(&src->callbacks); cb != NULL; cb = next) {
			next = TAILQ_NEXT(cb, next);
			if (cb->pending_delete) {
				TAILQ_REMOVE(&src->callbacks, cb, next);
				if (cb->ucb_fn)
					cb->ucb_fn(src->intr_handle, cb->cb_arg);
				free(cb);
				rv++;
			}
		}

		/* all callbacks for that source are removed. */
		if (TAILQ_EMPTY(&src->callbacks)) {
			TAILQ_REMOVE(&intr_sources, src, next);
			rte_intr_instance_free(src->intr_handle);
			free(src);
		}

		/* notify the pipe fd waited by epoll_wait to rebuild the wait list */
		if (rv > 0 && write(intr_pipe.writefd, "1", 1) < 0) {
			rte_spinlock_unlock(&intr_lock);
			return -EPIPE;
		}

		rte_spinlock_unlock(&intr_lock);
	}

	return 0;
}
```

## UDP Example Implementation

Below is an example if using UDP to transmit `timespec ts_msg;` between two ports called `#define PORT_PING   10000` and `#define PORT_PONG   10001`.

```cpp
#define PORT_PING   10000
#define PORT_PONG   10001
#define IP_PING_PONG     "127.0.0.1"

struct timespec ts, ts_msg, ts_now;

int main(int argc, char *argv[]) {

    // Initialize UDPDK
    int initRet = udpdk_init(argc, argv);

    struct sockaddr_in servaddr, destaddr;

    // Create a socket
    int sock = udpdk_socket(AF_INET, SOCK_DGRAM, 0);

    // Bind it
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = INADDR_ANY;
    if (strcmp("PING" , argv[1]))
      servaddr.sin_port = htons(PORT_PING);
    else
      servaddr.sin_port = htons(PORT_PONG);
    udpdk_bind(sock, (const struct sockaddr *)&servaddr, sizeof(servaddr));

    destaddr.sin_family = AF_INET;
    destaddr.sin_addr.s_addr = inet_addr(IP_PING_PONG);
    if (strcmp("PONG" , argv[1]))
      destaddr.sin_port = htons(PORT_PING);
    else
      destaddr.sin_port = htons(PORT_PONG);

    while (app_alive) {
        udpdk_sendto(sock, (void *)&ts, sizeof(struct timespec), 0,
                    (const struct sockaddr *) &destaddr, sizeof(destaddr));

        int n = udpdk_recvfrom(sock, (void *)&ts_msg, sizeof(struct timespec), 0, NULL, NULL);
    }
    return 0;
}
```

The init work is
* reserve and allocate mem for packet buffer
* check and config device port
* queue and ring buffer setup

```cpp
int udpdk_init(int argc, char *argv[])
{
    rte_eal_init(primary_argc, (char **)primary_argv);

    // Determine how much memory is needed (pool size + bitfield of free elems + variables) to reserve mem
    unsigned mem_needed = sizeof(struct allocator) + (size / 8 + 1);
    mem_needed = (mem_needed + elem_size - 1) / elem_size;  // align
    mem_needed += (size * elem_size);
    extern const void *udpdk_list_t_alloc = rte_memzone_reserve("udpdk_list_t_alloc", mem_needed, rte_socket_id(), 0);
    extern const void *udpdk_list_node_t_alloc = rte_memzone_reserve("udpdk_list_node_t_alloc", mem_needed, rte_socket_id(), 0);
    extern const void *udpdk_list_iterator_t_alloc = rte_memzone_reserve("udpdk_list_node_t_alloc", mem_needed, rte_socket_id(), 0);

    // launch mem pool for rx and tx
    const int socket = rte_socket_id();
    struct rte_mempool *rx_pktmbuf_pool = rte_pktmbuf_pool_create(PKTMBUF_POOL_RX_NAME, num_mbufs, MBUF_CACHE_SIZE, 0,
            RTE_MBUF_DEFAULT_BUF_SIZE, socket);
    struct rte_mempool *tx_pktmbuf_pool = rte_pktmbuf_pool_create(PKTMBUF_POOL_TX_NAME, num_mbufs, MBUF_CACHE_SIZE, 0,
            RTE_MBUF_DEFAULT_BUF_SIZE, socket);

    // given the valid dpdk-supported device port, load the config
    struct rte_eth_dev_info dev_info;
    if (rte_eth_dev_is_valid_port(port_num)) {
        rte_eth_dev_info_get(port_num, &dev_info);

        const struct rte_eth_conf port_conf = {
            .rxmode = {
                .mq_mode = ETH_MQ_RX_RSS,
                .max_rx_pkt_len = RTE_MIN(JUMBO_FRAME_MAX_SIZE, dev_info.max_rx_pktlen),
                .split_hdr_size = 0,
                .offloads = (DEV_RX_OFFLOAD_CHECKSUM |
                            DEV_RX_OFFLOAD_SCATTER |
                            DEV_RX_OFFLOAD_JUMBO_FRAME),
            },
            .txmode = {
                .offloads = DEV_TX_OFFLOAD_MULTI_SEGS,
            }
        };

        // Configure mode and number of rings
        rte_eth_dev_configure(port_num, rx_rings, tx_rings, &port_conf);
        rte_eth_dev_adjust_nb_rx_tx_desc(port_num, &rx_ring_size, &tx_ring_size);
        rte_eth_rx_queue_setup(port_num, q, rx_ring_size,
                rte_eth_dev_socket_id(port_num), NULL, rx_pktmbuf_pool);
        rte_eth_tx_queue_setup(port_num, q, tx_ring_size,
                rte_eth_dev_socket_id(port_num), NULL);
        // Promiscuous mode or promisc mode is a feature that makes the ethernet card pass all traffic it received to the kernel.
        rte_eth_promiscuous_enable(port_num);
        rte_eth_dev_start(port_num);

        /* Initialize IPC channel for the synchonization between app and poller processes */
        struct rte_ring *    ipc_app_to_pol = rte_ring_create("IPC_channel_app_to_pol", 1, rte_socket_id(), RING_F_SP_ENQ | RING_F_SC_DEQ);
        struct rte_ring *    ipc_pol_to_app = rte_ring_create("IPC_channel_pol_to_app", 1, rte_socket_id(), RING_F_SP_ENQ | RING_F_SC_DEQ);
        ipc_msg_pool = rte_mempool_create("IPC_msg_pool", 5, 64, 0, 0, NULL, NULL, NULL, NULL, rte_socket_id(), 0);

        /* Initialize a shared memory region to contain descriptors for the exchange slots */
        struct rte_memzone *exch_zone = rte_memzone_reserve("UDPDK_exchange_desc", sizeof(*exch_zone_desc), rte_socket_id(), 0);

        /* Initialize a shared memory region to store the L4 switching table */
        const struct rte_memzone *mz = rte_memzone_reserve("UDPDK_btable", UDP_MAX_PORT * sizeof(struct udpdk_list_t *), rte_socket_id(), 0);

        /* Initialize (statically) the slots to exchange packets between the application and the poller */
        exch_slots = rte_malloc(EXCH_SLOTS_NAME, sizeof(*exch_slots) * NUM_SOCKETS_MAX, 0);

        // Create a rte_ring for each RX and TX slot
        for (i = 0; i < NUM_SOCKETS_MAX; i++) {
            char *q_name = get_exch_ring_name(i, EXCH_RING_RX);
            exch_slots[i].rx_q = rte_ring_create(q_name, EXCH_RING_SIZE, socket_id, RING_F_SP_ENQ | RING_F_SC_DEQ);
            char *q_name = get_exch_ring_name(i, EXCH_RING_TX);
            exch_slots[i].tx_q = rte_ring_create(q_name, EXCH_RING_SIZE, socket_id, RING_F_SP_ENQ | RING_F_SC_DEQ);
            if (exch_slots[i].rx_q == NULL || exch_slots[i].tx_q == NULL) {
                RTE_LOG(ERR, INIT, "Cannot create exchange RX/TX exchange rings (index %d)\n", i);
                return -1;
            }
        }
    }
    return 0;
}
```

Socket preparation work is 
* set up `exch_zone_desc`, whose slot is to describe a socket
* bind an available socket to the socket
```cpp
/* Descriptor of a socket (current state and options) */
struct exch_slot_info {
    int used;       // used by an open socket
    int bound;      // used by a socket that did 'bind'
    int sockfd;     // NOTE: redundant atm because it matches the slot index in the current impl
    int udp_port;   // UDP port associated to the socket (only if bound)
    struct in_addr ip_addr;     // IPv4 address associated to the socket (only if bound)
    int so_options; // socket options
} __rte_cache_aligned;

/* Descriptor of the zone in shared memory where packets are exchanged between app and poller */
struct exch_zone_info {
    uint64_t n_zones_active;
    struct exch_slot_info slots[NUM_SOCKETS_MAX];
} exch_zone_desc;

int udpdk_socket(int domain, int type, int protocol) { 
    // Allocate a free sock_id
    int sock_id;
    for (sock_id = 0; sock_id < NUM_SOCKETS_MAX; sock_id++) {
        if (!exch_zone_desc->slots[sock_id].used) {
            exch_zone_desc->slots[sock_id].used = 1;
            exch_zone_desc->slots[sock_id].bound = 0;
            exch_zone_desc->slots[sock_id].sockfd = sock_id;
            exch_zone_desc->slots[sock_id].so_options = 0;
            break;
        }
    }
    return sock_id;
}
int sock = udpdk_socket(AF_INET, SOCK_DGRAM, 0) ;

// Bind it
struct sockaddr_in servaddr;
memset(&servaddr, 0, sizeof(servaddr));
servaddr.sin_family = AF_INET;
servaddr.sin_addr.s_addr = INADDR_ANY;
servaddr.sin_port = htons(PORT_PING);

int udpdk_bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    
    // Allocate and setup a new bind_info element
    b = (struct bind_info *)udpdk_shmalloc(bind_info_alloc);
    b->sockfd = s;
    b->ip_addr = ip;
    b->reuse_addr = opts & SO_REUSEADDR;
    b->reuse_port = opts & SO_REUSEPORT;
    b->closed = false;

    // Mark the slot as bound, and store the corresponding IP and port
    exch_zone_desc->slots[sockfd].bound = 1;
    exch_zone_desc->slots[sockfd].udp_port = (int)port;
    exch_zone_desc->slots[sockfd].ip_addr = addr_in->sin_addr;
}
udpdk_bind(sock, (const struct sockaddr *)&servaddr, sizeof(servaddr));
```

The sending work is
* first do some checking is the socket is valid, testing by `unlikely`
* reserve and allocate `tx` size of memory
* initialize the Ethernet header `eth_hdr = rte_pktmbuf_mtod(pkt, struct rte_ether_hdr *);`
* initialize the IP header `ip_hdr = (struct rte_ipv4_hdr *)(eth_hdr + 1);`
* (optional) fill other DPDK metadata indicating sizes/offsets of diff fields
* write payload `udp_data = (void *)(udp_hdr + 1);`
* put the packet in the `tx_ring` by `rte_ring_enqueue(exch_slots[sockfd].tx_q, (void *)pkt)`

```cpp
// Convert a 64-bit value from CPU order to little endian
#define rte_bswap16(x) ((uint16_t)(__builtin_constant_p(x) ?		\
				   rte_constant_bswap16(x) :		\
				   rte_arch_bswap16(x)))
#define rte_cpu_to_be_16(x) rte_bswap16(x)

ssize_t udpdk_sendto(int sockfd, const void *buf, size_t len, int flags,
                     const struct sockaddr *dest_addr, socklen_t addrlen)
{
    struct rte_mbuf *pkt;
    struct rte_ether_hdr *eth_hdr;
    struct rte_ipv4_hdr *ip_hdr;
    struct rte_udp_hdr *udp_hdr;
    void *udp_data;
    const struct sockaddr_in *dest_addr_in = (struct sockaddr_in *)dest_addr;

    ... // some validation checking, such as if (unlikely( socket is out of service )) {...}

    // Allocate one mbuf for the packet (will be freed when effectively sent)
    pkt = rte_pktmbuf_alloc(tx_pktmbuf_pool);

    // Initialize the Ethernet header
    eth_hdr = rte_pktmbuf_mtod(pkt, struct rte_ether_hdr *);
    rte_ether_addr_copy(&config.src_mac_addr, &eth_hdr->s_addr);
    rte_ether_addr_copy(&config.dst_mac_addr, &eth_hdr->d_addr);
    eth_hdr->ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4);

    // Initialize the IP header
    ip_hdr = (struct rte_ipv4_hdr *)(eth_hdr + 1);
    memset(ip_hdr, 0, sizeof(*ip_hdr));
    ip_hdr->version_ihl = IP_VHL_DEF;
    ip_hdr->type_of_service = 0;
    ip_hdr->fragment_offset = 0;
    ip_hdr->time_to_live = IP_DEFTTL;
    ip_hdr->next_proto_id = IPPROTO_UDP;
    ip_hdr->packet_id = 0;
    if ((exch_zone_desc->slots[sockfd].bound)
            && (exch_zone_desc->slots[sockfd].ip_addr.s_addr != INADDR_ANY)) {
        ip_hdr->src_addr = exch_zone_desc->slots[sockfd].ip_addr.s_addr;
    } else {
        ip_hdr->src_addr = config.src_ip_addr.s_addr;
    }
    ip_hdr->dst_addr = dest_addr_in->sin_addr.s_addr;
    ip_hdr->total_length = rte_cpu_to_be_16(len + sizeof(*ip_hdr) + sizeof(*udp_hdr));
    ip_hdr->hdr_checksum = rte_ipv4_cksum(ip_hdr);

    // Initialize the UDP header
    udp_hdr = (struct rte_udp_hdr *)(ip_hdr + 1);
    udp_hdr->src_port = exch_zone_desc->slots[sockfd].udp_port;
    udp_hdr->dst_port = dest_addr_in->sin_port;
    udp_hdr->dgram_cksum = 0;   // UDP checksum is optional
    udp_hdr->dgram_len = rte_cpu_to_be_16(len + sizeof(*udp_hdr));

    // Fill other DPDK metadata
    pkt->packet_type = RTE_PTYPE_L2_ETHER | RTE_PTYPE_L3_IPV4 | RTE_PTYPE_L4_UDP;
    pkt->pkt_len = len + sizeof(*eth_hdr) + sizeof(*ip_hdr) + sizeof(*udp_hdr);
    pkt->data_len = pkt->pkt_len;
    pkt->l2_len = sizeof(struct rte_ether_hdr);
    pkt->l3_len = sizeof(struct rte_ipv4_hdr);
    pkt->l4_len = sizeof(struct rte_udp_hdr);

    // Write payload
    udp_data = (void *)(udp_hdr + 1);
    rte_memcpy(udp_data, buf, len);

    // Put the packet in the tx_ring
    if (rte_ring_enqueue(exch_slots[sockfd].tx_q, (void *)pkt) < 0) {
        RTE_LOG(ERR, SYSCALL, "Sendto failed to put packet in the TX ring\n  Total: %d  Free: %d\n",
                rte_ring_count(exch_slots[sockfd].tx_q), rte_ring_free_count(exch_slots[sockfd].tx_q));
        errno = ENOBUFS;
        rte_pktmbuf_free(pkt);
        return -1;
    }

    return len;
}
```

the recv work is
* conduct some checkings such as socket availablility
* dequeue from buffer ring (enqueue action is one packet by one packet, dequeue can be multiple at once to `(void **)&pkt`)
* Given a sequence no. 1 packet, extract headers
  * ethernet header
  * ip header
  * udp header
* extract apyload by byte offsets.

Once `rte_ring_dequeue` is run, one packet `(void **)&pkt` is dequeued.
There is no copy within the dequeue process, but simply transfer of pointer ownership.

In DPDK, `(void **)&pkt` resides in UIO-supported netcard, and the data should be copied to `buf` by `rte_memcpy`.

```cpp
ssize_t udpdk_recvfrom(int sockfd, void *buf, size_t len, int flags,
                       struct sockaddr *src_addr, socklen_t *addrlen)
{
    int ret = -1;
    struct rte_mbuf *pkt = NULL;
    struct rte_mbuf *seg = NULL;
    uint32_t seg_len;           // number of bytes of payload in this segment
    uint32_t eff_len;           // number of bytes to read from this segment
    uint32_t eff_addrlen;
    uint32_t bytes_left = len;
    uint16_t dgram_payl_len;    // UDP payload len, inferred from UDP header
    unsigned nb_segs;
    unsigned offset_payload;
    struct rte_ether_hdr *eth_hdr;
    struct rte_ipv4_hdr *ip_hdr;
    struct rte_udp_hdr *udp_hdr;

    // Validate the arguments
    ...

    // Dequeue one packet (busy wait until one is available)
    while (ret < 0 ) {
        ret = rte_ring_dequeue(exch_slots[sockfd].rx_q, (void **)&pkt);
    }

    // Get some useful pointers to headers and data
    nb_segs = pkt->nb_segs; 
    eth_hdr = rte_pktmbuf_mtod(pkt, struct rte_ether_hdr *);
    ip_hdr = (struct rte_ipv4_hdr *)(eth_hdr + 1);
    udp_hdr = (struct rte_udp_hdr *)(ip_hdr + 1);
    dgram_payl_len = rte_be_to_cpu_16(udp_hdr->dgram_len) - sizeof(struct rte_udp_hdr);

    // Write source address (or part of it if addrlen is too short)
    if (src_addr != NULL) {
        ...
    }

    seg = pkt;
    for (int s = 0; s < nb_segs; s++) {
        // The the first segment includes eth + ipv4 + udp headers before the payload
        offset_payload = (s == 0) ?
                sizeof(struct rte_ether_hdr) + sizeof(struct rte_ipv4_hdr) + sizeof(struct rte_udp_hdr) : 0;
        // Find how many bytes of data are in this segment
        seg_len = seg->data_len - offset_payload;
        if ((s == 0) && (seg_len > dgram_payl_len)) {
            // for very small packets, Ethernet payload is padded to 46 bytes
            seg_len = dgram_payl_len;
        }
        // The amount of data to copy is the minimum between this segment length and the remaining requested bytes
        if (seg_len < bytes_left) {
            eff_len = seg_len;
        } else {
            eff_len = bytes_left;
        }
        // Copy payload into buffer
        rte_memcpy(buf, rte_pktmbuf_mtod(seg, void *) + offset_payload, eff_len);
        // Adjust pointers and counters
        buf += eff_len;
        bytes_left -= eff_len;
        seg = seg->next;
        if (bytes_left == 0) {
            break;
        }
    }
    // Free the mbuf (with all the chained segments)
    rte_pktmbuf_free(pkt);

    // Return how many bytes read
    return len - bytes_left;
}
```