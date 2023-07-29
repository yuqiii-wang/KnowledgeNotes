# Useful DPDK APIs

## `rte_pktmbuf_mtod`

```cpp
/**
 * A macro that points to an offset into the data in the mbuf.
 *
 * The returned pointer is cast to type t. Before using this
 * function, the user must ensure that the first segment is large
 * enough to accommodate its data.
 *
 * @param m
 *   The packet mbuf.
 * @param o
 *   The offset into the mbuf data.
 * @param t
 *   The type to cast the result into.
 */
#define rte_pktmbuf_mtod_offset(m, t, o)	\
	((t)(void *)((char *)(m)->buf_addr + (m)->data_off + (o)))

/**
 * A macro that points to the start of the data in the mbuf.
 *
 * The returned pointer is cast to type t. Before using this
 * function, the user must ensure that the first segment is large
 * enough to accommodate its data.
 *
 * @param m
 *   The packet mbuf.
 * @param t
 *   The type to cast the result into.
 */
#define rte_pktmbuf_mtod(m, t) rte_pktmbuf_mtod_offset(m, t, 0)
```

## `rte_mbuf`

A generic mem buffer containing a packet mbuf.

Data stored in `struct rte_mempool *pool;`, and next packet is pointed to `struct rte_mbuf *next;`.

The *cache aligned* means:
* CPU caches transfer data from and to main memory in chunks called cache lines
* A typical size for this seems to be 64 bytes.
* `rte_mbuf` has two cache lines: `cacheline0` and `cacheline1`

By introducing padding into the structure to inflate it to 64 bytes, 
it is guaranteed that no two such data structures end up in the same cache line, 
and the processes that access them are not blocked more that absolutely necessary.

Data that are frequently accessed by different execution threads are not located close enough in memory 
but they can end up in the same cache line.

```cpp
/**
 * The generic rte_mbuf, containing a packet mbuf.
 */
struct rte_mbuf {
	RTE_MARKER cacheline0;

	void *buf_addr;           /**< Virtual address of segment buffer. */

    /**
	 * Next segment of scattered packet.
	 * This field is valid when physical address field is undefined.
	 * Otherwise next pointer in the second cache line will be used.
	 */
	struct rte_mbuf *next;

    /* next 8 bytes are initialised on RX descriptor rearm */
	RTE_MARKER64 rearm_data;
	uint16_t data_off;

	/**
	 * Reference counter. Its size should at least equal to the size
	 * of port field (16 bits), to support zero-copy broadcast.
	 * It should only be accessed using the following functions:
	 * rte_mbuf_refcnt_update(), rte_mbuf_refcnt_read(), and
	 * rte_mbuf_refcnt_set(). The functionality of these functions (atomic,
	 * or non-atomic) is controlled by the RTE_MBUF_REFCNT_ATOMIC flag.
	 */
	uint16_t refcnt;

	/**
	 * Number of segments. Only valid for the first segment of an mbuf
	 * chain.
	 */
	uint16_t nb_segs;

	/** Input port (16 bits to support more than 256 virtual ports).
	 * The event eth Tx adapter uses this field to specify the output port.
	 */
	uint16_t port;

	uint64_t ol_flags;        /**< Offload features. */

	/* remaining bytes are set on RX when pulling packet from descriptor */
	RTE_MARKER rx_descriptor_fields1;

    /**
	 * Number of segments. Only valid for the first segment of an mbuf
	 * chain.
	 */
	uint16_t nb_segs;

	/*
	 * The packet type, which is the combination of outer/inner L2, L3, L4
	 * and tunnel types. The packet_type is about data really present in the
	 * mbuf. Example: if vlan stripping is enabled, a received vlan packet
	 * would have RTE_PTYPE_L2_ETHER and not RTE_PTYPE_L2_VLAN because the
	 * vlan is stripped from the data.
	 */
	RTE_STD_C11
	union {
		uint32_t packet_type; /**< L2/L3/L4 and tunnel information. */
		__extension__
		struct {
			uint8_t l2_type:4;   /**< (Outer) L2 type. */
			uint8_t l3_type:4;   /**< (Outer) L3 type. */
			uint8_t l4_type:4;   /**< (Outer) L4 type. */
			uint8_t tun_type:4;  /**< Tunnel type. */
			RTE_STD_C11
			union {
				uint8_t inner_esp_next_proto;
				/**< ESP next protocol type, valid if
				 * RTE_PTYPE_TUNNEL_ESP tunnel type is set
				 * on both Tx and Rx.
				 */
				__extension__
				struct {
					uint8_t inner_l2_type:4;
					/**< Inner L2 type. */
					uint8_t inner_l3_type:4;
					/**< Inner L3 type. */
				};
			};
			uint8_t inner_l4_type:4; /**< Inner L4 type. */
		};
	};

	uint32_t pkt_len;         /**< Total pkt len: sum of all segments. */
	uint16_t data_len;        /**< Amount of data in segment buffer. */
	/** VLAN TCI (CPU order), valid if RTE_MBUF_F_RX_VLAN is set. */
	uint16_t vlan_tci;

    ... // 

    /** Outer VLAN TCI (CPU order), valid if RTE_MBUF_F_RX_QINQ is set. */
	uint16_t vlan_tci_outer;

	uint16_t buf_len;         /**< Length of segment buffer. */

	struct rte_mempool *pool; /**< Pool from which mbuf was allocated. */

	/* second cache line - fields only used in slow path or on TX */
	RTE_MARKER cacheline1 __rte_cache_min_aligned;

    ... // uint64_t dynfield2;
}
```

Copy by memory aligned.

```cpp
static __rte_always_inline void *
rte_memcpy_aligned(void *dst, const void *src, size_t n)
{
	void *ret = dst;

    ... // if (n <= 16) { ... }

    ... // if (n <= 32) { ... }

	/* Copy 32 < size <= 64 bytes */
	if (n <= 64) {
		rte_mov32((uint8_t *)dst, (const uint8_t *)src);
		rte_mov32((uint8_t *)dst - 32 + n,
				(const uint8_t *)src - 32 + n);

		return ret;
	}

	/* Copy 64 bytes blocks */
	for (; n > 64; n -= 64) {
		rte_mov64((uint8_t *)dst, (const uint8_t *)src);
		dst = (uint8_t *)dst + 64;
		src = (const uint8_t *)src + 64;
	}

	/* Copy whatever left */
	rte_mov64((uint8_t *)dst - 64 + n,
			(const uint8_t *)src - 64 + n);

	return ret;
}

static __rte_always_inline void *
rte_memcpy(void *dst, const void *src, size_t n)
{
	if (!(((uintptr_t)dst | (uintptr_t)src) & ALIGNMENT_MASK))
		return rte_memcpy_aligned(dst, src, n);
	else
		return rte_memcpy_generic(dst, src, n);
}
```

## `rte_eth_rx_queue_setup`

Tx has similar setup process as rx; here uses rx as an example.

`rte_eth_rx_queue_setup` first has done some dev UIO support availability checking, 
then by `*dev->dev_ops->rx_queue_setup` that calls `rte_zmalloc_socket` to start mem allocation.
Socket here refers to NUMA node, not network socket.

It detects if `rte_eal_has_hugepages(...)` is true, then allocates memory by `alloc_more_mem_on_socket(...)`; 
if false, allocate memory by ordinary `heap_alloc(...)`.



```cpp
int
rte_eth_rx_queue_setup(uint16_t port_id, uint16_t rx_queue_id,
		       uint16_t nb_rx_desc, unsigned int socket_id,
		       const struct rte_eth_rxconf *rx_conf,
		       struct rte_mempool *mp)
{
	int ret;
	uint64_t rx_offloads;
	uint32_t mbp_buf_size = UINT32_MAX;
	struct rte_eth_dev *dev;
	struct rte_eth_dev_info dev_info;
	struct rte_eth_rxconf local_conf;

	ret = rte_eth_dev_info_get(port_id, &dev_info);
	if (ret != 0)
		return ret;

	rx_offloads = dev->data->dev_conf.rxmode.offloads;
	if (rx_conf != NULL)
		rx_offloads |= rx_conf->offloads;

	/* Ensure that we have one and only one source of Rx buffers */
    // return err if there are multiple mempools for rx
	...

    /* Single pool configuration launch. */
    ret = rte_eth_check_rx_mempool(mp, RTE_PKTMBUF_HEADROOM,
                    dev_info.min_rx_bufsize);
    // other rx mempool checks
    ...

    // set up the rx queue
	ret = (*dev->dev_ops->rx_queue_setup)(dev, rx_queue_id, nb_rx_desc,
					      socket_id, &local_conf, mp);
	if (!ret) {
		if (!dev->data->min_rx_buf_size ||
		    dev->data->min_rx_buf_size > mbp_buf_size)
			dev->data->min_rx_buf_size = mbp_buf_size;
	}

	rte_ethdev_trace_rxq_setup(port_id, rx_queue_id, nb_rx_desc, mp,
		rx_conf, ret);
	return eth_err(port_id, ret);
}


static const struct eth_dev_ops virtual_ethdev_default_dev_ops = {
	... // other bindings
	.rx_queue_setup = virtual_ethdev_rx_queue_setup_success,
	.tx_queue_setup = virtual_ethdev_tx_queue_setup_success,
	... // other bindings
};

static int
virtual_ethdev_rx_queue_setup_success(struct rte_eth_dev *dev,
		uint16_t rx_queue_id, uint16_t nb_rx_desc __rte_unused,
		unsigned int socket_id,
		const struct rte_eth_rxconf *rx_conf __rte_unused,
		struct rte_mempool *mb_pool __rte_unused)
{
	struct virtual_ethdev_queue *rx_q;

	rx_q = (struct virtual_ethdev_queue *)rte_zmalloc_socket(NULL,
			sizeof(struct virtual_ethdev_queue), 0, socket_id);

	if (rx_q == NULL)
		return -1;

	rx_q->port_id = dev->data->port_id;
	rx_q->queue_id = rx_queue_id;

	dev->data->rx_queues[rx_queue_id] = rx_q;

	return 0;
}

// `rte_zmalloc_socket` calls `malloc_socket` to allocate mem on heap.

static void *
malloc_socket(const char *type, size_t size, unsigned int align,
		int socket_arg, const bool trace_ena)
{
	void *ptr;

	/* return NULL if size is 0 or alignment is not power-of-2 */
	if (size == 0 || (align && !rte_is_power_of_2(align)))
		return NULL;

	/* if there are no hugepages and if we are not allocating from an
	 * external heap, use memory from any socket available. checking for
	 * socket being external may return -1 in case of invalid socket, but
	 * that's OK - if there are no hugepages, it doesn't matter.
	 */
	if (rte_malloc_heap_socket_is_external(socket_arg) != 1 &&
				!rte_eal_has_hugepages())
		socket_arg = SOCKET_ID_ANY;

	ptr = malloc_heap_alloc(type, size, socket_arg, 0,
			align == 0 ? 1 : align, 0, false);

	if (trace_ena)
		rte_eal_trace_mem_malloc(type, size, align, socket_arg, ptr);
	return ptr;
}
```

### Heap Memory Allocation

From `rte_eal_memory_init(...)`, DPDK sets up continuous virtual mem on heap.
`memtype` is used to tag different NUMA nodes as well as huge page mem, of which `memtype` has mmultiple `rte_memseg_list`, that contains many `rte_memseg`.
Each `rte_memseg` corresponds to a DPDK huge page.

The max reserved heap mem is 512G set by `RTE_MAX_MEM_MB`.

```cpp
/* init memory subsystem */
int
rte_eal_memory_init(void)
{
	const struct internal_config *internal_conf =
		eal_get_internal_configuration();
	int retval;

	RTE_LOG(DEBUG, EAL, "Setting up physically contiguous memory...\n");

	if (rte_eal_memseg_init() < 0)
		goto fail;

	if (eal_memalloc_init() < 0)
		goto fail;

	retval = rte_eal_process_type() == RTE_PROC_PRIMARY ?
			rte_eal_hugepage_init() :
			rte_eal_hugepage_attach();
	if (retval < 0)
		goto fail;

	if (internal_conf->no_shconf == 0 && rte_eal_memdevice_init() < 0)
		goto fail;

	return 0;
fail:
	return -1;
}
```

Then, DPDK maps the virtual mem to physical mem.

DPDK then inits heap mem alloc.
It first checks on socket/numa, then registers mem to heap.

`register_mp_requests(...)` calls `rte_mp_action_register` registering mem.

In detail in `rte_mp_action_register`, having implemented lock `pthread_mutex_lock(&mp_mutex_action);`, perform `TAILQ_INSERT_TAIL(...)` inserting the mem.

```cpp
int
rte_eal_malloc_heap_init(void)
{
	struct rte_mem_config *mcfg = rte_eal_get_configuration()->mem_config;
	unsigned int i;
	const struct internal_config *internal_conf =
		eal_get_internal_configuration();

	if (rte_eal_process_type() == RTE_PROC_PRIMARY) {
		/* assign min socket ID to external heaps */
		mcfg->next_socket_id = EXTERNAL_HEAP_MIN_SOCKET_ID;

		/* assign names to default DPDK heaps */
		for (i = 0; i < rte_socket_count(); i++) {
			struct malloc_heap *heap = &mcfg->malloc_heaps[i];
            ...
			heap->socket_id = socket_id;
		}
	}

    register_mp_requests();

	return 0;
}

int
rte_mp_action_register(const char *name, rte_mp_t action)
{
	struct action_entry *entry;
	const struct internal_config *internal_conf =
		eal_get_internal_configuration();

	... // define `entry`

	pthread_mutex_lock(&mp_mutex_action);
	if (find_action_entry_by_name(name) != NULL) {
		pthread_mutex_unlock(&mp_mutex_action);
		rte_errno = EEXIST;
		free(entry);
		return -1;
	}
	TAILQ_INSERT_TAIL(&action_entry_list, entry, next);
	pthread_mutex_unlock(&mp_mutex_action);
	return 0;
}
```

Mem allocation performed within spin lock/unlock `&(heap->lock)`:
* has huge page mem: `alloc_more_mem_on_socket(...)`
* no huge page mem: `heap_alloc(...)`

Free lists of mem blocks are sorted from small to large by `qsort(...)`, 
and it will try to allocate on small mem chuncks if possible.

The "allocation" behavior is basically done in `find_suitable_element(...)` that iterates free lists finding the available mem chunks then remove them from its belonged free list by `elem = LIST_NEXT(elem, free_list)`.

```cpp
void *
malloc_heap_alloc(const char *type, size_t size, int socket_arg,
		unsigned int flags, size_t align, size_t bound, bool contig)
{
	int socket, heap_id, i;
	void *ret;

    ... // checking on `rte_eal_has_hugepages`, etc.

	/* turn socket ID into heap ID */
	heap_id = malloc_socket_to_heap_id(socket);
	/* if heap id is negative, socket ID was invalid */
	if (heap_id < 0)
		return NULL;

	ret = malloc_heap_alloc_on_heap_id(type, size, heap_id, flags, align,
			bound, contig);
	if (ret != NULL || socket_arg != SOCKET_ID_ANY)
		return ret;

	/* try other heaps. we are only iterating through native DPDK sockets,
	 * so external heaps won't be included.
	 */
    ...

	return NULL;
}


/* this will try lower page sizes first */
static void *
malloc_heap_alloc_on_heap_id(const char *type, size_t size,
		unsigned int heap_id, unsigned int flags, size_t align,
		size_t bound, bool contig)
{
	struct rte_mem_config *mcfg = rte_eal_get_configuration()->mem_config;
	struct malloc_heap *heap = &mcfg->malloc_heaps[heap_id];
	unsigned int size_flags = flags & ~RTE_MEMZONE_SIZE_HINT_ONLY;
	int socket_id;
	void *ret;
	const struct internal_config *internal_conf =
		eal_get_internal_configuration();

	rte_spinlock_lock(&(heap->lock));

	align = align == 0 ? 1 : align;

	/* for legacy mode, try once and with all flags */
	... // legacy mode compatibility

	/*
	 * we do not pass the size hint here, because even if allocation fails,
	 * we may still be able to allocate memory from appropriate page sizes,
	 * we just need to request more memory first.
	 */

	socket_id = rte_socket_id_by_idx(heap_id);
	
    ... // goto `alloc_unlock` if allocation failed

    // hugepage mem alloc first, if failed, try ordinary alloc by `heap_alloc`
	if (!alloc_more_mem_on_socket(heap, size, socket_id, flags, align,
			bound, contig)) {
		ret = heap_alloc(heap, type, size, flags, align, bound, contig);

		/* this should have succeeded */
		if (ret == NULL)
			RTE_LOG(ERR, EAL, "Error allocating from heap\n");
	}
alloc_unlock:
	rte_spinlock_unlock(&(heap->lock));
	return ret;
}


static int
alloc_more_mem_on_socket(struct malloc_heap *heap, size_t size, int socket,
		unsigned int flags, size_t align, size_t bound, bool contig)
{
	struct rte_mem_config *mcfg = rte_eal_get_configuration()->mem_config;
	struct rte_memseg_list *requested_msls[RTE_MAX_MEMSEG_LISTS];
	struct rte_memseg_list *other_msls[RTE_MAX_MEMSEG_LISTS];
	uint64_t requested_pg_sz[RTE_MAX_MEMSEG_LISTS];
	uint64_t other_pg_sz[RTE_MAX_MEMSEG_LISTS];
	uint64_t prev_pg_sz;
	int i, n_other_msls, n_other_pg_sz, n_requested_msls, n_requested_pg_sz;
	bool size_hint = (flags & RTE_MEMZONE_SIZE_HINT_ONLY) > 0;
	unsigned int size_flags = flags & ~RTE_MEMZONE_SIZE_HINT_ONLY;
	void *ret;

	memset(requested_msls, 0, sizeof(requested_msls));
	memset(other_msls, 0, sizeof(other_msls));
	memset(requested_pg_sz, 0, sizeof(requested_pg_sz));
	memset(other_pg_sz, 0, sizeof(other_pg_sz));

	/*
	 * go through memseg list and take note of all the page sizes available,
	 * and if any of them were specifically requested by the user.
	 */
	n_requested_msls = 0;
	n_other_msls = 0;
	for (i = 0; i < RTE_MAX_MEMSEG_LISTS; i++) {
		struct rte_memseg_list *msl = &mcfg->memsegs[i];

		if (msl->socket_id != socket)
			continue;

		if (msl->base_va == NULL)
			continue;

		/* if pages of specific size were requested */
		if (size_flags != 0 && check_hugepage_sz(size_flags,
				msl->page_sz))
			requested_msls[n_requested_msls++] = msl;
		else if (size_flags == 0 || size_hint)
			other_msls[n_other_msls++] = msl;
	}

	/* sort the lists, smallest first */
	qsort(requested_msls, n_requested_msls, sizeof(requested_msls[0]),
			compare_pagesz);
	qsort(other_msls, n_other_msls, sizeof(other_msls[0]),
			compare_pagesz);

	/* now, extract page sizes we are supposed to try */
	prev_pg_sz = 0;
	n_requested_pg_sz = 0;
	for (i = 0; i < n_requested_msls; i++) {
		uint64_t pg_sz = requested_msls[i]->page_sz;

		if (prev_pg_sz != pg_sz) {
			requested_pg_sz[n_requested_pg_sz++] = pg_sz;
			prev_pg_sz = pg_sz;
		}
	}
    ... // same for other_pg_sz

	/* finally, try allocating memory of specified page sizes, starting from
	 * the smallest sizes
	 */
	for (i = 0; i < n_requested_pg_sz; i++) {
		uint64_t pg_sz = requested_pg_sz[i];

		/*
		 * do not pass the size hint here, as user expects other page
		 * sizes first, before resorting to best effort allocation.
		 */
		if (!try_expand_heap(heap, pg_sz, size, socket, size_flags,
				align, bound, contig))
			return 0;
	}
	if (n_other_pg_sz == 0)
		return -1;

	/* now, check if we can reserve anything with size hint */
	ret = find_suitable_element(heap, size, flags, align, bound, contig);
	if (ret != NULL)
		return 0;

	/*
	 * we still couldn't reserve memory, so try expanding heap with other
	 * page sizes, if there are any
	 */
    ...

	return -1;
}


/*
 * Iterates through the freelist for a heap to find a free element
 * which can store data of the required size and with the requested alignment.
 * If size is 0, find the biggest available elem.
 * Returns null on failure, or pointer to element on success.
 */
static struct malloc_elem *
find_suitable_element(struct malloc_heap *heap, size_t size,
		unsigned int flags, size_t align, size_t bound, bool contig)
{
	size_t idx;
	struct malloc_elem *elem, *alt_elem = NULL;

	for (idx = malloc_elem_free_list_index(size);
			idx < RTE_HEAP_NUM_FREELISTS; idx++) {
		for (elem = LIST_FIRST(&heap->free_head[idx]);
				!!elem; elem = LIST_NEXT(elem, free_list)) {
			if (malloc_elem_can_hold(elem, size, align, bound,
					contig)) {
				if (check_hugepage_sz(flags,
						elem->msl->page_sz))
					return elem;
				if (alt_elem == NULL)
					alt_elem = elem;
			}
		}
	}

	if (flags & RTE_MEMZONE_SIZE_HINT_ONLY)
		return alt_elem;

	return NULL;
}
```

### Memory pool on heap

`rte_mempool` has the advantages:
* Used lock-free ring buffer `rte_ring` to arrange resources
* Provided thread_local-level cache 

```cpp
/* create the mempool */
struct rte_mempool *rte_mempool_create(const char *name, unsigned n, unsigned elt_size,
	unsigned cache_size, unsigned private_data_size,
	rte_mempool_ctor_t *mp_init, void *mp_init_arg,
	rte_mempool_obj_cb_t *obj_init, void *obj_init_arg,
	int socket_id, unsigned flags)
{
	int ret;
	struct rte_mempool *mp;

	mp = rte_mempool_create_empty(name, n, elt_size, cache_size,
		private_data_size, socket_id, flags);
    
    ...
}


/* create an empty mempool */
struct rte_mempool *rte_mempool_create_empty(
    const char *name, unsigned n, unsigned elt_size,
	unsigned cache_size, unsigned private_data_size,
	int socket_id, unsigned flags)
{
	char mz_name[RTE_MEMZONE_NAMESIZE];
	struct rte_mempool_list *mempool_list;
	struct rte_mempool *mp = NULL;
	struct rte_tailq_entry *te = NULL;
	const struct rte_memzone *mz = NULL;
	size_t mempool_size;
	unsigned int mz_flags = RTE_MEMZONE_1GB|RTE_MEMZONE_SIZE_HINT_ONLY;
	struct rte_mempool_objsz objsz;
	unsigned lcore_id;
	int ret;

	/* compilation-time checks */
	...

	mempool_list = RTE_TAILQ_CAST(rte_mempool_tailq.head, rte_mempool_list);

    ... // validation checking and flag setup


	/* calculate mempool object sizes. */
	if (!rte_mempool_calc_obj_size(elt_size, flags, &objsz)) {
		rte_errno = EINVAL;
		return NULL;
	}

	rte_mcfg_mempool_write_lock();

	/*
	 * reserve a memory zone for this mempool: private data is
	 * cache-aligned
	 */
	private_data_size = (private_data_size +
			     RTE_MEMPOOL_ALIGN_MASK) & (~RTE_MEMPOOL_ALIGN_MASK);


	/* try to allocate tailq entry */
	te = rte_zmalloc("MEMPOOL_TAILQ_ENTRY", sizeof(*te), 0);
	if (te == NULL) {
		RTE_LOG(ERR, MEMPOOL, "Cannot allocate tailq entry!\n");
		goto exit_unlock;
	}

	mempool_size = RTE_MEMPOOL_HEADER_SIZE(mp, cache_size);
	mempool_size += private_data_size;
	mempool_size = RTE_ALIGN_CEIL(mempool_size, RTE_MEMPOOL_ALIGN);

	mz = rte_memzone_reserve(mz_name, mempool_size, socket_id, mz_flags);
	if (mz == NULL)
		goto exit_unlock;

	/* init the mempool structure */
	mp = mz->addr;
	memset(mp, 0, RTE_MEMPOOL_HEADER_SIZE(mp, cache_size));
	ret = strlcpy(mp->name, name, sizeof(mp->name));
	if (ret < 0 || ret >= (int)sizeof(mp->name)) {
		rte_errno = ENAMETOOLONG;
		goto exit_unlock;
	}
	mp->mz = mz;
	mp->size = n;
	mp->flags = flags;
	mp->socket_id = socket_id;
	mp->elt_size = objsz.elt_size;
	mp->header_size = objsz.header_size;
	mp->trailer_size = objsz.trailer_size;
	/* Size of default caches, zero means disabled. */
	mp->cache_size = cache_size;
	mp->private_data_size = private_data_size;
	STAILQ_INIT(&mp->elt_list);
	STAILQ_INIT(&mp->mem_list);

	/*
	 * local_cache pointer is set even if cache_size is zero.
	 * The local_cache points to just past the elt_pa[] array.
	 */
	mp->local_cache = (struct rte_mempool_cache *)
		RTE_PTR_ADD(mp, RTE_MEMPOOL_HEADER_SIZE(mp, 0));

	/* Init all default caches. */
	if (cache_size != 0) {
		for (lcore_id = 0; lcore_id < RTE_MAX_LCORE; lcore_id++)
			mempool_cache_init(&mp->local_cache[lcore_id],
					   cache_size);
	}

	te->data = mp;

	rte_mcfg_tailq_write_lock();
	TAILQ_INSERT_TAIL(mempool_list, te, next);
	rte_mcfg_tailq_write_unlock();
	rte_mcfg_mempool_write_unlock();

	rte_mempool_trace_create_empty(name, n, elt_size, cache_size,
		private_data_size, flags, mp);
	return mp;

exit_unlock:
	rte_mcfg_mempool_write_unlock();
	rte_free(te);
	rte_mempool_free(mp);
	return NULL;
}
```

## Ring Buffer

```cpp
/**
 * An RTE ring structure.
 *
 * The producer and the consumer have a head and a tail index. The particularity
 * of these index is that they are not between 0 and size(ring)-1. These indexes
 * are between 0 and 2^32 -1, and we mask their value when we access the ring[]
 * field. Thanks to this assumption, we can do subtractions between 2 index
 * values in a modulo-32bit base: that's why the overflow of the indexes is not
 * a problem.
 */
struct rte_ring {
	char name[RTE_RING_NAMESIZE] __rte_cache_aligned;
	/**< Name of the ring. */
	int flags;               /**< Flags supplied at creation. */
	const struct rte_memzone *memzone;
			/**< Memzone, if any, containing the rte_ring */
	uint32_t size;           /**< Size of ring. */
	uint32_t mask;           /**< Mask (size-1) of ring. */
	uint32_t capacity;       /**< Usable size of ring */

	char pad0 __rte_cache_aligned; /**< empty cache line */

	/** Ring producer status. */
	RTE_STD_C11
	union {
		struct rte_ring_headtail prod;
		struct rte_ring_hts_headtail hts_prod;
		struct rte_ring_rts_headtail rts_prod;
	}  __rte_cache_aligned;

	char pad1 __rte_cache_aligned; /**< empty cache line */

	/** Ring consumer status. */
	RTE_STD_C11
	union {
		struct rte_ring_headtail cons;
		struct rte_ring_hts_headtail hts_cons;
		struct rte_ring_rts_headtail rts_cons;
	}  __rte_cache_aligned;

	char pad2 __rte_cache_aligned; /**< empty cache line */
};
```

```cpp
/* create the ring */
struct rte_ring *
rte_ring_create(const char *name, unsigned count, int socket_id,
		unsigned flags)
{
	char mz_name[RTE_MEMZONE_NAMESIZE];
	struct rte_ring *r;
	struct rte_tailq_entry *te;
	const struct rte_memzone *mz;
	ssize_t ring_size;
	int mz_flags = 0;
	struct rte_ring_list* ring_list = NULL;
	const unsigned int requested_count = count;
	int ret;
    /*首先找到ring_list，rte_ring_tailq是维护所有rte ring队列的，不区分socket*/
	ring_list = RTE_TAILQ_CAST(rte_ring_tailq.head, rte_ring_list);

	/* for an exact size ring, round up from count to a power of two */
	if (flags & RING_F_EXACT_SZ)
		count = rte_align32pow2(count + 1);

	ring_size = rte_ring_get_memsize(count);
	if (ring_size < 0) {
		rte_errno = ring_size;
		return NULL;
	}

	ret = snprintf(mz_name, sizeof(mz_name), "%s%s",
		RTE_RING_MZ_PREFIX, name);
	if (ret < 0 || ret >= (int)sizeof(mz_name)) {
		rte_errno = ENAMETOOLONG;
		return NULL;
	}

    /*分配一个struct rte_tailq_entry *te;结构，在创建完成ring后，挂接这个队列元素到队列中去*/
	te = rte_zmalloc("RING_TAILQ_ENTRY", sizeof(*te), 0);
	if (te == NULL) {
		RTE_LOG(ERR, RING, "Cannot reserve memory for tailq\n");
		rte_errno = ENOMEM;
		return NULL;
	}

	rte_rwlock_write_lock(RTE_EAL_TAILQ_RWLOCK);

	/* reserve a memory zone for this ring. If we can't get rte_config or
	 * we are secondary process, the memzone_reserve function will set
	 * rte_errno for us appropriately - hence no check in this this function 
    */
	mz = rte_memzone_reserve_aligned(mz_name, ring_size, socket_id,
					 mz_flags, __alignof__(*r));
	if (mz != NULL) {
		r = mz->addr;
		/* no need to check return value here, we already checked the
		 * arguments above 队列初始化 */
		rte_ring_init(r, name, requested_count, flags);

		te->data = (void *) r;
		r->memzone = mz;
        /*te挂接到全局尾队列上，便于统一管理*/
		TAILQ_INSERT_TAIL(ring_list, te, next);
	} else {
		r = NULL;
		RTE_LOG(ERR, RING, "Cannot reserve memory\n");
		rte_free(te);
	}
	rte_rwlock_write_unlock(RTE_EAL_TAILQ_RWLOCK);

	return r;
}
```

`RTE_TAILQ_CAST` is used to find 

```cpp
/**
 * Return the first tailq entry cast to the right struct.
 */
#define RTE_TAILQ_CAST(tailq_entry, struct_name) \
	(struct struct_name *)&(tailq_entry)->tailq_head

/* the first and the last elem of the queue */
#define TAILQ_HEAD(name, type)    \ 
struct name {   \
    struct type *tqh_first; /* first element */   \
    struct type **tqh_last; /* addr of last next element */   \
} 

...

struct rte_tailq_entry *te = rte_zmalloc("RING_TAILQ_ENTRY", sizeof(*te), 0);

```