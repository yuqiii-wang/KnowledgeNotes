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

`struct kevent` is used for kernel event notification mechanism

```cpp
struct kevent {
    uintptr_t       ident;          /* identifier for this event */
    int16_t         filter;         /* filter for event，EVFILT_READ，EVFILT_WRITE，EVFILT_TIMER */
    uint16_t        flags;          /* general flags EV_ADD，EV_ENABLE， EV_DELETE */
    uint32_t        fflags;         /* filter-specific flags */
    intptr_t        data;           /* filter-specific data */
    void            *udata;         /* opaque user data identifier, can carry user defined data pointed to by a pointer */
};

static void *
eal_intr_thread_main(void *arg __rte_unused)
{
	struct kevent events[MAX_INTR_EVENTS];
	int nfds;

	/* host thread, never break out */
	for (;;) {
		/* do not change anything, just wait */
		nfds = kevent(kq, NULL, 0, events, MAX_INTR_EVENTS, NULL);

		/* kevent fail */
		if (nfds < 0) {
			if (errno == EINTR)
				continue;
			RTE_LOG(ERR, EAL,
				"kevent returns with fail\n");
			break;
		}
		/* kevent timeout, will never happen here */
		else if (nfds == 0)
			continue;

		/* kevent has at least one fd ready to read */
		eal_intr_process_interrupts(events, nfds);
	}
	close(kq);
	kq = -1;
	return NULL;
}
```

```cpp

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