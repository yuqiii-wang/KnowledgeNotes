# TCP recv from NIC

TCP process starts after a packet is passed from the IP layer (further down from NIC (network interface card)).
Notice here that `tcp_v4_rcv` is registered as the process handler.

```cpp
static struct net_protocol tcp_protocol = {
    .early_demux    =    tcp_v4_early_demux,
    .early_demux_handler =  tcp_v4_early_demux,
    .handler    =    tcp_v4_rcv,
    .err_handler    =    tcp_v4_err,
    .no_policy    =    1,
    .netns_ok    =    1,
    .icmp_strict_tag_validation = 1,
};

static int ip_local_deliver_finish(struct net *net, struct sock *sk, struct sk_buff *skb)
{
    // get protocol
    ipprot = rcu_dereference(inet_protos[protocol]);
    if (ipprot) {
        int ret;

        // invoke the protocol registered handler to process this sk_buff
        ret = ipprot->handler(skb);
        if (ret < 0) {
            protocol = -ret;
            goto resubmit;
        }
        __IP_INC_STATS(net, IPSTATS_MIB_INDELIVERS);
    }
}
```

`tcp_v4_rcv` processes a tcp packet, reads info from the packet's control block, commits action on this packet according to the packet's different tcp state, and filters out invalid packets.
`tcp_v4_rcv` handles tcp packets compliant with tcp protocol such as checking packet sequence no and sending `ACK` back to client asking for the next packet. 
```cpp
int tcp_v4_rcv(struct sk_buff *skb)
{
	const struct iphdr *iph;
	const struct tcphdr *th;
	struct sock *sk;
	int ret;
	struct net *net = dev_net(skb->dev);

	if (skb->pkt_type != PACKET_HOST)
		goto discard_it;

	/* Count it even if it's bad */
	TCP_INC_STATS_BH(net, TCP_MIB_INSEGS);

	if (!pskb_may_pull(skb, sizeof(struct tcphdr)))
		goto discard_it;

	th = tcp_hdr(skb);

	if (th->doff < sizeof(struct tcphdr) / 4)
		goto bad_packet;
	if (!pskb_may_pull(skb, th->doff * 4))
		goto discard_it;

	/* An explanation is required here, I think.
	 * Packet length and doff are validated by header prediction,
	 * provided case of th->doff==0 is eliminated.
	 * So, we defer the checks. */

	if (skb_checksum_init(skb, IPPROTO_TCP, inet_compute_pseudo))
		goto csum_error;

	// tcp header and ip header
	th = tcp_hdr(skb);
	iph = ip_hdr(skb);
	/* This is tricky : We move IPCB at its correct location into TCP_SKB_CB()
	 * barrier() makes sure compiler wont play fool^Waliasing games.
	 */
	memmove(&TCP_SKB_CB(skb)->header.h4, IPCB(skb),
		sizeof(struct inet_skb_parm));
	barrier();

    // set tcp skb control info
	TCP_SKB_CB(skb)->seq = ntohl(th->seq);
	TCP_SKB_CB(skb)->end_seq = (TCP_SKB_CB(skb)->seq + th->syn + th->fin +
				    skb->len - th->doff * 4);
	TCP_SKB_CB(skb)->ack_seq = ntohl(th->ack_seq);
	TCP_SKB_CB(skb)->tcp_flags = tcp_flag_byte(th);
	TCP_SKB_CB(skb)->tcp_tw_isn = 0;
	TCP_SKB_CB(skb)->ip_dsfield = ipv4_get_dsfield(iph);
	TCP_SKB_CB(skb)->sacked	 = 0;

lookup:
	sk = __inet_lookup_skb(&tcp_hashinfo, skb, th->source, th->dest);
	if (!sk)
		goto no_tcp_socket;

process:
	if (sk->sk_state == TCP_TIME_WAIT)
		goto do_time_wait;

	if (sk->sk_state == TCP_NEW_SYN_RECV) {
		struct request_sock *req = inet_reqsk(sk);
		struct sock *nsk = NULL;

		sk = req->rsk_listener;
		if (tcp_v4_inbound_md5_hash(sk, skb))
			goto discard_and_relse;
		if (likely(sk->sk_state == TCP_LISTEN)) {
			nsk = tcp_check_req(sk, skb, req, false);
		} else {
			inet_csk_reqsk_queue_drop_and_put(sk, req);
			goto lookup;
		}
		if (!nsk) {
			reqsk_put(req);
			goto discard_it;
		}
		if (nsk == sk) {
			sock_hold(sk);
			reqsk_put(req);
		} else if (tcp_child_process(sk, nsk, skb)) {
			tcp_v4_send_reset(nsk, skb);
			goto discard_it;
		} else {
			return 0;
		}
	}
	if (unlikely(iph->ttl < inet_sk(sk)->min_ttl)) {
		NET_INC_STATS_BH(net, LINUX_MIB_TCPMINTTLDROP);
		goto discard_and_relse;
	}

	if (!xfrm4_policy_check(sk, XFRM_POLICY_IN, skb))
		goto discard_and_relse;

	if (tcp_v4_inbound_md5_hash(sk, skb))
		goto discard_and_relse;

	nf_reset(skb);

	if (sk_filter(sk, skb))
		goto discard_and_relse;

	skb->dev = NULL;

	if (sk->sk_state == TCP_LISTEN) {
		ret = tcp_v4_do_rcv(sk, skb);
		goto put_and_return;
	}

	sk_incoming_cpu_update(sk);

    // `sock_owned_by_user` checks if a user is currently occupying using this socket.
    // if so, that means the user might operate on prequeue or receive queue.
    // coming requests are now placed in backlog. 
    // if backlog is full, other coming requests will be dropped
    // if not used by user, socket attempts to put requests in prequeue 
	bh_lock_sock_nested(sk);
	tcp_sk(sk)->segs_in += max_t(u16, 1, skb_shinfo(skb)->gso_segs);
	ret = 0;
	if (!sock_owned_by_user(sk)) {
		if (!tcp_prequeue(sk, skb))
			ret = tcp_v4_do_rcv(sk, skb);
	} else if (unlikely(sk_add_backlog(sk, skb,
					   sk->sk_rcvbuf + sk->sk_sndbuf))) {
		bh_unlock_sock(sk);
		NET_INC_STATS_BH(net, LINUX_MIB_TCPBACKLOGDROP);
		goto discard_and_relse;
	}
	bh_unlock_sock(sk);

put_and_return:
	sock_put(sk);

	return ret;

no_tcp_socket:
	if (!xfrm4_policy_check(NULL, XFRM_POLICY_IN, skb))
		goto discard_it;

	if (tcp_checksum_complete(skb)) {
csum_error:
		TCP_INC_STATS_BH(net, TCP_MIB_CSUMERRORS);
bad_packet:
		TCP_INC_STATS_BH(net, TCP_MIB_INERRS);
	} else {
		tcp_v4_send_reset(NULL, skb);
	}

discard_it:
	/* Discard frame. */
	kfree_skb(skb);
	return 0;

discard_and_relse:
	sock_put(sk);
	goto discard_it;

do_time_wait:
	if (!xfrm4_policy_check(NULL, XFRM_POLICY_IN, skb)) {
		inet_twsk_put(inet_twsk(sk));
		goto discard_it;
	}

	if (tcp_checksum_complete(skb)) {
		inet_twsk_put(inet_twsk(sk));
		goto csum_error;
	}
	switch (tcp_timewait_state_process(inet_twsk(sk), skb, th)) {
	case TCP_TW_SYN: {
		struct sock *sk2 = inet_lookup_listener(dev_net(skb->dev),
							&tcp_hashinfo,
							iph->saddr, th->source,
							iph->daddr, th->dest,
							inet_iif(skb));
		if (sk2) {
			inet_twsk_deschedule_put(inet_twsk(sk));
			sk = sk2;
			goto process;
		}
		/* Fall through to ACK */
	}
	case TCP_TW_ACK:
		tcp_v4_timewait_ack(sk, skb);
		break;
	case TCP_TW_RST:
		goto no_tcp_socket;
	case TCP_TW_SUCCESS:;
	}
	goto discard_it;
}
```

Fast path choice is for tcp packets satisfying conditions such as tcp sliding window is big and buffer has plenty of space, etc.
```cpp
/* The socket must have it's spinlock held when we get
 * here.
 *
 * We have a potential double-lock case here, so even when
 * doing backlog processing we use the BH locking scheme.
 * This is because we cannot sleep with the original spinlock held.
 */
int tcp_v4_do_rcv(struct sock *sk, struct sk_buff *skb)
{
	struct sock *rsk;
#ifdef CONFIG_TCP_MD5SIG
	/*
	 * We really want to reject the packet as early as possible
	 * if:
	 *  o We're expecting an MD5'd packet and this is no MD5 tcp option
	 *  o There is an MD5 option and we're not expecting one
	 */
	if (tcp_v4_inbound_md5_hash(sk, skb))
		goto discard;
#endif

    // fast path is a dedicated process logic when sk is on `TCP_ESTABLISHED`, will be handled by `tcp_rcv_established()`
	if (sk->sk_state == TCP_ESTABLISHED) { /* Fast path */
		TCP_CHECK_TIMER(sk);
		if (tcp_rcv_established(sk, skb, tcp_hdr(skb), skb->len)) {
			rsk = sk;
			goto reset;
		}
		TCP_CHECK_TIMER(sk);
		return 0;
	}
 
	// validate tcp request header
	if (skb->len < tcp_hdrlen(skb) || tcp_checksum_complete(skb))
		goto csum_err;
 
	// should first establish a connection if the socket is on LISTEN state
	if (sk->sk_state == TCP_LISTEN) {
		struct sock *nsk = tcp_v4_hnd_req(sk, skb);
		if (!nsk)
			goto discard;
 
		if (nsk != sk) {
			if (tcp_child_process(sk, nsk, skb)) {
				rsk = nsk;
				goto reset;
			}
			return 0;
		}
	}
 
    // socket tcps of other states are handled by `tcp_rcv_state_process`
	TCP_CHECK_TIMER(sk);
	if (tcp_rcv_state_process(sk, skb, tcp_hdr(skb), skb->len)) {
		rsk = sk;
		goto reset;
	}
	TCP_CHECK_TIMER(sk);
	return 0;
 
reset:
	tcp_v4_send_reset(rsk, skb);
discard:
	kfree_skb(skb);
	/* Be careful here. If this function gets more complicated and
	 * gcc suffers from register pressure on the x86, sk (in %ebx)
	 * might be destroyed here. This current version compiles correctly,
	 * but you have been warned.
	 */
	return 0;
 
csum_err:
	TCP_INC_STATS_BH(TCP_MIB_INERRS);
	goto discard;
}
```

Inside `tcp_rcv_established`, the data copy is dome by `tcp_copy_to_iovec` which 
```cpp
void tcp_rcv_established(struct sock *sk, struct sk_buff *skb,
             const struct tcphdr *th, unsigned int len)
{
    struct tcp_sock *tp = tcp_sk(sk);

    skb_mstamp_get(&tp->tcp_mstamp);

    // router reset
    if (unlikely(!sk->sk_rx_dst))
        inet_csk(sk)->icsk_af_ops->sk_rx_dst_set(sk, skb);
    /*
     *    Header prediction.
     *    The code loosely follows the one in the famous
     *    "30 instruction TCP receive" Van Jacobson mail.
     *
     *    Van's trick is to deposit buffers into socket queue
     *    on a device interrupt, to call tcp_recv function
     *    on the receive process context and checksum and copy
     *    the buffer to user space. smart...
     *
     *    Our current scheme is not silly either but we take the
     *    extra cost of the net_bh soft interrupt processing...
     *    We do checksum and copy also but from device to kernel.
     */

    tp->rx_opt.saw_tstamp = 0;

    /*    pred_flags is 0xS?10 << 16 + snd_wnd
     *    if header_prediction is to be made
     *    'S' will always be tp->tcp_header_len >> 2
     *    '?' will be 0 for the fast path, otherwise pred_flags is 0 to
     *  turn it off    (when there are holes in the receive
     *     space for instance)
     *    PSH flag is ignored.
     */

    //fast path checking header, such as seq number.
    if ((tcp_flag_word(th) & TCP_HP_BITS) == tp->pred_flags &&
        TCP_SKB_CB(skb)->seq == tp->rcv_nxt &&
        !after(TCP_SKB_CB(skb)->ack_seq, tp->snd_nxt)) {

        int tcp_header_len = tp->tcp_header_len;

        /* Timestamp header prediction: tcp_header_len
         * is automatically equal to th->doff*4 due to pred_flags
         * match.
         */

        /* Check timestamp */
        if (tcp_header_len == sizeof(struct tcphdr) + TCPOLEN_TSTAMP_ALIGNED) {
            /* No timestamp? goto slow path! */
            if (!tcp_parse_aligned_timestamp(tp, th))
                goto slow_path;

            /* If PAWS failed, check it more carefully in slow path */
            if ((s32)(tp->rx_opt.rcv_tsval - tp->rx_opt.ts_recent) < 0)
                goto slow_path;

            /* DO NOT update ts_recent here, if checksum fails
             * and timestamp was corrupted part, it will result
             * in a hung connection since we will drop all
             * future packets due to the PAWS test.
             */
        }

        /* no data */
        if (len <= tcp_header_len) {
            /* Bulk data transfer: sender */
            if (len == tcp_header_len) {
                /* Predicted packet is in window by definition.
                 * seq == rcv_nxt and rcv_wup <= rcv_nxt.
                 * Hence, check seq<=rcv_wup reduces to:
                 */
                /*
                    the received tcp header got timestamp, if this tcp is valid,
                    save this timestamp
                */
                if (tcp_header_len ==
                    (sizeof(struct tcphdr) + TCPOLEN_TSTAMP_ALIGNED) &&
                    tp->rcv_nxt == tp->rcv_wup)
                    tcp_store_ts_recent(tp);

                /* We know that such packets are checksummed
                 * on entry.
                 ack this tcp request, free memory of this socket buffer
                 */
                tcp_ack(sk, skb, 0);
                __kfree_skb(skb);

                /* if there is data to send, andn check buffer space */
                tcp_data_snd_check(sk);
                return;
            }
            else { /* Header too small, invalid packet */
                TCP_INC_STATS(sock_net(sk), TCP_MIB_INERRS);
                goto discard;
            }
        }
        /* got data */
        else {
            int eaten = 0;
            bool fragstolen = false;

			// if tp's packet is valid (as judged by the below conditioins) 
			// copy skb to user space's socket
            if (tp->ucopy.task == current &&
                // sequence numbers of read and next received should be the same 
                tp->copied_seq == tp->rcv_nxt &&
                // copied data length so far
                len - tcp_header_len <= tp->ucopy.len &&
                /* user is now using this sock */
                sock_owned_by_user(sk)) {

                __set_current_state(TASK_RUNNING);

                if (!tcp_copy_to_iovec(sk, skb, tcp_header_len)) {
                    /* Predicted packet is in window by definition.
                     * seq == rcv_nxt and rcv_wup <= rcv_nxt.
                     * Hence, check seq<=rcv_wup reduces to:
                     */
                    /* update timestamp */
                    if (tcp_header_len ==
                        (sizeof(struct tcphdr) +
                         TCPOLEN_TSTAMP_ALIGNED) &&
                        tp->rcv_nxt == tp->rcv_wup)
                        tcp_store_ts_recent(tp);

                    /* estimate rtt */
                    tcp_rcv_rtt_measure_ts(sk, skb);

                    __skb_pull(skb, tcp_header_len);

                    /* update next received packet sequence number */
                    tcp_rcv_nxt_update(tp, TCP_SKB_CB(skb)->end_seq);
                    NET_INC_STATS(sock_net(sk),
                            LINUX_MIB_TCPHPHITSTOUSER);
                    eaten = 1;
                }
            }

            /* failed to copy to user space */
            if (!eaten) {
                if (tcp_checksum_complete(skb))
                    goto csum_error;

                if ((int)skb->truesize > sk->sk_forward_alloc)
                    goto step5;

                /* Predicted packet is in window by definition.
                 * seq == rcv_nxt and rcv_wup <= rcv_nxt.
                 * Hence, check seq<=rcv_wup reduces to:
                 */
                /* update timestamp */
                if (tcp_header_len ==
                    (sizeof(struct tcphdr) + TCPOLEN_TSTAMP_ALIGNED) &&
                    tp->rcv_nxt == tp->rcv_wup)
                    tcp_store_ts_recent(tp);

                /* compute rtt */
                tcp_rcv_rtt_measure_ts(sk, skb);

                NET_INC_STATS(sock_net(sk), LINUX_MIB_TCPHPHITS);

                /* Bulk data transfer: receiver */
                // add data to receive queue
                eaten = tcp_queue_rcv(sk, skb, tcp_header_len,
                              &fragstolen);
            }

            tcp_event_data_recv(sk, skb);

            /* verify data by seuqence number */
            if (TCP_SKB_CB(skb)->ack_seq != tp->snd_una) {
                /* Well, only one small jumplet in fast path... */
                // ack skb
                tcp_ack(sk, skb, FLAG_DATA);
                // if got data to send
                tcp_data_snd_check(sk);
                // no ack to send
                if (!inet_csk_ack_scheduled(sk))
                    goto no_ack;
            }

            // check if ack needs to send, and send it if required 
            __tcp_ack_snd_check(sk, 0);
no_ack:
            //skb already copied to user space, can be freed
            if (eaten)
                kfree_skb_partial(skb, fragstolen);

            /* wakae up user process to read data */
            sk->sk_data_ready(sk);
            return;
        }
    }

slow_path:
    /* length err|| checksum validation err */
    if (len < (th->doff << 2) || tcp_checksum_complete(skb))
        goto csum_error;

    /* no ack, no rst, no syn; discard the tcp packet */
    if (!th->ack && !th->rst && !th->syn)
        goto discard;

    /*
     *    Standard slow path.
     */
    /* many validations */
    if (!tcp_validate_incoming(sk, skb, th, 1))
        return;

step5:
    if (tcp_ack(sk, skb, FLAG_SLOWPATH | FLAG_UPDATE_TS_RECENT) < 0)
        goto discard;

    /* compute rtt */
    tcp_rcv_rtt_measure_ts(sk, skb);

    /* Process urgent data. */
    tcp_urg(sk, skb, th);

    /* step 7: process the segment text */
    /* handle data */
    tcp_data_queue(sk, skb);

    tcp_data_snd_check(sk);

    tcp_ack_snd_check(sk);
    return;

csum_error:
    TCP_INC_STATS(sock_net(sk), TCP_MIB_CSUMERRORS);
    TCP_INC_STATS(sock_net(sk), TCP_MIB_INERRS);

discard:
    tcp_drop(sk, skb);
}
```

```cpp
/* This routine deals with incoming acks, but not outgoing ones. */
static int tcp_ack(struct sock *sk, const struct sk_buff *skb, int flag)
{
	struct inet_connection_sock *icsk = inet_csk(sk);
	struct tcp_sock *tp = tcp_sk(sk);
	struct tcp_sacktag_state sack_state;
	u32 prior_snd_una = tp->snd_una;
	u32 ack_seq = TCP_SKB_CB(skb)->seq;
	u32 ack = TCP_SKB_CB(skb)->ack_seq;
	bool is_dupack = false;
	u32 prior_fackets;
	int prior_packets = tp->packets_out;
	const int prior_unsacked = tp->packets_out - tp->sacked_out;
	int acked = 0; /* Number of packets newly acked */

	sack_state.first_sackt.v64 = 0;

	/* We very likely will need to access write queue head. */
	prefetchw(sk->sk_write_queue.next);

	/* If the ack is older than previous acks
	 * then we can probably ignore it.
	 */
	if (before(ack, prior_snd_una)) {
		/* RFC 5961 5.2 [Blind Data Injection Attack].[Mitigation] */
		if (before(ack, prior_snd_una - tp->max_window)) {
			tcp_send_challenge_ack(sk, skb);
			return -1;
		}
		goto old_ack;
	}

	/* If the ack includes data we haven't sent yet, discard
	 * this segment (RFC793 Section 3.9).
	 */
	if (after(ack, tp->snd_nxt))
		goto invalid_ack;

	if (icsk->icsk_pending == ICSK_TIME_EARLY_RETRANS ||
	    icsk->icsk_pending == ICSK_TIME_LOSS_PROBE)
		tcp_rearm_rto(sk);

	if (after(ack, prior_snd_una)) {
		flag |= FLAG_SND_UNA_ADVANCED;
		icsk->icsk_retransmits = 0;
	}

	prior_fackets = tp->fackets_out;

	/* ts_recent update must be made after we are sure that the packet
	 * is in window.
	 */
	if (flag & FLAG_UPDATE_TS_RECENT)
		tcp_replace_ts_recent(tp, TCP_SKB_CB(skb)->seq);

	if (!(flag & FLAG_SLOWPATH) && after(ack, prior_snd_una)) {
		/* Window is constant, pure forward advance.
		 * No more checks are required.
		 * Note, we use the fact that SND.UNA>=SND.WL2.
		 */
		tcp_update_wl(tp, ack_seq);
		tcp_snd_una_update(tp, ack);
		flag |= FLAG_WIN_UPDATE;

		tcp_in_ack_event(sk, CA_ACK_WIN_UPDATE);

		NET_INC_STATS_BH(sock_net(sk), LINUX_MIB_TCPHPACKS);
	} else {
		u32 ack_ev_flags = CA_ACK_SLOWPATH;

		if (ack_seq != TCP_SKB_CB(skb)->end_seq)
			flag |= FLAG_DATA;
		else
			NET_INC_STATS_BH(sock_net(sk), LINUX_MIB_TCPPUREACKS);

		flag |= tcp_ack_update_window(sk, skb, ack, ack_seq);

		if (TCP_SKB_CB(skb)->sacked)
			flag |= tcp_sacktag_write_queue(sk, skb, prior_snd_una,
							&sack_state);

		if (tcp_ecn_rcv_ecn_echo(tp, tcp_hdr(skb))) {
			flag |= FLAG_ECE;
			ack_ev_flags |= CA_ACK_ECE;
		}

		if (flag & FLAG_WIN_UPDATE)
			ack_ev_flags |= CA_ACK_WIN_UPDATE;

		tcp_in_ack_event(sk, ack_ev_flags);
	}

	/* We passed data and got it acked, remove any soft error
	 * log. Something worked...
	 */
	sk->sk_err_soft = 0;
	icsk->icsk_probes_out = 0;
	tp->rcv_tstamp = tcp_time_stamp;
	if (!prior_packets)
		goto no_queue;

	/* See if we can take anything off of the retransmit queue. */
	acked = tp->packets_out;
	flag |= tcp_clean_rtx_queue(sk, prior_fackets, prior_snd_una,
				    &sack_state);
	acked -= tp->packets_out;

	if (tcp_ack_is_dubious(sk, flag)) {
		is_dupack = !(flag & (FLAG_SND_UNA_ADVANCED | FLAG_NOT_DUP));
		tcp_fastretrans_alert(sk, acked, prior_unsacked,
				      is_dupack, flag);
	}
	if (tp->tlp_high_seq)
		tcp_process_tlp_ack(sk, ack, flag);

	/* Advance cwnd if state allows */
	if (tcp_may_raise_cwnd(sk, flag))
		tcp_cong_avoid(sk, ack, acked);

	if ((flag & FLAG_FORWARD_PROGRESS) || !(flag & FLAG_NOT_DUP)) {
		struct dst_entry *dst = __sk_dst_get(sk);
		if (dst)
			dst_confirm(dst);
	}

	if (icsk->icsk_pending == ICSK_TIME_RETRANS)
		tcp_schedule_loss_probe(sk);
	tcp_update_pacing_rate(sk);
	return 1;

no_queue:
	/* If data was DSACKed, see if we can undo a cwnd reduction. */
	if (flag & FLAG_DSACKING_ACK)
		tcp_fastretrans_alert(sk, acked, prior_unsacked,
				      is_dupack, flag);
	/* If this ack opens up a zero window, clear backoff.  It was
	 * being used to time the probes, and is probably far higher than
	 * it needs to be for normal retransmission.
	 */
	if (tcp_send_head(sk))
		tcp_ack_probe(sk);

	if (tp->tlp_high_seq)
		tcp_process_tlp_ack(sk, ack, flag);
	return 1;

invalid_ack:
	SOCK_DEBUG(sk, "Ack %u after %u:%u\n", ack, tp->snd_una, tp->snd_nxt);
	return -1;

old_ack:
	/* If data was SACKed, tag it and see if we should send more data.
	 * If data was DSACKed, see if we can undo a cwnd reduction.
	 */
	if (TCP_SKB_CB(skb)->sacked) {
		flag |= tcp_sacktag_write_queue(sk, skb, prior_snd_una,
						&sack_state);
		tcp_fastretrans_alert(sk, acked, prior_unsacked,
				      is_dupack, flag);
	}

	SOCK_DEBUG(sk, "Ack %u before %u:%u\n", ack, tp->snd_una, tp->snd_nxt);
	return 0;
}
```

`tcp_prequeue` is enabled when `sysctl_tcp_low_latency || !tp->ucopy.task` is true.
```cpp
/* Packet is added to VJ-style prequeue for processing in process
 * context, if a reader task is waiting. Apparently, this exciting
 * idea (VJ's mail "Re: query about TCP header on tcp-ip" of 07 Sep 93)
 * failed somewhere. Latency? Burstiness? Well, at least now we will
 * see, why it failed. 8)8)				  --ANK
 *
 */
bool tcp_prequeue(struct sock *sk, struct sk_buff *skb)
{
	struct tcp_sock *tp = tcp_sk(sk);

	if (sysctl_tcp_low_latency || !tp->ucopy.task)
		return false;

	if (skb->len <= tcp_hdrlen(skb) &&
	    skb_queue_len(&tp->ucopy.prequeue) == 0)
		return false;

	/* Before escaping RCU protected region, we need to take care of skb
	 * dst. Prequeue is only enabled for established sockets.
	 * For such sockets, we might need the skb dst only to set sk->sk_rx_dst
	 * Instead of doing full sk_rx_dst validity here, let's perform
	 * an optimistic check.
	 */
	if (likely(sk->sk_rx_dst))
		// Drops dst reference count if a reference was taken.
		skb_dst_drop(skb);
	else
		skb_dst_force(skb);

	__skb_queue_tail(&tp->ucopy.prequeue, skb);
	tp->ucopy.memory += skb->truesize;
	if (tp->ucopy.memory > sk->sk_rcvbuf) {
		struct sk_buff *skb1;

		BUG_ON(sock_owned_by_user(sk));

		while ((skb1 = __skb_dequeue(&tp->ucopy.prequeue)) != NULL) {
			sk_backlog_rcv(sk, skb1);
			NET_INC_STATS_BH(sock_net(sk),
					 LINUX_MIB_TCPPREQUEUEDROPPED);
		}

		tp->ucopy.memory = 0;
	} else if (skb_queue_len(&tp->ucopy.prequeue) == 1) {
		wake_up_interruptible_sync_poll(sk_sleep(sk),
					   POLLIN | POLLRDNORM | POLLRDBAND);
		if (!inet_csk_ack_scheduled(sk))
			inet_csk_reset_xmit_timer(sk, ICSK_TIME_DACK,
						  (3 * tcp_rto_min(sk)) / 4,
						  TCP_RTO_MAX);
	}
	return true;
}
```

Copy happens recursively calling `skb_copy_datagram_iter` on nodes of `sk_buff *skb`, taking the data from `skb` to `iovec` (remember, `iovec` describes a user space cache area).
```cpp
static int tcp_copy_to_iovec(struct sock *sk, struct sk_buff *skb, int hlen)
{
	struct tcp_sock *tp = tcp_sk(sk);
	int chunk = skb->len - hlen;
	int err;

	if (skb_csum_unnecessary(skb))
		err = skb_copy_datagram_msg(skb, hlen, tp->ucopy.msg, chunk);
	else
		err = skb_copy_and_csum_datagram_msg(skb, hlen, tp->ucopy.msg);

	if (!err) {
		tp->ucopy.len -= chunk;
		tp->copied_seq += chunk;
		tcp_rcv_space_adjust(sk);
	}

	return err;
}

static inline int skb_copy_datagram_msg(const struct sk_buff *from, int offset,
					struct msghdr *msg, int size)
{
	return skb_copy_datagram_iter(from, offset, &msg->msg_iter, size);
}

/**
 *	skb_copy_datagram_iter - Copy a datagram to an iovec iterator.
 *	@skb: buffer to copy
 *	@offset: offset in the buffer to start copying from
 *	@to: iovec iterator to copy to
 *	@len: amount of data to copy from buffer to iovec
 */
int skb_copy_datagram_iter(const struct sk_buff *skb, int offset,
			   struct iov_iter *to, int len)
{
	int start = skb_headlen(skb);
	int i, copy = start - offset;
	struct sk_buff *frag_iter;

	trace_skb_copy_datagram_iovec(skb, len);

	/* Copy header. */
	if (copy > 0) {
		if (copy > len)
			copy = len;
		if (copy_to_iter(skb->data + offset, copy, to) != copy)
			goto short_copy;
		if ((len -= copy) == 0)
			return 0;
		offset += copy;
	}

	for (i = 0; i < skb_shinfo(skb)->nr_frags; i++) {
		int end;
		const skb_frag_t *frag = &skb_shinfo(skb)->frags[i];

		WARN_ON(start > offset + len);

		end = start + skb_frag_size(frag);
		if ((copy = end - offset) > 0) {
			if (copy > len)
				copy = len;
			if (copy_page_to_iter(skb_frag_page(frag),
					      frag->page_offset + offset -
					      start, copy, to) != copy)
				goto short_copy;
			if (!(len -= copy))
				return 0;
			offset += copy;
		}
		start = end;
	}

	skb_walk_frags(skb, frag_iter) {
		int end;

		WARN_ON(start > offset + len);

		end = start + frag_iter->len;
		if ((copy = end - offset) > 0) {
			if (copy > len)
				copy = len;
			if (skb_copy_datagram_iter(frag_iter, offset - start,
						   to, copy))
				goto fault;
			if ((len -= copy) == 0)
				return 0;
			offset += copy;
		}
		start = end;
	}
	if (!len)
		return 0;

	/* This is not really a user copy fault, but rather someone
	 * gave us a bogus length on the skb.  We should probably
	 * print a warning here as it may indicate a kernel bug.
	 */

fault:
	return -EFAULT;

short_copy:
	if (iov_iter_count(to))
		goto fault;

	return 0;
}
```

`skb_dequeue` removes an `skb` node from a list by operating on pointers. 
```cpp
/**
 *	__skb_dequeue - remove from the head of the queue
 *	@list: list to dequeue from
 *
 *	Remove the head of the list. This function does not take any locks
 *	so must be used with appropriate locks held only. The head item is
 *	returned or %NULL if the list is empty.
 */
struct sk_buff *skb_dequeue(struct sk_buff_head *list);
static inline struct sk_buff *__skb_dequeue(struct sk_buff_head *list)
{
	struct sk_buff *skb = skb_peek(list);
	if (skb)
		__skb_unlink(skb, list);
	return skb;
}

/*
 * remove sk_buff from list. _Must_ be called atomically, and with
 * the list known..
 */
void skb_unlink(struct sk_buff *skb, struct sk_buff_head *list);
static inline void __skb_unlink(struct sk_buff *skb, struct sk_buff_head *list)
{
	struct sk_buff *next, *prev;

	list->qlen--;
	next	   = skb->next;
	prev	   = skb->prev;
	skb->next  = skb->prev = NULL;
	next->prev = prev;
	prev->next = next;
}
```