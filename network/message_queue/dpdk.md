# Data Plane Development Kit (DPDK)

The Data Plane Development Kit (DPDK) is an open source software project managed by the Linux Foundation (initially launched in 2010 by Intel). 
It is basically an enhancement patch of many Linux modules for low latency packet processing, such as kernel bypass for fast network packet processing (supported User-Space I/O (UIO)), CPU core affinity, TSC for timing, etc.

DPDK is built on Environment Abstraction Layer (EAL) that takes into consideration hardware architectures such as x86, that are collectively abstracted as unified APIs for use.

## Quick Start

Install build tools 

* `sudo apt-get install ninja-build meson`
* `pip3 install pyelftools`/`sudo apt-get install -y python3-pyelftools python-pyelftools`

Download from `https://core.dpdk.org/download/`.
Then compile by

```bash
meson -Dexamples=all build
ninja -C build
```

Config to reserve large page memory (if `echo 64` fails for permission denied, just `sudo vim` to update the number to `64`)
```bash
mkdir -p /dev/hugepages
mountpoint -q /dev/hugepages || mount -t hugetlbfs nodev /dev/hugepages
echo 64 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
```

Testing

```bash
build/app/dpdk-testpmd -c7 --vdev=net_pcap0,iface=eth0 --vdev=net_pcap1,iface=eth1 --\
                  -i --nb-cores=2 --nb-ports=2 --total-num-mbufs=2048
```

build/app/dpdk-testpmd -c 3 -n 4 \
  --vdev='net_pcap0,rx_pcap=/tmp/tcp.pcap,tx_pcap=/tmp/tx0.pcap' \
  --vdev='net_pcap1,rx_pcap=/tmp/tcp.pcap,tx_pcap=/tmp/tx1.pcap' \
  -- \
  --port-topology=chained \
  --no-flush-rx -i --nb-ports=2

##