# Linux Performance DevOps

## General Performance Checking

### By `cat /proc/stat`

```bash
# cpu us ni sy id wa hi si st
cpu  530005 17184 149424 13286397 6269 0 4150 0 0 0
cpu0 43867 1464 12553 1107652 533 0 181 0 0 0
cpu1 44252 1501 12234 1106417 623 0 1323 0 0 0
cpu2 46022 1272 12565 1106049 543 0 140 0 0 0
cpu3 44501 1590 12100 1107422 627 0 89 0 0 0
```

* us, user    : time running un-niced user processes
* sy, system  : time running kernel processes
* ni, nice    : time running niced user processes
* id, idle    : time spent in the kernel idle handler
* wa, IO-wait : time waiting for I/O completion
* hi : time spent servicing hardware interrupts
* si : time spent servicing software interrupts
* st : time stolen from this vm by the hypervisor


Niceness is about process scheduling priority that,
if a process is very nice (high nice values), it will be "polite" and allow other processes to take precedence and use more CPU time (in other words, it will have a low priority). 
If it is not nice, it will try to get as much CPU time as possible for itself (so it will have a high priority).

### By `top`

For `top`, press `F` button on keyboard to go the config page; 
press `arrow` to move to different options;
press `SPACE` to select options;
press `q` to return then press `SHIFT+W` to save personalized config to `/root/.bashrc` (should use `sudo` if to save).

Some commonly monitored stats are shown as below
* PID     = Process Id             
* USER    = Effective User Name    
* PR      = Priority               
* NI      = Nice Value             
* VIRT    = Virtual Image (KiB)    
* RES     = Resident Size (KiB)    
* SHR     = Shared Memory (KiB)    
* S       = Process Status         
* %MEM    = Memory Usage (RES)   
* %CPU    = CPU Usage              
* TIME+   = CPU Time, hundredths   
* P       = Last Used Cpu (SMP)   
* COMMAND = Command Name/Line

```
    PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND                                                                                                       
   7255 yuqi      20   0   50.6g 384564 121472 S  17.3   2.4  14:55.05 code                                                                                                          
  78459 yuqi      20   0 2775780 297664 125188 S  13.3   1.8   1:00.33 Isolated Web Co                                                                                               
   1265 root      20   0   24.5g 154960  96788 S   6.0   1.0   9:37.98 Xorg                                                                                                          
   7213 yuqi      20   0   32.6g 145052  82716 S   5.6   0.9   9:12.61 code                                                                                                          
   7179 yuqi      20   0   36.9g 184996 126440 S   4.3   1.1   1:57.92 code                                                                                                          
   1615 yuqi      20   0 4821088 331400 114624 S   3.0   2.0   7:30.24 gnome-shell                                                                                                   
   7376 yuqi      20   0   36.5g 203224  71000 S   2.0   1.3   4:43.91 code                                                                                                          
   7377 yuqi      20   0   36.5g 217660  71008 S   2.0   1.3  11:57.26 code                                                                                                          
  74945 yuqi      20   0 4290536 419972 240356 S   1.7   2.6   3:46.02 firefox                                                                                                       
   2660 systemd+  20   0 2604092 121480  59904 S   1.3   0.8   1:46.05 mongod                   
```

### By `vmstat`

`vmstat [options] [delay [count]]` shows

For example, `vmstat 2 6` retrieves stats every 2 secs, running/printing for 6 times.
```
procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----
 r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st
 1  0   6912 1842144 420012 7465144    0    0    30    40  212  211  4  1 95  0  0
 0  0   6912 1841672 420012 7466488    0    0     0     0 1209 2428  1  1 98  0  0
 0  0   6912 1837208 420020 7469852    0    0     0   170 1068 2158  1  1 99  0  0
 0  0   6912 1845084 420020 7463064    0    0     0     0 1128 2167  1  0 99  0  0
 0  0   6912 1846380 420020 7460912    0    0     0    14 1213 2233  1  1 99  0  0
 0  0   6912 1843548 420028 7463092    0    0     0   126 1071 2203  1  0 99  0  0
```

### By `pidstat`

`pidstat [options] [monitor_period [output_count]]` can monitor individual tasks managed by Linux kernel.

For example, `pidstat -w 10 1` monitors processes' switching activities
```
09:58:20 PM   UID       PID   cswch/s nvcswch/s  Command
09:58:30 PM     0         1      1.80      0.10  systemd
09:58:30 PM     0         2      0.40      0.00  kthreadd
09:58:30 PM     0        13      0.90      0.00  ksoftirqd/0
09:58:30 PM     0        14     84.73      0.00  rcu_sched
09:58:30 PM     0        15      0.20      0.00  migration/0
09:58:30 PM     0        21      0.20      0.00  migration/1
09:58:30 PM     0        22      0.70      0.00  ksoftirqd/1
09:58:30 PM     0        27      0.20      0.00  migration/2
09:58:30 PM     0        28      1.00      0.00  ksoftirqd/2
...
```
where `cswch/s` and `nvcswch/s` refer to voluntary/non-voluntary context switch.

A voluntarily switch is about a process requiring resources currently not available, such as I/O operations; a non-voluntary switch happens when a process runs out of its allocated time slice.

Inside the context switch records, `rcu_sched` consumes most of the resources.
Read-copy-update (RCU) is a synchronization mechanism that avoids the use of lock primitives while multiple threads concurrently read and update elements that are linked through pointers and that belong to shared data structures.


### Interrupt Checking by `/proc/interrupts`

This command `grep "nvme" /proc/interrupts |sed 's/     / /g'` shows all CPUs' I/O interrupts (`nvme` is an SSD disk interface protocol).

Here shows 12 CPUs handling nvme interrupts, indicating that I/O interrupts are evenly distributed to be handled by all the 12 CPUs.
Sometimes if wrong, one CPU might accumulate many interrupts awaiting being handled, resulted in one CPU having high usage while others are idle.
```bash
 133:  0  0  0  0  0  1  0  0     36  0  0  0  IR-PCI-MSI 31457280-edge  nvme0q0
 134:  26829  0  0  0  0  0  0  0  0  0  0  0  IR-PCI-MSI 31457281-edge  nvme0q1
 135:  0  33917  0  0  0  0  0  0  0  0  0  0  IR-PCI-MSI 31457282-edge  nvme0q2
 136:  0  0  27975  0  0  0  0  0  0  0  0  0  IR-PCI-MSI 31457283-edge  nvme0q3
 137:  0  0  0  30392  0  0  0  0  0  0  0  0  IR-PCI-MSI 31457284-edge  nvme0q4
 138:  0  0  0  0  28798  0  0  0  0  0  0  0  IR-PCI-MSI 31457285-edge  nvme0q5
 139:  0  0  0  0  0  22153  0  0  0  0  0  0  IR-PCI-MSI 31457286-edge  nvme0q6
 140:  0  0  0  0  0  0  21230  0  0  0  0  0  IR-PCI-MSI 31457287-edge  nvme0q7
 141:  0  0  0  0  0  0  0  27289  0  0  0  0  IR-PCI-MSI 31457288-edge  nvme0q8
 142:  0  0  0  0  0  0  0  0  32428  0  0  0  IR-PCI-MSI 31457289-edge  nvme0q9
 143:  0  0  0  0  0  0  0  0  0  24772  0  0  IR-PCI-MSI 31457290-edge  nvme0q10
 144:  0  0  0  0  0  0  0  0  0  0  21992  0  IR-PCI-MSI 31457291-edge  nvme0q11
 145:  0  0  0  0  0  0  0  0  0  0  0  27022  IR-PCI-MSI 31457292-edge  nvme0q12
```