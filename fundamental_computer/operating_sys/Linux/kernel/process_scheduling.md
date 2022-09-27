# Process Scheduling

## Basic concepts

* Preemption/preemptive multitasking

Linux OS determines when a process is to cease running and a new process is to begin running.

* Cooperative multitasking

A process does not stop running until it voluntary decides to do so.

* Timeslice

The time a process runs before it is preempted is usually predetermined, and it is called the timeslice of the process.

The timeslice, in effect, gives each runnable process a slice of the processor’s time (such as 10 ms). 

Context switch is time-consuming and time allotments for processes should be carefully managed.

* Yielding

The act of a process voluntarily suspending itself is called yielding. 

* I/O-bound process scheduling policy

It is characterized as a process that spends much of its time submitting and waiting on I/O requests.

Should be a small runnable, such as a keyboard input.

* processor-bound process scheduling policy

Computation intensive and long execution duration.

* Process priority

Processes are ranked and high priority processes are executed first.

Processes with the same priority level are scheduled in a *round-robin* fashion.

* Completely Fair Scheduler (CFS)
 
CFS is the Linux process scheduler implementation.

## Scheduling Implementation

CFS tracks the consumed time of a process. When timeslice reaches zero, this process is preempted.

Inside `#include <linux/sched.h>`, defines `sched_entity`
```cpp
struct sched_entity {
    struct load_weight          load;
    struct rb_node              run_node;
    struct list_head            group_node;
    unsigned int                on_rq;
    u64                         exec_start;
    u64                         sum_exec_runtime;
    u64                         vruntime;
    u64                         prev_sum_exec_runtime;
    u64                         last_wakeup;
    u64                         avg_overlap;
    u64                         nr_migrations;
    u64                         start_runtime;
    u64                         avg_wakeup;
}
```

The `vruntime` variable stores the virtual runtime of a process, which is the actual runtime (the amount of time spent running) normalized (or weighted) by the number of runnable processes.

### Process Selection

CFS uses a red-black tree to manage the list of runnable processes and efficiently find the process with the smallest `vruntime`. The leftmost leave node is the task with the smallest `vruntime`.

### Scheduler recursive pick

CFS recursively invokes `pick_next_task(struct rq* rq)`, in which 

* `rq->nr_running == rq->cfs.nr_running` simply returns if the task request is already placed in CFS next running task
* `class = sched_class_highest;` and  `for ( ; ; ){...}` together go through next high priority task one by one

```cpp
static inline struct task_struct *
pick_next_task(struct rq *rq)
{
  const struct sched_class *class;
  struct task_struct *p;
  /*
  * Optimization: we know that if all tasks are in
  * the fair class we can call that function directly:
  */
  if (likely(rq->nr_running == rq->cfs.nr_running)) {
    p = fair_sched_class.pick_next_task(rq);
    if (likely(p))
      return p;
  }
  class = sched_class_highest;
  for ( ; ; ) {
    p = class->pick_next_task(rq);
    if (p)
      return p;

    /*
    * Will never be NULL as the idle class always
    * returns a non-NULL p:
    */
    class = class->next;
  }
}
```

## Preemption and Context Switch

Context switching, the switching from one runnable task to another, is handled by the `context_switch()` function defined in `kernel/sched.c` . It is called by `schedule()` when a new process has been selected to run. It does two basic jobs:
* Calls `switch_mm()` , which is declared in `<asm/mmu_context.h>` , to switch the virtual memory mapping from the previous process’s to that of the new process.
* Calls `switch_to()` , declared in `<asm/system.h>` , to switch the processor state from the previous process’s to the current’s.This involves saving and restoring stack information and the processor registers and any other architecture-specific state that must be managed and restored on a per-process basis.

## Real-time Scheduling 

Linux provides two real-time scheduling policies, `SCHED_FIFO` and `SCHED_RR` .The normal, not real-time scheduling policy is `SCHED_NORMAL`. Real-time policies are managed not by the Completely Fair Scheduler (CFS).

* `SCHED_FIFO`

First-in first-out; 

A runnable `SCHED_FIFO` task is always scheduled over any `SCHED_NORMAL` tasks by CFS.

Once become a runnable, it blocks until finished, or preempted by higher priority tasks. 

* `SCHED_RR`

Used a real-time, round-robin scheduling algo.