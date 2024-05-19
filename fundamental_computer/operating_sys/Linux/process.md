# Process

## Children process creation

Linux manages processes with assigned priority and locks to shared memory access.

System calls are

* `fork()` creates a new process by duplicating the calling process. On success, the PID of the child process is returned in the parent, and 0 is returned in the child.

* `exec()` family of functions replaces the current process image with a new process image. It loads the program into the current process space and runs it from the entry point, such as `exec("ls")` runs `ls` from the current process.

* `clone()` gives a new process or a new thread depending on passed arguments to determined various shared memory regions. For example, `CLONE_FS` dictates shared file system; `CLONE_SIGHAND` dictates shared signal handlers. If with no argument flags, it is same as `fork()`.

## Process attributes

* PID - Process ID

Process ID is a unique identifier of a process.

* PPID - Parent Process ID

The parent process ID of a process is the process ID of its creator, for the lifetime of the creator. After the creator's lifetime has ended, the parent process ID is the process ID of an implementation-defined system process.

* SID - Session ID

A collection of process groups established for job control purposes. Each process group is a member of a session.

* PGID - Process Group ID

A collection of processes that permits the signaling of related processes.

* EUID - Effective User ID

An attribute of a process that is used in determining various permissions, including file access permissions; see also User ID.

## Process Communications

### Pipe

The oldest UNIX IPC communication tool working from parent sending data to child process, such as `cat <file> | grep <word>`, where the `|` is a pipe communication.

```cpp
#include <unistd.h>
 
int pipe(int pipefd[2]);
```

### FIFO

Work on file I/O operation reading/writing by First-In-First-Out (FIFO) order.

```cpp
#include <sys/types.h>
#include <sys/stat.h>
 
int mkfifo(const char *filename, mode_t mode);
```

For example, below code writes and reads the `"file"` in a FIFO order.
```cpp
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
 
int main()
{
    int fd;
    int pid;
    int sum = 3;
    char r_buf[128] = {0};

    if (((mkfifo("file", 0600)) == -1) && (errno != EEXIST)) {	// create FIFO
            perror("mkfifo fail");
            exit(-1);
    }

    if ((pid = fork()) == -1) {	// launch a process
            perror("fork fail");
            exit(-1);
    } else if (pid > 0) {	// parent process
            fd = open("file", O_WRONLY);	// open FIFO and write data
            while(sum)	
            {
                    write(fd, "message form fifo", strlen("message from fifo"));
                    sleep(1); // write data every second
                    sum--;
            }
            close(fd);	// close FIFO
    } else if (pid == 0) {	// child process
            fd = open("file", O_RDONLY);	// open FIFO, and read data from the file
            while (sum)		//
            {
                    read(fd, r_buf, 128);
                    puts(r_buf);
                    sleep(1); // read every second
                    sum--;
            }
            close(fd);	// close FIFO
    }

    return 0;
}
```

### Message Queue

Data packets are sent as an individual message, where packets' format and size are defined before transmission.

```cpp
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>
 
// open/create an MQ
int msgget(key_t key, int msgflg);

// send data
int msgsnd(int msqid, const void *msgp, size_t msgsz, int flag);

// recv data
ssize_t msgrcv(int msqid, void *msgp, size_t msgsz, long msgtyp, int flag);
```

### Shared Memory

The fastest way to share data between processes.

There is data racing issue, usually used semaphore to sync data read/writing.

```cpp
#include <sys/ipc.h>
#include <sys/shm.h>

// get/create a block of shared mem
int shmget(key_t key, size_t size, int flag);

// associate the shared mem to the current proc
void *shmat(int shm_id, const void *addr, int flag);
```

* Linux implementation details:

Every process has its own virtual memory space. 

Creating shared memory (by `shmget`) allocates a chunk of memory that does not belong to any particular process.

A process has to request access to it – that's the role of `shmat`. 
By doing that, kernel maps the shared memory into process' virtual memory space. 

### Signal

Typically served as a soft interrupt. For example, `CTRL+C` is a `kill` signal to terminate a running program.

`kill -l` shows the Linux signals.
```
kill -l

 1) SIGHUP	    2) SIGINT	     3) SIGQUIT	     4) SIGILL	     5) SIGTRAP
 2) SIGABRT	    7) SIGBUS	     8) SIGFPE	     9) SIGKILL	    10) SIGUSR1
1)  SIGSEGV	    12) SIGUSR2	    13) SIGPIPE	    14) SIGALRM	    15) SIGTERM
2)  SIGSTKFLT	17) SIGCHLD	    18) SIGCONT	    19) SIGSTOP	    20) SIGTSTP
3)  SIGTTIN	    22) SIGTTOU	    23) SIGURG	    24) SIGXCPU	    25) SIGXFSZ
4)  SIGVTALRM	27) SIGPROF	    28) SIGWINCH	29) SIGIO	    30) SIGPWR
5)  SIGSYS	    34) SIGRTMIN	35) SIGRTMIN+1	36) SIGRTMIN+2	37) SIGRTMIN+3
6)  SIGRTMIN+4	39) SIGRTMIN+5	40) SIGRTMIN+6	41) SIGRTMIN+7	42) SIGRTMIN+8
7)  SIGRTMIN+9	44) SIGRTMIN+10	45) SIGRTMIN+11	46) SIGRTMIN+12	47) SIGRTMIN+13
8)  SIGRTMIN+14	49) SIGRTMIN+15	50) SIGRTMAX-14	51) SIGRTMAX-13	52) SIGRTMAX-12
9)  SIGRTMAX-11	54) SIGRTMAX-10	55) SIGRTMAX-9	56) SIGRTMAX-8	57) SIGRTMAX-7
10) SIGRTMAX-6	59) SIGRTMAX-5	60) SIGRTMAX-4	61) SIGRTMAX-3	62) SIGRTMAX-2
11) SIGRTMAX-1	64) SIGRTMAX
```

In practice, can associate signal with a handler/callback function.
```cpp
#include <signal.h>
 
typedef void (*sighandler_t)(int); // interrupt handler
sighandler_t signal(int signum, sighandler_t handler);
```

### Semaphore

Semaphore is used to sync processes' operation (similar to mutex in threads) when there are multiple processes accessing the same resource.

```cpp
int semop(int semid, struct sembuf semoparray[], size_t numops);
/*
semid:semaphore id  
numops:number of semaphores
*/
struct sembuf 
{
    short sem_num; 
    short sem_op;  
    short sem_flg; // IPC_NOWAIT, SEM_UNDO
};
```

### Socket

* `int fd = socket(AF_INET, SOCK_STREAM, 0);`

* `int ret = bind(fd, (struct sockaddr*)&servaddr, sizeof(servaddr));`

* `ret = listen(fd, 200);`

* `int cfd = accept(fd, (struct sockaddr*)&cliaddr, &caddr_len);`

* `int ret = connect(fd, (struct sockaddr*)&servaddr, sizeof(servaddr));`