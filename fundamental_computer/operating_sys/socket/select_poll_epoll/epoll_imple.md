# Epoll Implementation Detail

The biggest performance bottleneck of kernel calling I/O is context switch that for non-epoll methods, such as a traditional `read` every time when I/O receives data, context switch costs about 2-3 microseconds.

