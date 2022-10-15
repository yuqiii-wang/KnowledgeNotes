#include <iostream>
#include <sys/mman.h>
#include <stdlib.h>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#define MIN_MMAP_LENGTH 4096

int main(){

    int* mmapPtr = (int*)mmap(NULL, MIN_MMAP_LENGTH, 
                        PROT_WRITE|PROT_READ, 
                        MAP_PRIVATE|MAP_ANONYMOUS, 0, 0);

    int* mallocPtr = (int*)malloc(MIN_MMAP_LENGTH);

    int oneInt = 0;

    std::cout << "pid: " << getpid() 
                << std::hex
                << " mmapPtr: " << mmapPtr
                << " mallocPtr: " << mallocPtr
                << " &oneInt: " << &oneInt 
                << std::endl;

    sleep(100);
    munmap(mmapPtr, MIN_MMAP_LENGTH);
    free(mallocPtr);
    
    return 0;
}