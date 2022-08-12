# CUDA practices 

## Use pinned memory

Use `cudaMallocHost` to make data's memory persistent on host device, rather than `malloc` or `new` operation. For memory discharge, use `cudaFreeHost`, saved the time to copy from pageable host memory to page-locked host memory.

```cp
int* h_dataA;
cudaMallocHost(&h_dataA, N * sizeof(int));

cudaFreeHost(h_dataA);
```

## Max of kernel input memory and threads

Given a kernel function:
```cpp
dim3 grid(gridDimX, gridDimY, gridDimZ);
dim3 block(blockDimX, blockDimY, blockDimZ);
MatAdd<<<grid, block>>>(input);
```

There are two hardware conditions to be satisfied

* The total memory size for kernel process should be less than GPU's memory

`gridDimX` $\times$ `gridDimY` $\times$ `gridDimZ` 
$\times$ `blockDimX` $\times$ `blockDimY` $\times$ `blockDimZ` $\le$ `GPU-Mem`

* Each block should have the number of threads less than the number of GPU cores

`blockDimX` $\times$ `blockDimY` $\times$ `blockDimZ` $\le$ `GPU-Core-Number`

## Thread Safty

Similar to a thread for CPU, write to the same addr by multiple threads is forbidden (resulting in undefined behavior).

For read operation, the access data should be consistent throughout the whole kernel function execution by multiplee threads.

For example, below code results in undefined behavior, that there are too many threads at the same time accessing the same addr `C[0]`, and `C[0]`'s value is undetermined. 
```cpp
__global__ void setData(int* C)
{
    C[0] = 111;
}

int main()
{
    //...
    dim3 grid( 1<<4, 1<<4, 1<<4 );
    dim3 block( 1<<4, 1<<4, 1<<4 );
    setData<<<grid, block>>>(C);
    //...
}
```