# CUDA practices 

## Install and Quick Start

### Prepares

`lspci | grep -i nvidia` to check hardware

`sudo apt-get install linux-headers-$(uname -r)` to install Linux tools

### Download

```bash
## To get cuda in apt list
distro=ubuntu2004
arch=x86_64
wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

## Actually install cuda, and reboot the system
sudo apt-get install cuda
sudo apt-get install nvidia-gds
sudo reboot
```

### Driver Install

Go to https://www.nvidia.com/download/index.aspx to find your cuda driver version
```bash
# for 1660 Ti
sudo apt-get install cuda-drivers-525
```

### `nvcc` Install

`nvcc` is cuda c++ compiler. 
Run `nvcc --version` to check if `nvcc` is installed,
if not, install it by
```bash
sudo apt install nvidia-cuda-toolkit
```

### CMake Use

In `CMakeLists.txt`, add `project(cuda_proj LANGUAGES CXX CUDA)` to enable cuda.

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

## Thread Safety

Similar to a thread for CPU, write to the same addr by multiple threads is forbidden (resulting in undefined behavior).

For read operation, the access data should be consistent throughout the whole kernel function execution by multiple threads.

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

## CUDA Builtin Functions

Cuda has builtin math functions that best utilizes BLAS.

## Nsight Compute Profiling

Nsight is the CUDA performance profiling tool.
It should be installed once user has finished CUDA full installation.

* Make sure rebooting compute once CUDA is finished first time installation
* Set Nsight access privilege (otherwise, use `sudo` to run it)

Simply, put `options nvidia "NVreg_RestrictProfilingToAdminUsers=0"` into `/etc/modprobe.d/nvidia-access.conf`.
```bash
echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' > /etc/modprobe.d/nvidia-access.conf
```

Nsight GUI should help generate some arguments (`--set full` should be set to catch all performance indicators, although it takes long time to run).
By `sudo` running the command below, should give a performance result file `/home/yuqi/Desktop/KnowledgeNotes/languages/cpp/optimization/cuda/examples/build/perf.ncu-rep`.
```bash
sudo /opt/nvidia/nsight-compute/2023.1.0/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /home/yuqi/Desktop/KnowledgeNotes/languages/cpp/optimization/cuda/examples/build/perf --force-overwrite --target-processes all --set full --call-stack --nvtx --apply-rules no /home/yuqi/Desktop/KnowledgeNotes/languages/cpp/optimization/cuda/examples/build/cuda_3d1d
```

<div style="display: flex; justify-content: center;">
      <img src="imgs/cuda_nsight_start.png" width="40%" height="40%" alt="cuda_nsight_start">
</div>
</br>
