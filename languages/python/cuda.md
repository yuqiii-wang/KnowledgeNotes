# Cuda with python

Cuda kernel is defined with annotation `@cuda.jit`

Grid and block (similar to c++ extension `<<<grid, block>>>`) are defined in `[grid, block]`

```py
from numba import cuda
import numpy as np
import math
from time import time

@cuda.jit
def gpu_add(a, b, result, n):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n :
        result[idx] = a[idx] + b[idx]

def main():
    n = 20000000
    x = np.arange(n).astype(np.int32)
    y = 2 * x

    # host to device copy
    x_device = cuda.to_device(x)
    y_device = cuda.to_device(y)
    # device mem allocation for result data storage
    gpu_result = cuda.device_array(n)
    cpu_result = np.empty(n)

    threads_per_block = 1024
    blocks_per_grid = math.ceil(n / threads_per_block)
    start = time()
    gpu_add[blocks_per_grid, threads_per_block](x_device, y_device, gpu_result, n)
    cuda.synchronize()
    print("gpu vector add time " + str(time() - start))
    start = time()
    cpu_result = np.add(x, y)
    print("cpu vector add time " + str(time() - start))

    if (np.array_equal(cpu_result, gpu_result.copy_to_host())):
        print("result correct!")

if __name__ == "__main__":
    main()
```


## Debug

* `Failed to dlopen libcuda.so`

`libcuda.so.1` is a symlink to a file that is specific to the version of your NVIDIA drivers. It may be pointing to the wrong version or it may not exist.

* Could not open cuda

Use `navidia-smi` to check if Python is running on GPU. Nvidia GPU manager often automatically kills gpu-python process for inactivity.

A common solution is to reboot your computer.