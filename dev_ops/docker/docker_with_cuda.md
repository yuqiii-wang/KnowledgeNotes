# Docker With CUDA

Illustrated in the below figure, that a host machine MUST install CUDA driver.
Check the driver availability by `nvidia-smi`.

<div style="display: flex; justify-content: center;">
      <img src="imgs/cuda_nvidia_gpu_arch.png" width="50%" height="50%" alt="cuda_nvidia_gpu_arch" />
</div>
</br>

CUDA toolkit container https://github.com/NVIDIA/nvidia-container-toolkit is not a necessity to install in a docker to run CUDA.
However, advanced features such as `--gpus all` flag needs CUDA toolkit container.