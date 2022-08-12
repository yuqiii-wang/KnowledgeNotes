#include <stdio.h>
#include <algorithm>
#include <iostream>

__global__ void kernelSum(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

int main(void)
{
    int N = 1 << 20;
    float *x, *y, *d_x, *d_y;
    x = (float *)malloc(N * sizeof(float));
    y = (float *)malloc(N * sizeof(float));
    std::fill_n(x, N, 1.0);
    std::fill_n(y, N, 2.0);

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Perform kernelSum on 1M elements
    kernelSum<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);

    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N>>16; i++)
    {
        if (y[i] != 0)
        {
            if (i % 8 == 0)
            {
                std::cout << std::endl;
            }
            std::cout << "y[" << i << "]:" << y[i] << " ";
        }
    }
    std::cout << std::endl;

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
}