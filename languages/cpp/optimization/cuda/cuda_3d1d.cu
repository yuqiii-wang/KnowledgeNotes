#include <iostream>
#include <chrono>
#include <stdlib.h>
#include <unordered_map>
#include <thread>
#include <future>

#define NanoSec 1000000000.0
const int GridDim = 1 << 6;
const int BlockDim = 1 << 3;
const int ThreadNum = 32;

// Kernel definition
__global__ void MatAdd(int* A, int* B, int* C, int N)
{
    int gridDim = GridDim;
    int idx = 
        gridDim*gridDim*blockDim.z*blockDim.y*blockDim.x*blockIdx.z + 
        gridDim*blockDim.z*blockDim.y*blockDim.x*blockIdx.y + 
        blockDim.z*blockDim.y*blockDim.x*blockIdx.x +
        threadIdx.x*blockDim.z*blockDim.y +
        threadIdx.y*blockDim.z +
        threadIdx.z;
    C[idx] = A[idx] + B[idx];

}

int main()
{
    const int N = pow(GridDim, 3) * pow(BlockDim, 3) ;
    std::cout << "Number of elems: " << N << std::endl;

    int* h_dataA, * h_dataB, * h_dataC; 
    int * d_dataA, * d_dataB, * d_dataC; // host and device data ptr
    
    h_dataA = (int*)malloc(N * sizeof(int)); 
    h_dataB = (int*)malloc(N * sizeof(int)); 
    h_dataC = (int*)malloc(N * sizeof(int)); 
    std::fill_n(h_dataA, N, 10); // represents A
    std::fill_n(h_dataB, N, 5); // represents B

    auto timestamp_nano_start = std::chrono::system_clock::now().time_since_epoch().count();
    cudaMalloc(&d_dataA, N * sizeof(int));
    cudaMalloc(&d_dataB, N * sizeof(int));
    cudaMalloc(&d_dataC, N * sizeof(int));
    // first time cudaMemcpy invocation might be slow, next time shall be faster
    cudaMemcpy(d_dataA, h_dataA, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataB, h_dataB, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataC, h_dataC, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_dataC, 0, N * sizeof(int));
    auto timestamp_nano_end = std::chrono::system_clock::now().time_since_epoch().count();
    std::cout << "Cuda MemCpy to device elapsed time (second): " << (double)(timestamp_nano_end-timestamp_nano_start)/NanoSec << std::endl;

    // Kernel invocation
    dim3 grid( GridDim, GridDim, GridDim );
    // blockDim.x x blockDim.y x blockDim.z
    dim3 block( BlockDim, BlockDim, BlockDim );
    timestamp_nano_start = std::chrono::system_clock::now().time_since_epoch().count();
    for (int i = 0; i < 1000; i++)
        MatAdd<<<grid, block>>>(d_dataA,d_dataB, d_dataC, N);
    timestamp_nano_end = std::chrono::system_clock::now().time_since_epoch().count();
    std::cout << "Cuda MatAdd elapsed time (second): " << (double)(timestamp_nano_end-timestamp_nano_start)/NanoSec << std::endl;

    timestamp_nano_start = std::chrono::system_clock::now().time_since_epoch().count();
    // copy back from gpu to host
    cudaMemcpy(h_dataC, d_dataC, N * sizeof(int), cudaMemcpyDeviceToHost);
    timestamp_nano_end = std::chrono::system_clock::now().time_since_epoch().count();
    std::cout << "Cuda MemCpy to host elapsed time (second): " << (double)(timestamp_nano_end-timestamp_nano_start)/NanoSec << std::endl;


    // The below code snippet is irrelevant to cuda, just a summary of results
    std::unordered_map<int,int> resultCountDictSum;
    auto threadRun = [&](std::unordered_map<int,int>& resultCountDict, int threadId, int blockRange)
    {
        for (int i = blockRange*threadId; i < blockRange*(threadId+1); i++)
        {
            if (resultCountDict.find(h_dataC[i]) != resultCountDict.end())
            {
                resultCountDict[h_dataC[i]]++;
            }
            else
            {
                resultCountDict[h_dataC[i]] = 0;
            }
        }
    };
    std::thread* threadSummArr[ThreadNum];
    std::unordered_map<int,int> subResults[ThreadNum];
    for (int i = 0; i < ThreadNum; i++)
    {
        threadSummArr[i] = new std::thread(threadRun, std::ref(subResults[i]), i, N/ThreadNum);
    }
    for (int i = 0; i < ThreadNum; i++)
    {
        threadSummArr[i]->join();
    }

    
    for (int i = 0; i < ThreadNum; i++)
    {
        for (auto& eachResult : subResults[i])
        {
            if (subResults[i].find(eachResult.first) != subResults[i].end())
            {
                resultCountDictSum[eachResult.first] += eachResult.second;
            }
            else
            {
                resultCountDictSum[eachResult.first] = eachResult.second;
            }
        }
    }
    for (auto& eachResult : resultCountDictSum)
    {
        std::cout << "[" << eachResult.first << "]:" 
        << eachResult.second << " ";
    }
    std::cout << std::endl;
    
    cudaFree(d_dataA);
    cudaFree(d_dataB);
    cudaFree(d_dataC);
    free(h_dataA);
    free(h_dataB);
    free(h_dataC);

    return 0;
}