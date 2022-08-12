#include <iostream>
#include <chrono>
#include <stdlib.h>
#include <unordered_map>
#include <thread>
#include <future>
#include "cublas_v2.h"

#define NanoSec 1000000000.0
const int GridDim = 1 << 6;
const int BlockDim = 1 << 3;
const int ThreadNum = 32;

// compile with `-lcublas`
int main()
{
    const int N = pow(GridDim, 3) * pow(BlockDim, 3) ;
    std::cout << "Number of elems: " << N << std::endl;

    float* h_dataA, * h_dataB, * h_dataC; 
    float * d_dataA, * d_dataB; // host and device data ptr
    
    cudaMallocHost(&h_dataA, N * sizeof(float));
    cudaMallocHost(&h_dataB, N * sizeof(float));
    cudaMallocHost(&h_dataC, N * sizeof(float));
    std::fill_n(h_dataA, N, 10.0); // represents A
    std::fill_n(h_dataB, N, 5.0); // represents B

    cublasHandle_t cuHandle;
    cublasStatus_t cublasCode;
    cublasCreate_v2(&cuHandle);

    auto timestamp_nano_start = std::chrono::system_clock::now().time_since_epoch().count();
    cudaMalloc(&d_dataA, N * sizeof(float));
    cudaMalloc(&d_dataB, N * sizeof(float));
    // first time cudaMemcpy invocation might be slow, next time shall be faster
    cublasCode = cublasSetVector(N, sizeof(float), h_dataA, 1, d_dataA, 1);
    cublasCode = cublasSetVector(N, sizeof(float), h_dataB, 1, d_dataB, 1);
    if (cublasCode != CUBLAS_STATUS_SUCCESS)
        std::cout << "cublasCodeErr: " << cublasCode << std::endl;
    auto timestamp_nano_end = std::chrono::system_clock::now().time_since_epoch().count();
    std::cout << "Cuda MemCpy to device elapsed time (second): " << (double)(timestamp_nano_end-timestamp_nano_start)/NanoSec << std::endl;

    timestamp_nano_start = std::chrono::system_clock::now().time_since_epoch().count();
    float alpha = 1.0;
    for (int i = 0; i < 1000; i++)
        cublasCode = cublasSaxpy_v2(cuHandle, N, &alpha, d_dataA, 1, d_dataB, 1);
    if (cublasCode != CUBLAS_STATUS_SUCCESS)
        std::cout << "cublasCodeErr: " << cublasCode << std::endl;
    timestamp_nano_end = std::chrono::system_clock::now().time_since_epoch().count();
    std::cout << "Cuda MatAdd elapsed time (second): " << (double)(timestamp_nano_end-timestamp_nano_start)/NanoSec << std::endl;

    timestamp_nano_start = std::chrono::system_clock::now().time_since_epoch().count();
    // copy back from gpu to host
    cublasGetVector(N, sizeof(float), d_dataB, 1, h_dataC, 1);
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
    
    cudaError cudaErr[3];
    cudaErr[0] = cudaFreeHost(h_dataA);
    cudaErr[1] = cudaFreeHost(h_dataB);
    cudaErr[2] = cudaFreeHost(h_dataC);
    for (auto eachErr : cudaErr)
    {
        if (eachErr != cudaSuccess)
        {
            std::cout << "Cuda host memory release failed" << std::endl;
        }
    }


    return 0;
}