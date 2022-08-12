#include <iostream>
#include <chrono>
#include <stdlib.h>
#include <unordered_map>
#include <thread>
#include <future>

#define NanoSec 1000000000.0

const int GridDim = 1 << 2;
const int BlockDim = 1 << 3;
const int ThreadNum = 32;
const int N = pow(GridDim, 3) * pow(BlockDim, 3) ;
const int kernelNum = 1 << 3;

__global__ void kernelSum(int n, float *x, float *y)
{
    int gridDim = GridDim >> 1;
    int idx = 
        gridDim*gridDim*blockDim.z*blockDim.y*blockDim.x*blockIdx.z + 
        gridDim*blockDim.z*blockDim.y*blockDim.x*blockIdx.y + 
        blockDim.z*blockDim.y*blockDim.x*blockIdx.x +
        threadIdx.x*blockDim.z*blockDim.y +
        threadIdx.y*blockDim.z +
        threadIdx.z;
    if (idx < n)
        y[idx] = x[idx] + y[idx];
}

int main(void)
{
    int streamNum = kernelNum*3;
    cudaStream_t streams[streamNum] ;
    for (int i = 0; i < streamNum; i++)
    {
        cudaStreamCreate(streams+i);
    }
    
    int eventNum = kernelNum*6;
    cudaEvent_t events[eventNum];
    for (int i = 0; i < eventNum; i++)
    {
        cudaEventCreate(events+i);
    }

    float *h_x, *h_y, *d_x, *d_y;
    cudaMallocHost(&h_x, N * sizeof(float));
    cudaMallocHost(&h_y, N * sizeof(float));
    std::fill_n(h_x, N, 10.0);
    std::fill_n(h_y, N, 5.0);

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    cudaError_t cudaErr;

    for (int i = 0; i < kernelNum; i++)
    {
        cudaErr = cudaMemcpyAsync(d_x + (N / kernelNum) * i, 
                        h_x + (N / kernelNum) * i, 
                        N * sizeof(float) / kernelNum, 
                        cudaMemcpyHostToDevice, streams[i]);
        cudaErr = cudaMemcpyAsync(d_y + (N / kernelNum) * i, 
                        h_y + (N / kernelNum) * i, 
                        N * sizeof(float) / kernelNum, 
                        cudaMemcpyHostToDevice, streams[i]);
        cudaErr = cudaEventRecord(events[i], streams[i]); // record events 
    }
    
    dim3 grid( GridDim>>1, GridDim>>1, GridDim>>1 );
    dim3 block( BlockDim, BlockDim, BlockDim );
    for (int i = 0; i < kernelNum; i++)
    {
        cudaErr = cudaStreamWaitEvent(streams[i], events[i+kernelNum]);    // wait for event in stream1 
        cudaEventRecord(events[i+kernelNum*2], streams[i+kernelNum]); // record events

        kernelSum<<<grid, block, 0, streams[i+kernelNum]>>>(N/kernelNum, 
                                            d_x + i * N / kernelNum, 
                                            d_y + i * N / kernelNum
                                            );
    }

    for (int i = 0; i < kernelNum; i++)
    {
        cudaStreamWaitEvent(streams[i+kernelNum], events[i+kernelNum*3]); // wait events
        cudaErr = cudaEventRecord(events[i+kernelNum*4], streams[i+kernelNum*2]); // record events

        cudaErr = cudaMemcpyAsync(h_y + (N / kernelNum) * i, 
                        d_y + (N / kernelNum) * i, 
                        N * sizeof(float) / kernelNum, 
                        cudaMemcpyDeviceToHost, streams[i+kernelNum*2]);
    }
    for (int i = 0; i < kernelNum; i++)
    {
        cudaStreamWaitEvent(streams[i+kernelNum*2], events[i+kernelNum*5]); // wait events
    }

    // The below code snippet is irrelevant to cuda, just a summary of results
    std::unordered_map<int,int> resultCountDictSum;
    auto threadRun = [&](std::unordered_map<int,int>& resultCountDict, int threadId, int blockRange)
    {
        for (int i = blockRange*threadId; i < blockRange*(threadId+1); i++)
        {
            if (resultCountDict.find(h_y[i]) != resultCountDict.end())
            {
                resultCountDict[h_y[i]]++;
            }
            else
            {
                resultCountDict[h_y[i]] = 0;
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
    
    cudaFree(d_y);
    cudaFree(d_x);

    cudaErr = cudaFreeHost(h_x);
    cudaErr = cudaFreeHost(h_y);

    return 0;
}