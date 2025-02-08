#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <random>
#include <memory>
#include <unistd.h>


#define ImageWidth 640
#define ImageHeight 640
#define ImageTotalBytes ImageWidth*ImageHeight*sizeof(uint8_t) // size corresponds to CV_8UC1
#define ImageTotalFloatBytes ImageWidth*ImageHeight*sizeof(float) // size corresponds to CV_32FC1

#define kTimesSimulation 90
#define AnimationFrames 10
#define TempMax 255
#define TempMin 1
#define K 0.02

dim3 grids(ImageWidth/16, ImageHeight/16);
dim3 threads(16, 16);

cudaError_t err;

// internal computation should use float against uint8_t for higher precision computatioin
__global__ void copyInit_kernel(float *iptr, const uint8_t *cptr)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if (cptr[offset] != 0)
        iptr[offset] = cptr[offset]/255;
}

__global__ void castColor_kernel(uchar *outSrc, const float *inSrc)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    outSrc[offset] = static_cast<uchar>(static_cast<uint8_t>(inSrc[offset]*255));
}

__global__ void blend_kernel(float *outSrc, const float *inSrc)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int left = offset - 1;
    int right = offset + 1;
    if (x == 0) left++;
    if (x == ImageWidth-1) right--;

    int up = offset - ImageWidth;
    int down = offset + ImageWidth;
    if (y == 0) up += ImageWidth;
    if (y == ImageHeight-1) down -= ImageWidth;

    float neighborSum = inSrc[left] + inSrc[right] + inSrc[up] + inSrc[down] ;
    float reduceOld = 4 * inSrc[offset];
    float update = K * (neighborSum - reduceOld );

    outSrc[offset] = inSrc[offset] + update > 1.0f ? 1.0f : inSrc[offset] + update;

}

struct DataBlock
{
    DataBlock(){
        err = cudaMalloc((void**)&pixel_gpu, ImageTotalBytes);
        if (err != 0) {
            std::cout << "cudaMalloc err" << std::endl;
            std::cout << cudaGetErrorString ( err ) << std::endl;
        }

        err = cudaMalloc((void**)&dev_inSrc, ImageTotalFloatBytes);
        if (err != 0) {
            std::cout << "cudaMalloc err" << std::endl;
            std::cout << cudaGetErrorString ( err ) << std::endl;
        }

        err = cudaMalloc((void**)&dev_outSrc, ImageTotalFloatBytes);
        if (err != 0) {
            std::cout << "cudaMalloc err" << std::endl;
            std::cout << cudaGetErrorString ( err ) << std::endl;
        }

        err = cudaMalloc((void**)&dev_initSrc, ImageTotalBytes);
        if (err != 0) {
            std::cout << "cudaMalloc err" << std::endl;
            std::cout << cudaGetErrorString ( err ) << std::endl;
        }

        pixel_cpu = (uchar*)malloc(ImageTotalBytes);
    }

    ~DataBlock(){
        cudaFree(pixel_gpu);
        cudaFree(dev_inSrc);
        cudaFree(dev_outSrc);
        cudaFree(dev_initSrc);
        free(pixel_cpu);
    }

    uchar *pixel_cpu;
    uchar *pixel_gpu;

    float *dev_inSrc;
    float *dev_outSrc;
    uint8_t *dev_initSrc;

};

void animate_gpu ( DataBlock *d )
{
    for (int i = 0; i < kTimesSimulation; i++)
    {
        blend_kernel<<<grids, threads>>>( d->dev_outSrc, d->dev_inSrc);
        std::swap(d->dev_inSrc, d->dev_outSrc);
    }
    castColor_kernel<<<grids, threads>>>(d->pixel_gpu, d->dev_inSrc);

    err = cudaMemcpy(d->pixel_cpu, d->pixel_gpu, ImageTotalBytes, cudaMemcpyDeviceToHost);
    if (err != 0) {
        std::cout << "pixel_gpu cudaMemcpy err" << std::endl;
        std::cout << cudaGetErrorString ( err ) << std::endl;
    }
    
}

// Manually create a heat map with two heat areas, then return this heat map
uint8_t * initHeatMap()
{
    uint8_t *tmpHeatMap = (uint8_t*)malloc(ImageTotalBytes);
    for (int i=0; i < ImageTotalBytes; i++)
    {
        tmpHeatMap[i] = TempMin;
        int x = i % ImageWidth;
        int y = i / ImageHeight;
        if ( (x>ImageWidth/8) && (x<ImageWidth/4) && (y>ImageHeight/8) && (y<ImageHeight/4) )
            tmpHeatMap[i] = TempMax;
        if ( (x>(ImageWidth/8)*5) && (x<(ImageWidth/8)*7) && (y>(ImageHeight/8)*5) && (y<(ImageHeight/8)*7) )
            tmpHeatMap[i] = TempMax;
    }
    return tmpHeatMap;
}

int main()
{
    DataBlock data;

    uint8_t *heapMapCpu = initHeatMap();

    err = cudaMemcpy(data.dev_initSrc, heapMapCpu,  ImageTotalBytes, cudaMemcpyHostToDevice);
    if (err != 0) {
        std::cout << "heapMapCpu cudaMemcpy err" << std::endl;
        std::cout << cudaGetErrorString ( err ) << std::endl;
    }

    copyInit_kernel<<<grids, threads>>>(data.dev_inSrc, data.dev_initSrc);
    
    cv::Mat img(ImageWidth, ImageHeight, CV_8UC1, heapMapCpu);
    cv::imshow("heat_transfer", img);
    cv::waitKey(300);
    for (int i = 0; i < AnimationFrames; i++) {
        animate_gpu(&data);
        img = cv::Mat(ImageWidth, ImageHeight, CV_8UC1, data.pixel_cpu);
        cv::imshow("heat_transfer", img);
        int key = cv::waitKey(300);
    }

    free(heapMapCpu);

}