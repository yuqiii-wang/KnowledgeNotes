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

texture<float, 2> devTex_inSrc;
texture<float, 2> devTex_outSrc;

cudaError_t err;


struct DataBlock
{
    DataBlock(){
        descFloat = cudaCreateChannelDesc<float>(); 

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
        err = cudaBindTexture2D(NULL, devTex_inSrc, dev_inSrc, descFloat, ImageWidth, ImageHeight, ImageWidth*sizeof(float) );
        if (err != 0) {
            std::cout << "cudaBindTexture2D devTex_inSrc err" << std::endl;
            std::cout << cudaGetErrorString ( err ) << std::endl;
        }

        err = cudaMalloc((void**)&dev_outSrc, ImageTotalFloatBytes);
        if (err != 0) {
            std::cout << "cudaMalloc err" << std::endl;
            std::cout << cudaGetErrorString ( err ) << std::endl;
        }
        err = cudaBindTexture2D(NULL, devTex_outSrc, dev_outSrc, descFloat, ImageWidth, ImageHeight, ImageHeight*sizeof(float));
        if (err != 0) {
            std::cout << "cudaBindTexture2D devTex_outSrc err" << std::endl;
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
        cudaUnbindTexture(devTex_inSrc);
        cudaUnbindTexture(devTex_outSrc);
        cudaFree(pixel_gpu);
        cudaFree(dev_initSrc);
        cudaFree(dev_inSrc);
        cudaFree(dev_outSrc);
        free(pixel_cpu);
    }

    uchar *pixel_cpu;
    uchar *pixel_gpu;

    float *dev_inSrc;
    float *dev_outSrc;
    uint8_t *dev_initSrc;

    cudaChannelFormatDesc descFloat;

};

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

__global__ void blend_kernel( float *outSrc, bool isUseInSrc )
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int left_idx = x-1<0?0:x-1;
    int right_idx = x+1>ImageWidth-1?ImageWidth-1:x+1;
    int up_idx = y+1>ImageHeight-1?ImageHeight-1:y+1;
    int down_idx = y-1<0?0:y-1;

    float left, right, up, down, self;
    if (isUseInSrc) {
        left = tex2D(devTex_inSrc, left_idx, y);
        right = tex2D(devTex_inSrc, right_idx, y);
        up = tex2D(devTex_inSrc, x, up_idx);
        down = tex2D(devTex_inSrc, x, down_idx);
        self = tex2D(devTex_inSrc, x, y);
    }
    else {
        left = tex2D(devTex_outSrc, left_idx, y);
        right = tex2D(devTex_outSrc, right_idx, y);
        up = tex2D(devTex_outSrc, x, up_idx);
        down = tex2D(devTex_outSrc, x, down_idx);
        self = tex2D(devTex_outSrc, x, y);
    }

    outSrc[offset] = self + K * (left + right + up + down - 4 * self);

}

void animate_gpu ( DataBlock *d )
{
    for (int i = 0; i < kTimesSimulation; i++)
    {
        blend_kernel<<<grids, threads>>>( d->dev_outSrc, i % 2 == 0 );
        std::swap(d->dev_inSrc, d->dev_outSrc);
    }
    castColor_kernel<<<grids, threads>>>(d->pixel_gpu, d->dev_outSrc);

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