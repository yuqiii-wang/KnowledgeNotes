#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <random>
#include <memory>

#define ImageWidth 640
#define ImageHeight 640
#define INF 2e10f // if a ray hits something
#define SphereNums 20
#define rnd(x) (x*rand() / RAND_MAX)
#define ImageTotalBytes ImageWidth*ImageHeight*3*sizeof(uchar) // size corresponds to CV_8UC3

struct Sphere {
    uint8_t r,g,b; // sphere rgb color value
    float x,y,z, radius; // sphere centroid (x,y,z) and radius

    __device__ float hit(float ox, float oy, float *n) {
        float dx = ox - x;
        float dy = oy - y;
        float dz;
        if (dx*dx + dy*dy < radius*radius ) { 
            dz = sqrtf(radius*radius - (dx*dx + dy*dy));
            *n = dz / sqrtf(radius*radius); // n is a gradient disappearance scale, that
                                            // when ray hit the edge of an object, it should
                                            // show gray color rather than the bright
        }
        else {
            return -INF;
        }
        return dz + z;
    }
};

__global__ void kernel(uchar* imgGpuData, Sphere *sphGpu)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float ox = x - ImageWidth/2;
    float oy = y - ImageHeight/2;

    uint8_t r=0, g=0, b=0;
    float zMax = -INF;

    for (int i=0; i < SphereNums; i++) {
        float n;
        float t = sphGpu[i].hit(ox, oy, &n);
        if (t > zMax) {
            r  = sphGpu[i].r * n;
            g  = sphGpu[i].g * n;
            b  = sphGpu[i].b * n;
        }
    }

    imgGpuData[offset * 3 + 0] = static_cast<uchar>((uint8_t)r);
    imgGpuData[offset * 3 + 1] = static_cast<uchar>((uint8_t)g);
    imgGpuData[offset * 3 + 2] = static_cast<uchar>((uint8_t)b);

}

Sphere* initSphereInCpu()
{
    Sphere* tmp_s = (Sphere*)malloc(sizeof(Sphere)*SphereNums);
    for (int i = 0; i < SphereNums; i++)
    {
        tmp_s[i].r = rnd(1.0f)*255;
        tmp_s[i].g = rnd(1.0f)*255;
        tmp_s[i].b = rnd(1.0f)*255;
        float centroidX = ImageWidth;
        float centroidY = ImageHeight;
        tmp_s[i].x = rnd(centroidX) - ImageWidth/2;
        tmp_s[i].y = rnd(centroidY) - ImageHeight/2;
        tmp_s[i].z = rnd(100.0f) - 100;
        tmp_s[i].radius = std::abs(rnd(100.0f));
    }
    return tmp_s;
}

int main()
{
    uchar* imgCpuData = (uchar*)malloc(ImageTotalBytes);
    memset(imgCpuData, 0, ImageTotalBytes);
    uchar* imgGpuData;

    Sphere *sphCpu, *sphGpu;
    sphCpu = initSphereInCpu();

    cudaError_t err;

    err = cudaMalloc((void**)&imgGpuData, ImageTotalBytes); 
    if (err != 0) {
        std::cout << "imgGpuData cudaMalloc err" << std::endl;
        std::cout << cudaGetErrorString ( err ) << std::endl;
    }
    err = cudaMalloc((void**)&sphGpu, sizeof(Sphere)*SphereNums);
    if (err != 0) {
        std::cout << "sphGpu cudaMalloc err" << std::endl;
        std::cout << cudaGetErrorString ( err ) << std::endl;
    }
    err = cudaMemcpy(sphGpu, sphCpu, sizeof(Sphere)*SphereNums, cudaMemcpyHostToDevice);
    if (err != 0) {
        std::cout << "imgGpuData cudaMemcpy back err" << std::endl;
        std::cout << cudaGetErrorString ( err ) << std::endl;
    }

    dim3 grids(ImageWidth/16, ImageHeight/16);
    dim3 threads(16, 16);
    kernel<<<grids, threads>>>(imgGpuData, sphGpu);

    err = cudaMemcpy(imgCpuData, imgGpuData, ImageTotalBytes, cudaMemcpyDeviceToHost);
    if (err != 0) {
        std::cout << "imgGpuData cudaMemcpy back err" << std::endl;
        std::cout << cudaGetErrorString ( err ) << std::endl;
    }

    cv::Mat img(ImageWidth, ImageHeight, CV_8UC3, imgCpuData);

    cv::imshow("ray_tracing", img);
    cv::waitKey(0);

    free(imgCpuData);
    free(sphCpu);
    cudaFree(imgGpuData);
    cudaFree(sphGpu);

    return 0;
}