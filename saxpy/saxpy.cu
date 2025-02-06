#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
#else
#define cudaCheckError(ans) ans
#endif

extern float toBW(int bytes, float sec);

__global__ void saxpy_kernel(int N, float alpha, float *x, float *y,
                             float *result) {

  // compute overall index from position of thread in current block,
  // and given the block we are in
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < N)
    result[index] = alpha * x[index] + y[index];
}

void saxpyCuda(int N, float alpha, float *xarray, float *yarray,
               float *resultarray) {

  int totalBytes = sizeof(float) * 3 * N;

  // compute number of blocks and threads per block
  const int threadsPerBlock = 512;
  const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

  float *device_x;
  float *device_y;
  float *device_result;
  int bytes = sizeof(float) * N;

  //
  // TODO allocate device memory buffers on the GPU using cudaMalloc
  //
  cudaCheckError(cudaMalloc(&device_x, bytes));
  cudaCheckError(cudaMalloc(&device_y, bytes));
  cudaCheckError(cudaMalloc(&device_result, bytes));

  // start timing after allocation of device memory
  double startTime = CycleTimer::currentSeconds();

  //
  // TODO copy input arrays to the GPU using cudaMemcpy
  //
  cudaCheckError(cudaMemcpy(device_x, xarray, bytes, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(device_y, yarray, bytes, cudaMemcpyHostToDevice));

  double startTime2 = CycleTimer::currentSeconds();

  // run kernel
  saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y,
                                            device_result);
  cudaDeviceSynchronize();
  double endTime2 = CycleTimer::currentSeconds();

  //
  // TODO copy result from GPU using cudaMemcpy
  //
  cudaCheckError(
      cudaMemcpy(resultarray, device_result, bytes, cudaMemcpyDeviceToHost));
  // end timing after result has been copied back into host memory
  double endTime = CycleTimer::currentSeconds();

  cudaError_t errCode = cudaPeekAtLastError();
  if (errCode != cudaSuccess) {
    fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode,
            cudaGetErrorString(errCode));
  }

  double overallDuration = endTime - startTime;
  double overallDuration2 = endTime2 - startTime2;

  printf("Overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration,
         toBW(totalBytes, overallDuration));

  printf("Our Overall2: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration2,
         toBW(totalBytes, overallDuration2));

  // TODO free memory buffers on the GPU

  cudaCheckError(cudaFree(device_x));
  cudaCheckError(cudaFree(device_y));
  cudaCheckError(cudaFree(device_result));
}

void printCudaInfo() {

  // for fun, just print out some stats on the machine

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  printf("---------------------------------------------------------\n");
  printf("Found %d CUDA devices\n", deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);
    printf("Device %d: %s\n", i, deviceProps.name);
    printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
    printf("   Global mem: %.0f MB\n",
           static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
    printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
  }
  printf("---------------------------------------------------------\n");
}
