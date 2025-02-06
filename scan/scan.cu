#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

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

/* Helper function to round up to a power of 2.
 */
static inline int nextPow2(int n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

__global__ void upsweep_kernel(int *device_data, int twod1, int twod,
                               int length) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index == length - 1)
    device_data[index] = 0;
  else if (index % twod1 == 0 && index < length)
    device_data[index + twod1 - 1] += device_data[index + twod - 1];
}

__global__ void downsweep_kernel(int *device_data, int twod1, int twod,
                                 int length) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index % twod1 == 0 && index < length) {
    int t = device_data[index + twod - 1];
    device_data[index + twod - 1] = device_data[index + twod1 - 1];
    device_data[index + twod1 - 1] += t;
  }
}

void exclusive_scan(int *device_data, int length) {

  /* TODO
   * Fill in this function with your exclusive scan implementation.
   * You are passed the locations of the data in device memory
   * The data are initialized to the inputs.  Your code should
   * do an in-place scan, generating the results in the same array.
   * This is host code -- you will need to declare one or more CUDA
   * kernels (with the __global__ decorator) in order to actually run code
   * in parallel on the GPU.
   * Note you are given the real length of the array, but may assume that
   * both the data array is sized to accommodate the next
   * power of 2 larger than the input.
   */
  const int threadsPerBlock = 512;
  const int blocks = (length + threadsPerBlock - 1) / threadsPerBlock;
  for (int twod = 1; twod < length; twod *= 2) {
    int twod1 = twod * 2;
    upsweep_kernel<<<blocks, threadsPerBlock>>>(device_data, twod1, twod,
                                                length);
    cudaCheckError(cudaDeviceSynchronize());
  }
  // printf("we got:");
  // for (int i = 0; i < length; i++)
  //   printf("%d, ", device_data[i]);
  // printf("\n");
  // WARNING this is probaly wrong!!!
  // device_data[length - 1] = 0;

  for (int twod = length / 2; twod >= 1; twod /= 2) {
    int twod1 = twod * 2;
    /*
      saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y,
                                            device_result);
    */
    downsweep_kernel<<<blocks, threadsPerBlock>>>(device_data, twod1, twod,
                                                  length);
    cudaCheckError(cudaDeviceSynchronize());
  }
}

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan()
 * function above. You should not modify it.
 */
double cudaScan(int *inarray, int *end, int *resultarray) {
  printf("input:");
  for (int i = 0; i < (end - inarray); i++)
    printf("%d, ", inarray[i]);
  printf("\n");
  int *device_data;
  // We round the array size up to a power of 2, but elements after
  // the end of the original input are left uninitialized and not checked
  // for correctness.
  // You may have an easier time in your implementation if you assume the
  // array's length is a power of 2, but this will result in extra work on
  // non-power-of-2 inputs.
  int rounded_length = nextPow2(end - inarray);
  cudaMalloc((void **)&device_data, sizeof(int) * rounded_length);

  cudaMemcpy(device_data, inarray, (end - inarray) * sizeof(int),
             cudaMemcpyHostToDevice);

  double startTime = CycleTimer::currentSeconds();

  exclusive_scan(device_data, end - inarray);

  // Wait for any work left over to be completed.
  cudaDeviceSynchronize();
  double endTime = CycleTimer::currentSeconds();
  double overallDuration = endTime - startTime;

  cudaMemcpy(resultarray, device_data, (end - inarray) * sizeof(int),
             cudaMemcpyDeviceToHost);

  printf("output:");
  for (int i = 0; i < (end - inarray); i++)
    printf("%d, ", resultarray[i]);

  printf("\n");
  return overallDuration;
}

/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
double cudaScanThrust(int *inarray, int *end, int *resultarray) {

  int length = end - inarray;
  thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
  thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

  cudaMemcpy(d_input.get(), inarray, length * sizeof(int),
             cudaMemcpyHostToDevice);

  double startTime = CycleTimer::currentSeconds();

  thrust::exclusive_scan(d_input, d_input + length, d_output);

  cudaDeviceSynchronize();
  double endTime = CycleTimer::currentSeconds();

  cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
             cudaMemcpyDeviceToHost);
  thrust::device_free(d_input);
  thrust::device_free(d_output);
  double overallDuration = endTime - startTime;
  return overallDuration;
}

__global__ void lookLeftRight(int *device_input, int length,
                              char *device_oneHot, int *device_prefixOneHot) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index != 0 && index != length - 1) {
    if (device_input[index - 1] < device_input[index] &&
        device_input[index] > device_input[index + 1]) {
      device_oneHot[index] = 1;
      device_prefixOneHot[index] = 1;
    } else {
      device_oneHot[index] = 0;
      device_prefixOneHot[index] = 0;
    }
  }
  return;
}

__global__ void populateOutput(char *device_oneHot, int *device_prefixOneHot,
                               int *device_output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (device_oneHot[index] == 1) {
    int outputIndex = device_prefixOneHot[index];
    device_output[outputIndex] = index;
  }
  return;
}

int find_peaks(int *device_input, int length, int *device_output) {

  /* TODO:
   * Finds all elements in the list that are greater than the elements before
   * and after, storing the index of the element into device_result. Returns the
   * number of peak elements found. By definition, neither element 0 nor element
   * length-1 is a peak.
   *
   * Your task is to implement this function. You will probably want to
   * make use of one or more calls to exclusive_scan(), as well as
   * additional CUDA kernel launches.
   * Note: As in the scan code, we ensure that allocated arrays are a power
   * of 2 in size, so you can use your exclusive_scan function with them if
   * it requires that. However, you must ensure that the results of
   * find_peaks are correct given the original length.
   */
  const int threadsPerBlock = 512;
  const int blocks = (length + threadsPerBlock - 1) / threadsPerBlock;
  char *device_oneHot;
  int *device_prefixOneHot;
  cudaCheckError(cudaMalloc(&device_oneHot, length * sizeof(char)));
  cudaCheckError(cudaMalloc(&device_prefixOneHot, length * sizeof(int)));
  lookLeftRight<<<blocks, threadsPerBlock>>>(
      device_input, length, device_oneHot, device_prefixOneHot);

  cudaDeviceSynchronize();

  exclusive_scan(device_prefixOneHot, length);
  populateOutput<<<blocks, threadsPerBlock>>>(
      device_oneHot, device_prefixOneHot, device_output);

  cudaDeviceSynchronize();
  int outputLength = device_prefixOneHot[length - 1];

  cudaCheckError(cudaFree(device_oneHot));
  cudaCheckError(cudaFree(device_prefixOneHot));

  return outputLength;
}

/* Timing wrapper around find_peaks. You should not modify this function.
 */
double cudaFindPeaks(int *input, int length, int *output, int *output_length) {
  int *device_input;
  int *device_output;
  int rounded_length = nextPow2(length);
  cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
  cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
  cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

  double startTime = CycleTimer::currentSeconds();

  int result = find_peaks(device_input, length, device_output);

  cudaDeviceSynchronize();
  double endTime = CycleTimer::currentSeconds();

  *output_length = result;

  cudaMemcpy(output, device_output, length * sizeof(int),
             cudaMemcpyDeviceToHost);

  cudaFree(device_input);
  cudaFree(device_output);

  return endTime - startTime;
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
