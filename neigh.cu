#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "CycleTimer.h"
#include <vector>
#include <cfloat>

#define THREADSPERBLOCK 1024 

__device__ double x;
__device__ double y;
__device__ double z;

__global__ void findNearestNeighbor(double *X, double *Y, double *Z, double *res, int *resIndices, int N) {
  __shared__ double distances[THREADSPERBLOCK];
  __shared__ int indices[THREADSPERBLOCK];

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  indices[threadIdx.x] = tid;
  if (tid >= N) {
    distances[threadIdx.x] = -1.0; 
  } else {
    // Calculate distances from (x,y,z) to (X[i],Y[i],Z[i])
    // Store in distances[i]
    double tmp1 = X[tid]-x;
    double tmp2 = Y[tid]-y;
    double tmp3 = Z[tid]-z;
    distances[threadIdx.x] = sqrt(tmp1*tmp1 + tmp2*tmp2 + tmp3*tmp3);
  }
  __syncthreads();

  // Reduce over all the arrays in shared memory
  // Note: for reduction to work, THREADSPERBLOCK must be a perfect square
  int nThreads = THREADSPERBLOCK;
  while(nThreads > 1) {
    int halfPoint = nThreads/2; // divide by two
    // only use the first half of the threads
   
    if (threadIdx.x < halfPoint) {
      int thread2 = threadIdx.x + halfPoint;
   
      // Get the shared value stored by another thread
      double temp = distances[thread2];
      if (temp < distances[threadIdx.x] & temp!=-1.0) {
        distances[threadIdx.x] = temp; 
        indices[threadIdx.x] = indices[thread2];
      }
    }
    __syncthreads();
   
    nThreads = halfPoint;
  }

  if (threadIdx.x==0) {
    resIndices[blockIdx.x] = indices[0];
    res[blockIdx.x] = distances[0];
  }

}

// Code to do the reduction in CUDA instead of CPU. 
__global__ void reduceFindMin(double *distances, int *indices, int N) {
  __shared__ double distancesRounded[THREADSPERBLOCK];
  __shared__ int indicesRounded[THREADSPERBLOCK];

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= N) {
    distancesRounded[threadIdx.x] = -1.0;
    indicesRounded[threadIdx.x] = threadIdx.x;
  } else {
    distancesRounded[threadIdx.x] = distances[tid];
    indicesRounded[threadIdx.x] = indices[tid];
  }
  __syncthreads();
  
  int nThreads = THREADSPERBLOCK;
  while(nThreads > 1) {
    int halfPoint = nThreads/2; // divide by two
    // only the first half of the threads will be active.
   
    if (threadIdx.x < halfPoint) {
      int thread2 = threadIdx.x + halfPoint;
   
      // Get the shared value stored by another thread
      double temp = distancesRounded[thread2];
      if (temp <= distancesRounded[threadIdx.x] & temp!=-1.0) {
        distancesRounded[threadIdx.x] = temp; 
        indicesRounded[threadIdx.x] = indicesRounded[thread2];
      }
    }
    __syncthreads();
   
    nThreads = halfPoint;
  }

  if (threadIdx.x==0) {
    indices[blockIdx.x] = indicesRounded[0];
    distances[blockIdx.x] = distancesRounded[0];
  }
}

double* cudaSetUp(int size, double* X) {
  // Copy arrays to device memory
  double *XC;
  cudaMalloc((void **)&XC, sizeof(double)*size);
  cudaMemcpy(XC, X, sizeof(double)*size, cudaMemcpyHostToDevice);
  return XC;
}

int cudaFindNearestNeighbor(int size, double* X, double* Y, double* Z, 
  double xCoord, double yCoord, double zCoord) {
  // Copy (x,y,z) we're querying to device memory
  cudaMemcpyToSymbol(x, (void **)&xCoord, sizeof(double),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(y, (void **)&yCoord, sizeof(double),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(z, (void **)&zCoord, sizeof(double),0,cudaMemcpyHostToDevice);

  int threadsPerBlock = 1024; // just copied these parts from assignment 2
  int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

  // Store the minimum distance + corresponding index from each thread block 
  double *distances;
  cudaMalloc((void **)&distances, sizeof(double)*blocks);
  int *indices;
  cudaMalloc((void **)&indices, sizeof(int)*blocks);

  findNearestNeighbor<<<blocks, threadsPerBlock>>>(X, Y, Z, distances, indices, size);

  // Option 1: Do reduction in kernel
  // while (blocks > 1) {
  //   int N = blocks;
  //   blocks = (blocks/threadsPerBlock)+1; 
  //   reduceFindMin<<<blocks, threadsPerBlock>>>(distances, indices, N);
  // }
  // int index;
  // cudaMemcpy(&index, indices, sizeof(int), cudaMemcpyDeviceToHost);
  // Freeing memory made this too slow so let's just be rebels...
  // return index;

  // Option 2: Reduction  on CPU
  // Reduction in CPU is faster when there's ~120 elements, as is the case here
  double distances2[blocks];
  cudaMemcpy(distances2, distances, sizeof(double)*blocks, cudaMemcpyDeviceToHost);
  int indices2[blocks];
  cudaMemcpy(indices2, indices, sizeof(int)*blocks, cudaMemcpyDeviceToHost);

  double minSeen = FLT_MAX;
  int minIndex = -1;
  for (int i=0; i < blocks; i++) {
    if (minSeen > distances2[i]) {
      minSeen = distances2[i];
      minIndex = indices2[i];
    }
  }
  return minIndex;
  
}

int serialFindNearestNeighbor(int size, double* X, double* Y,
 double* Z, double x, double y, double z) {
  double distances[size];
  for (int i=0; i<size; i++) {
    // Cauclate the distances
    double tmp1 = X[i]-x;
    double tmp2 = Y[i]-y;
    double tmp3 = Z[i]-z;
    double dist = sqrt(tmp1*tmp1 + tmp2*tmp2 + tmp3*tmp3);
    distances[i] = dist;
  }
  double minSeen = FLT_MAX;
  double minIndex = -1;
  for (int i=0; i<size; i++) {
    if (minSeen > distances[i]) {
      minSeen = distances[i];
      minIndex = i;
    }
  }
  return minIndex;
}

void printCudaInfo() {
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  printf("---------------------------------------------------------\n");
  printf("Found %d CUDA devices\n", deviceCount);

  for (int i=0; i<deviceCount; i++)
  {
      cudaDeviceProp deviceProps;
      cudaGetDeviceProperties(&deviceProps, i);
      printf("Device %d: %s\n", i, deviceProps.name);
      printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
      printf("   Global mem: %.0f MB\n",
             static_cast<double>(deviceProps.totalGlobalMem) / (1024 * 1024));
      printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
  }
  printf("---------------------------------------------------------\n"); 
  return;
}