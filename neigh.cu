#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "CycleTimer.h"
#include <vector>
#include <cfloat>

#define THREADSPERBLOCK 512 
#define CHUNKSIZE 4


__global__ void findNearestNeighbor(float *X_0, float *Y_0, float *Z_0, float *X_1, float *Y_1, float *Z_1, 
  int start, int N0, int N1, float *rDistances, int *rIndices) {
  // X_0, Y_0, Z_0 are from point cloud 0
  // X_1, Y_1, Z_1 are from point cloud 1
  // In each thread, finds the minimum distance for PC1[start,....,start+CHUNKSIZE] to
  // points PC0[tid_0,...,tid_blocksize]
  // Stores the corresponding distance in distances[]
  // And corresponding index in indices[]
  __shared__ float distances[THREADSPERBLOCK*CHUNKSIZE];
  __shared__ int indices[THREADSPERBLOCK*CHUNKSIZE];

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N0) {
    for (int i=0; ((i<CHUNKSIZE)); i++) {
      distances[threadIdx.x*CHUNKSIZE+i] = -1.0;
      indices[threadIdx.x*CHUNKSIZE+i] = tid;
    }
  } else {
    for (int i=0; ((i<CHUNKSIZE)); i++) {
      indices[threadIdx.x*CHUNKSIZE+i] = tid;
      // Calculate the distance
      float tmp1 = X_0[tid]-X_1[start+i];
      float tmp2 = Y_0[tid]-Y_1[start+i];
      float tmp3 = Z_0[tid]-Z_1[start+i];
      distances[threadIdx.x*CHUNKSIZE+i] = sqrt(tmp1*tmp1 + tmp2*tmp2 + tmp3*tmp3);
    }
  }
  __syncthreads();
  // Begin reduction within thread block
  // THREADSPERBLOCK must be a prefect square
  for (int i=0; (i<CHUNKSIZE&(start+i<N1)); i++) {
    int n = THREADSPERBLOCK; 
    while (n > 1) {
      int half = n/2;
      // Activate the first half of threads only
      if (threadIdx.x < half) {
        int thread2 = threadIdx.x + half;
        float dist2 = distances[thread2*CHUNKSIZE+i];
        float dist1 = distances[threadIdx.x*CHUNKSIZE+i];
        // The other value is smaller, do some replacement
        if ((dist2<=dist1) & (dist2 != -1.0)) {
          distances[threadIdx.x*CHUNKSIZE+i] = dist2;
          indices[threadIdx.x*CHUNKSIZE+i] = indices[thread2*CHUNKSIZE+i];
        }
      }
      __syncthreads();
      n = half;
    }
    __syncthreads();
    if (threadIdx.x==0) {
      // Store this in the result array
      rDistances[blockIdx.x*CHUNKSIZE+i] = distances[i];
      rIndices[blockIdx.x*CHUNKSIZE+i] = indices[i];
    }
  }
}

float* cudaSetUp(int size, float* X) {
  // Copy arrays to device memory
  float *XC;
  cudaMalloc((void **)&XC, sizeof(float)*size);
  cudaMemcpy(XC, X, sizeof(float)*size, cudaMemcpyHostToDevice);
  return XC;
}

void cudaFindNearestNeighbor(int N0, int N1, float* X_0, float* Y_0, float* Z_0, 
  float *X_1, float *Y_1, float *Z_1, int start, int* results) {
  int threadsPerBlock = THREADSPERBLOCK;
  int blocks = (N0 + threadsPerBlock - 1) / threadsPerBlock;

  // Store the minimum distance + corresponding index from each thread block 
  float *distances;
  cudaMalloc((void **)&distances, sizeof(float)*blocks*CHUNKSIZE);
  int *indices;
  cudaMalloc((void **)&indices, sizeof(int)*blocks*CHUNKSIZE);

  findNearestNeighbor<<<blocks, threadsPerBlock>>>(X_0, Y_0, Z_0, X_1, Y_1, Z_1, start, N0, N1, distances, indices);
  cudaError_t launch = cudaGetLastError();
  if (launch != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(launch));
  cudaError_t cudaerr =  cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

  float rDistances[blocks*CHUNKSIZE];
  cudaMemcpy(rDistances, distances, sizeof(float)*blocks*CHUNKSIZE, cudaMemcpyDeviceToHost);
  int rIndices[blocks*CHUNKSIZE];
  cudaMemcpy(rIndices, indices, sizeof(int)*blocks*CHUNKSIZE, cudaMemcpyDeviceToHost);

  for (int i=0; i < CHUNKSIZE; i++) {
    int minIndex = -1;
    float minSeen = FLT_MAX;
    for (int j=0; j<blocks; j++) {
      if (minSeen >= rDistances[j*CHUNKSIZE+i]) {
        minIndex = rIndices[j*CHUNKSIZE+i];
        minSeen = rDistances[j*CHUNKSIZE+i];
      }
    }
    results[i] = minIndex;
  }

  return;
}

int serialFindNearestNeighbor(int size, float* X, float* Y,
 float* Z, float x, float y, float z) {
  float distances[size];
  for (int i=0; i<size; i++) {
    // Cauclate the distances
    float tmp1 = X[i]-x;
    float tmp2 = Y[i]-y;
    float tmp3 = Z[i]-z;
    float dist = sqrt(tmp1*tmp1 + tmp2*tmp2 + tmp3*tmp3);
    distances[i] = dist;
  }
  float minSeen = FLT_MAX;
  float minIndex = -1;
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
      printf("Shared memory (kb): %d\n", deviceProps.sharedMemPerBlock);
      printf("   Global mem: %.0f MB\n",
             static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
      printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
  }
  printf("---------------------------------------------------------\n"); 
  return;
}