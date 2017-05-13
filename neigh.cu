#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

#include <vector>
#include <cfloat>

#define THREADSPERBLOCK 512 
#define CHUNKSIZE 40


__global__ void findNearestNeighbor(float *X_0, float *Y_0, float *Z_0, float *X_1, float *Y_1, float *Z_1,
 int *startIndices, int *endIndices, int *rIndices) {
  // X_0, Y_0, Z_0 are from point cloud 0
  // X_1, Y_1, Z_1 are from point cloud 1

  __shared__ float distances[THREADSPERBLOCK];
  __shared__ int indices[THREADSPERBLOCK];

  int startIndex = startIndices[blockIdx.x];
  int endIndex = endIndices[blockIdx.x];
  indices[threadIdx.x] = startIndex+threadIdx.x;
  if (threadIdx.x > endIndex-startIndex+1) {
    distances[threadIdx.x] = -1.0; 
  } else {
    // Calculate distances from (x,y,z) to (X[i],Y[i],Z[i])
    // Store in distances[i]
    float tmp1 = X_0[startIndex+threadIdx.x]-X_1[blockIdx.x];
    float tmp2 = Y_0[startIndex+threadIdx.x]-Y_1[blockIdx.x];
    float tmp3 = Z_0[startIndex+threadIdx.x]-Z_1[blockIdx.x];
    distances[threadIdx.x] = sqrt(tmp1*tmp1 + tmp2*tmp2 + tmp3*tmp3);
  }
  __syncthreads();

  // Reduce over all the stuff
  // Note: for reduction to work, THREADSPERBLOCK must be a perfect square
  int nThreads = THREADSPERBLOCK;
  while(nThreads > 1) {
    int halfPoint = nThreads/2; // divide by two
    // only use the first half of the threads
   
    if (threadIdx.x < halfPoint) {
      int thread2 = threadIdx.x + halfPoint;
   
      // Get the shared value stored by another thread
      float temp = distances[thread2];
      if (temp < distances[threadIdx.x] & temp!=-1.0) {
        distances[threadIdx.x] = temp; 
        indices[threadIdx.x] = indices[thread2];
      }
    }
    __syncthreads();
   
    nThreads = halfPoint;
  }

  if (threadIdx.x==0) {
    rIndices[blockIdx.x] = indices[0];
  }
}


float *cudaSetUp(int arr_size, float *X) {
  // Copy arrays to device memory
  float *XC;
  cudaError_t mallocErr = cudaMalloc((void **)&XC, sizeof(float)*arr_size);
  if (mallocErr != cudaSuccess)
    printf("Malloc Error:  \"%s\".\n", cudaGetErrorString(mallocErr));

  // Copy all points from pointsInPies into t
  cudaError_t cpyErr = cudaMemcpy(XC, X, sizeof(float)*arr_size, cudaMemcpyHostToDevice);
  if (cpyErr != cudaSuccess)
    printf("cudaMemcpy Error 1: \"%s\".\n", cudaGetErrorString(cpyErr));
  return XC;
}

float *createFloatArray(int blocks) {
  float *distances;
  cudaMalloc((void **)&distances, sizeof(float)*blocks);
  return distances;
}

int *createIntArray(int blocks) {
  int *indices;
  cudaMalloc((void **)&indices, sizeof(int)*blocks);
  return indices;
}

void cudaFindNearestNeighbor(int N1, int *startIndex, int *endIndex, float* X_0, float* Y_0, float* Z_0, 
  float *X_1, float *Y_1, float *Z_1, int *indices, float *abc, float *def, float *ghi, int *mno, int *pqr, int *cudaAns) {
  int threadsPerBlock = THREADSPERBLOCK;
  int blocks = CHUNKSIZE; 
  // the intention now is to make the computation for each item in point cloud 1 fit into a single block  

  // Convert them arrays to device memory
  cudaError_t cpyErr;
  cpyErr=cudaMemcpy(abc, X_1, sizeof(float)*CHUNKSIZE, cudaMemcpyHostToDevice);
  cpyErr=cudaMemcpy(def, Y_1, sizeof(float)*CHUNKSIZE, cudaMemcpyHostToDevice);
  cpyErr=cudaMemcpy(ghi, Z_1, sizeof(float)*CHUNKSIZE, cudaMemcpyHostToDevice);
  cpyErr=cudaMemcpy(mno, startIndex, sizeof(int)*CHUNKSIZE, cudaMemcpyHostToDevice);
  cpyErr=cudaMemcpy(pqr, endIndex, sizeof(int)*CHUNKSIZE, cudaMemcpyHostToDevice);

  findNearestNeighbor<<<blocks, threadsPerBlock>>>(X_0, Y_0, Z_0, abc, def, ghi, mno, pqr, indices);
  cudaError_t launch = cudaGetLastError();
  if (launch != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(launch));
  cudaMemcpy(cudaAns, indices, sizeof(int)*blocks, cudaMemcpyDeviceToHost);
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
  // for fun, just print out some stats on the machine

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