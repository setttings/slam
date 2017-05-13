#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include "CycleTimer.h"
#include <vector>
#include <cmath>

#define e 0.0000000000001f
#define CHUNKSIZE 4

void printCudaInfo();
float* cudaSetUp(int size, float* X);
void cudaFindNearestNeighbor(int N0, int N1, float* X_0, float* Y_0, float* Z_0, 
  float *X_1, float *Y_1, float *Z_1, int start, int* results);
int serialFindNearestNeighbor(int size, float* X, float* Y,
 float* Z, float x, float y, float z);

void loadFileWrapper(int i, float *X, float *Y, float *Z) {
  char buff[40];
  snprintf(buff, sizeof(buff), "../point_clouds/00000%05d.txt", i);

  std::string buffStr = buff;
  std::ifstream infile(buffStr.c_str());
  std::cout << "File name: " << buffStr << std::endl;
  float a, b, c ,d;
  int index = 0;
  while (infile >> a >> b >> c >> d) {
    X[index] = a;
    Y[index] = b;
    Z[index] = c;
    index++;
  }
  return;
}


int main(int argc, char** argv) {
  printCudaInfo();
  // sum up total runtime for all NUM_ITER iterations and divide by NUM_ITER
  float cudaTimes = 0.0;
  float serialTimes = 0.0;
  float startTime;
  float endTime;

  // Read file 0
  int N0 = 120574; 
  float X[N0];
  float Y[N0];
  float Z[N0];
  loadFileWrapper(0, X, Y, Z);

  // Read file 1
  int N1 = 120831;
  float X2[N1];
  float Y2[N1];
  float Z2[N1];
  loadFileWrapper(1, X2, Y2, Z2);
  
  // Pointers to arrays on GPU(?) memory
  startTime = CycleTimer::currentSeconds();
  float *XC = cudaSetUp(N0, X);
  float *YC = cudaSetUp(N0, Y);
  float *ZC = cudaSetUp(N0, Z);
  // Convert the query array into cuda code too
  float *XC2 = cudaSetUp(N1, X2);
  float *YC2 = cudaSetUp(N1, Y2);
  float *ZC2 = cudaSetUp(N1, Z2);
  endTime = CycleTimer::currentSeconds();
  std::cout << "Construction: " << endTime-startTime << std::endl;

  int result[CHUNKSIZE];
  for (int i=0; i<N1; i+=CHUNKSIZE) {
    startTime = CycleTimer::currentSeconds();
    cudaFindNearestNeighbor(N0, N1, XC, YC, ZC, XC2, YC2, ZC2, i, result);
    endTime = CycleTimer::currentSeconds();
    cudaTimes = cudaTimes + (endTime-startTime);

    for (int j=0; j<CHUNKSIZE; j++) {
      startTime = CycleTimer::currentSeconds();
      int serialAns = serialFindNearestNeighbor(N0, X, Y, Z, X2[i+j], Y2[i+j], Z2[i+j]);
      endTime = CycleTimer::currentSeconds();
      serialTimes = serialTimes + (endTime-startTime);
      if (serialAns != result[j]) {
        std::cout << "res:" << i+j << " " << result[j] << " " << serialAns <<  std::endl;
      }
    }
  }

  std::cout << "CUDA Time: " << cudaTimes << std::endl;
  std::cout << "Serial Time: " << serialTimes << std::endl;
  std::cout << "Speedup: " << serialTimes/cudaTimes << std::endl;
  return 0;
}