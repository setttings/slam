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

void printCudaInfo();
double* cudaSetUp(int size, double* X);
int cudaFindNearestNeighbor(int size, double* X, double* Y, double* Z,
 double x, double y, double z);
int serialFindNearestNeighbor(int size, double* X, double* Y,
 double* Z, double x, double y, double z);



void loadFileWrapper(int i, double *X, double *Y, double *Z) {
  char buff[40];
  snprintf(buff, sizeof(buff), "../point_clouds/00000%05d.txt", i);

  std::string buffStr = buff;
  std::ifstream infile(buffStr.c_str());
  std::cout << "File name: " << buffStr << std::endl;
  double a, b, c ,d;
  int index = 0;
  while (infile >> a >> b >> c >> d) {
    X[index] = a;
    Y[index] = b;
    Z[index] = c;
    index++;
  }
  infile.close();
  return;
}


int main(int argc, char** argv) {
  printCudaInfo();
  // sum up total runtime for all NUM_ITER iterations and divide by NUM_ITER
  double cudaTimes = 0.0;
  double serialTimes = 0.0;
  double startTime;
  double endTime;

  // Read file 0
  int size = 120574;  // hardcoded for file 0
  double X[size];
  double Y[size];
  double Z[size];
  loadFileWrapper(0, X, Y, Z);

  // Read file 1
  int size2 = 120831;
  double X2[size2];
  double Y2[size2];
  double Z2[size2];
  loadFileWrapper(1, X2, Y2, Z2);
  
  // Pointers to arrays on GPU(?) memory
  startTime = CycleTimer::currentSeconds();
  double *XC = cudaSetUp(size, X);
  double *YC = cudaSetUp(size, Y);
  double *ZC = cudaSetUp(size, Z);
  endTime = CycleTimer::currentSeconds();
  std::cout << "Construction: " << endTime-startTime << std::endl;


  for (int i=0; i<size2; i++) {
    startTime = CycleTimer::currentSeconds();
    int cudaAns = cudaFindNearestNeighbor(size, XC, YC, ZC, X2[i], Y2[i], Z2[i]);
    endTime = CycleTimer::currentSeconds();
    cudaTimes = cudaTimes + (endTime-startTime);

    startTime = CycleTimer::currentSeconds();
    int serialAns = serialFindNearestNeighbor(size, X, Y, Z, X2[i], Y2[i], Z2[i]);
    endTime = CycleTimer::currentSeconds();
    serialTimes = serialTimes + (endTime-startTime);

    if (cudaAns!=serialAns) { 
      // Note: there is some difference in floating point precision when the code is run on CPU/GPU
      // Accept the answer if it doesn't differ by e
      double cuda = (X[cudaAns]-X2[i]*X[cudaAns]-X2[i]) + (Y[cudaAns]-Y2[i]*X[cudaAns]-Y2[i]) + (Z[cudaAns]-Z2[i]*Z[cudaAns]-Z2[i]);
      double serial = (X[serialAns]-X2[i]*X[serialAns]-X2[i]) + (Y[serialAns]-Y2[i]*X[serialAns]-Y2[i]) + (Z[serialAns]-Z2[i]*Z[serialAns]-Z2[i]);
      double diff = sqrt(cuda)-sqrt(serial);
      if (diff > e) {
        std::cout << "CUDA's answer: " << X[cudaAns] << " " << Y[cudaAns] << " " << Z[cudaAns] << std::endl;
        std::cout << "Serial answer: " << X[serialAns] << " " << Y[serialAns] << " " << Z[serialAns] << std::endl;
        std::cout << "CUDA: " << cudaAns << std::endl;
        std::cout << "Serial: " << serialAns << std::endl;
        return -1;
      }
    }
  }

  std::cout << "CUDA Time: " << cudaTimes << std::endl;
  std::cout << "Serial Time: " << serialTimes << std::endl;
  std::cout << "Speedup: " << serialTimes/cudaTimes << std::endl;
  return 0;
}
