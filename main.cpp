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
#include <cassert>
#include <math.h>


#define e 0.000000000000001f
#define CHUNKSIZE 40
#define PI 3.14159265359
#define N_BINS 300

typedef struct{
  float x;
  float y;
  float z;
} PointType;
typedef struct{
  std::vector<PointType> points;
} PointCloudType;


void printCudaInfo();
float *cudaSetUp(int arr_size, float *X);
void cudaFindNearestNeighbor(int N1, int *startIndex, int *endIndex, float* X_0, float* Y_0, float* Z_0, 
  float *X_1, float *Y_1, float *Z_1, int *indices, float *abc, float *def, float *ghi, int *jkl, int *mno, int *cudaAns);
int serialFindNearestNeighbor(int size, float* X, float* Y,
 float* Z, float x, float y, float z);
int *createIntArray(int blocks);
float *createFloatArray(int blocks);



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
  infile.close();
  return;
}

void load_point_cloud_file(int i, PointCloudType* point_cloud) {
  point_cloud->points.clear();
  char buff[40];
  snprintf(buff, sizeof(buff), "../point_clouds/00000%05d.txt", i);

  std::string buffStr = buff;
  std::ifstream infile(buffStr.c_str());
  std::cout << "File name: " << buffStr << std::endl;

  float x,y,z,a;

  while (infile >> x >> y >> z >> a) {
    PointType X;
    X.x = x;
    X.y = y;
    X.z = z;
    point_cloud->points.push_back(X);
  }

  std::cout << "Point cloud has " << point_cloud->points.size() << " points" << std::endl;
  infile.close();
}

void computeAzimuthAndAltitude(PointType& point, float& azimuth, float& altitude) {
  float x = point.x;
  float y = point.y;
  float z = point.z;
  azimuth = atan2(y, x);
  altitude = atan2(z, sqrt(x*x + y*y));
}

float degrees(float radians) {
  return radians/PI*180;
}

int computeBin(PointType& point) {
  float azimuth, altitude;
  computeAzimuthAndAltitude(point, azimuth, altitude);
  int bin = int((degrees(azimuth) + 180) / (360.0 / N_BINS));
  if (bin == N_BINS) bin = N_BINS - 1;
  return bin;
}

void readAnswers(int *answers) {
  std::ifstream infile("answers.txt");
  float answer;
  int i = 0;
  while (infile >> answer) {
    answers[i] = answer;
    i++;
  }
  infile.close();
}



int main(int argc, char** argv) {
  printCudaInfo();
  // sum up total runtime for all NUM_ITER iterations and divide by NUM_ITER
  float cudaTimes = 0.0;
  float serialTimes = 0.0;
  float startTime;
  float endTime;

  assert(argc == 3);

  PointCloudType* pc0(new PointCloudType);
  PointCloudType* pc1(new PointCloudType);
  load_point_cloud_file(atoi(argv[1]), pc0);
  load_point_cloud_file(atoi(argv[2]), pc1);
  int N0 = 120574;
  int N1 = 120831;
  float X_1[N0];
  float Y_1[N0];
  float Z_1[N0];
  loadFileWrapper(0, X_1, Y_1, Z_1);

  startTime = CycleTimer::currentSeconds();
  // Place points into bins
  std::vector<float> pointsInPiesX[N_BINS];
  std::vector<float> pointsInPiesY[N_BINS];
  std::vector<float> pointsInPiesZ[N_BINS];
  for (uint i=0; i<pc0->points.size(); i++) {
    PointType Xi = pc0->points[i];
    int bin = computeBin(Xi);
    pointsInPiesX[bin].push_back(Xi.x);
    pointsInPiesY[bin].push_back(Xi.y);
    pointsInPiesZ[bin].push_back(Xi.z);
  }

  // Copy each of the coordinates in the bin into a giant bin of contiguous memory so cudaMemcpy will be happy
  // tmpX0 tmpY0 tmpZ0 is the the equivalent of X_0 Y_0 Z_0 in device memory
  float tmpX0[N0];
  float tmpY0[N0];
  float tmpZ0[N0];
  int index=0;
  for (int i=0; i<N_BINS; i++) {
    for (uint j=0; j<pointsInPiesX[i].size(); j++) {
      tmpX0[index]=pointsInPiesX[i][j];
      tmpY0[index]=pointsInPiesY[i][j];
      tmpZ0[index]=pointsInPiesZ[i][j];
      index++;
    }
  }

  // Calculate the starting index of each bin in the giant array from above
  int binIndices[N_BINS];
  binIndices[0]=0;
  for (int i=1; i<N_BINS; i++) {
    binIndices[i]=pointsInPiesX[i-1].size()+binIndices[i-1];
  }

  // Convert bins into cuda memory
  // X_0, Y_0, Z_0 points to a 1D array of coordinates
  // because cuda is just confusing wrt to struct stuff
  float *X_0 = cudaSetUp(N0, tmpX0);
  float *Y_0 = cudaSetUp(N0, tmpY0);
  float *Z_0 = cudaSetUp(N0, tmpZ0);

  endTime = CycleTimer::currentSeconds();
  std::cout << "Construction: " << endTime-startTime << std::endl;

  // Allocate CUDA memory for the result array. 
  // Pointers are stored here so that we don't have to do a cudamalloc each time we do a kernel call
  // Can just reuse these arrays
  int *indices = createIntArray(CHUNKSIZE);
  float *abc = createFloatArray(CHUNKSIZE);
  float *def = createFloatArray(CHUNKSIZE);
  float *ghi = createFloatArray(CHUNKSIZE);
  int *jkl = createIntArray(CHUNKSIZE);
  int *mno = createIntArray(CHUNKSIZE);
  int startIndices[CHUNKSIZE];
  int endIndices[CHUNKSIZE];
  float X_Q[CHUNKSIZE];
  float Y_Q[CHUNKSIZE];
  float Z_Q[CHUNKSIZE];
  
  int answers[N1];
  for (int i=0; i<N1; i+=CHUNKSIZE) {
    // Compute which bin the point belongs to
    // And the start and end indices of this bin in the data arrays
    startTime = CycleTimer::currentSeconds();
    for (int j=0; j<CHUNKSIZE; j++) {
      PointType X2 = pc1->points[j];
      int bin = computeBin(X2);
      startIndices[j] = binIndices[bin];
      if (bin==N_BINS-1) {
        endIndices[j] = N0;
      } else {
        endIndices[j] = binIndices[bin+1];
      }
      X_Q[j] = X2.x;
      Y_Q[j] = X2.y;
      Z_Q[j] = X2.z;
    }
    int cudaAns[CHUNKSIZE];
    cudaFindNearestNeighbor(N1, startIndices, endIndices, X_0, Y_0, Z_0, X_Q, Y_Q, Z_Q, indices, abc, def, ghi, jkl, mno, cudaAns);
    endTime = CycleTimer::currentSeconds();
    cudaTimes = cudaTimes + (endTime-startTime);
    for (int j=0; j<CHUNKSIZE; j++) {
      answers[i+j]=cudaAns[j];
    }
  }

  std::cout << "CUDA Time: " << cudaTimes << std::endl;
  std::cout << "Serial Time: " << serialTimes << std::endl;
  std::cout << "Speedup: " << serialTimes/cudaTimes << std::endl;
  return 0;
}