#include <iostream>
#include <fstream>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "CycleTimer.h"

typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<PointType> PointCloudType;

// Loads points described in file filename into point cloud pointed to by point_cloud
void load_point_cloud_file(int i, PointCloudType::Ptr point_cloud) {
  point_cloud->clear();
  char buff[40];
  snprintf(buff, sizeof(buff), "../../point_clouds/00000%05d.txt", i);

  std::string buffStr = buff;
  std::ifstream infile(buffStr.c_str());
  std::cout << "File name: " << buffStr << std::endl;

  float x,y,z,a;

  while (infile >> x >> y >> z >> a) {
    //std::cout << x << " " << y << " " <<  z << std::endl;
    point_cloud->push_back(pcl::PointXYZ(x,y,z));
  }

  std::cout << "Point cloud has " << point_cloud->size() << " points" << std::endl;
}

void writeAnswers(int* answers, int N) {
  // Set up output file
  std::ofstream outfile;
  outfile.open("answers.txt");
  for (int i = 0; i < N; i++) {
    // Output answers[i]
    outfile << answers[i] << " ";
  }
  outfile.close();
}

// Assumes answers[] is big enough to read file
void readAnswers(int *answers) {
  std::ifstream infile("answers.txt");
  float answer;
  int i = 0;
  while (infile >> answer) {
    answers[i] = answer;
    i++;
  }
  // Close file
  infile.close();
}

int main() {
  PointCloudType::Ptr pc0(new PointCloudType);
  PointCloudType::Ptr pc1(new PointCloudType);
  load_point_cloud_file(0, pc0);
  load_point_cloud_file(1, pc1);

  int k = 1;
  std::vector<int> indices(k);
  std::vector<float> distances(k);

  // Trial 1: Query points come from pc1
  // Begin timing
  // Build kdtree out of pc0
  double start = CycleTimer::currentSeconds();
  pcl::KdTreeFLANN<PointType>::Ptr kdtree(new pcl::KdTreeFLANN<PointType>());
  kdtree->setInputCloud(pc0);

  double end2 = CycleTimer::currentSeconds();

  int nQueryPoints = pc1->size();
  int answers[nQueryPoints];
  readAnswers(answers);

  for (int i = 0; i < nQueryPoints; i++) {
    kdtree->nearestKSearch(pc1->points[i], k, indices, distances);
    assert(answers[i] == indices[0]);
    //answers[i] = indices[0];
  }
  double end = CycleTimer::currentSeconds();
  // End timing
  printf("Time taken for %d queries from another point cloud = %fs\n", nQueryPoints, end-start);
  printf("Construction time = %f, queries = %f\n\n", end2-start, end-end2);

  //writeAnswers(answers, nQueryPoints);

}
