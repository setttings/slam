#include <iostream>
#include <fstream>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "CycleTimer.h"

#define N_SCANS 64
#define MAX_ITERS 10
#define PI 3.14159265359

typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<PointType> PointCloudType;

pcl::visualization::PCLVisualizer viewer;

// Loads points described in file filename into point cloud pointed to by point_cloud
void load_point_cloud_file(int i, PointCloudType::Ptr point_cloud) {
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

void writeMapToFile(PointCloudType::Ptr point_cloud) {
  std::ofstream outfile;
  outfile.open("map.txt");
  for (int i = 0; i < point_cloud->size(); i++) {
    PointType pt = point_cloud->points[i];
    outfile << pt.x << " " << pt.y << " " << pt.z << " " << std::max(pt.z,-4.0f) << std::endl;
  }
  outfile.close();
}

void detect_edge_and_planar_points(const PointCloudType& cloud_in, PointCloudType& edge_points, PointCloudType& planar_points) {
  edge_points.clear();
  planar_points.clear();

  int cloudSize = cloud_in.size();

  std::vector<int> scanStartInd(N_SCANS, 0);
  std::vector<int> scanEndInd(N_SCANS, 0);
  std::vector<std::pair<float, int> > cloudCurvature(cloudSize);
  std::vector<float> disqualified(cloudSize, 0);
  scanStartInd[0] = 5;
  scanEndInd[N_SCANS - 1] = cloudSize - 5;

  int scanCount = 0;
  float azimuth, azimuth_prev = 1;
  // Compute curvature
  for (int i = 5; i < cloudSize - 5; i++) {
    float diffX = cloud_in[i - 5].x + cloud_in[i - 4].x 
                + cloud_in[i - 3].x + cloud_in[i - 2].x 
                + cloud_in[i - 1].x - 10 * cloud_in[i].x 
                + cloud_in[i + 1].x + cloud_in[i + 2].x
                + cloud_in[i + 3].x + cloud_in[i + 4].x
                + cloud_in[i + 5].x;
    float diffY = cloud_in[i - 5].y + cloud_in[i - 4].y 
                + cloud_in[i - 3].y + cloud_in[i - 2].y 
                + cloud_in[i - 1].y - 10 * cloud_in[i].y 
                + cloud_in[i + 1].y + cloud_in[i + 2].y
                + cloud_in[i + 3].y + cloud_in[i + 4].y
                + cloud_in[i + 5].y;
    float diffZ = cloud_in[i - 5].z + cloud_in[i - 4].z 
                + cloud_in[i - 3].z + cloud_in[i - 2].z 
                + cloud_in[i - 1].z - 10 * cloud_in[i].z 
                + cloud_in[i + 1].z + cloud_in[i + 2].z
                + cloud_in[i + 3].z + cloud_in[i + 4].z
                + cloud_in[i + 5].z;
    cloudCurvature[i] = std::make_pair(diffX * diffX + diffY * diffY + diffZ * diffZ, i);

    azimuth = atan2(cloud_in[i].y, cloud_in[i].x);
    if (azimuth_prev < 0 && azimuth > 0) {
      scanCount++;
      scanStartInd[scanCount] = i + 5;
      scanEndInd[scanCount - 1] = i - 5;
    }
    azimuth_prev = azimuth;

  }
  assert(scanCount == 63);
  
  /*  
  // Print scanStartInd and scanEndInd
  for (int i = 0; i < 64; i++) {
    std::cout << scanStartInd[i] << " " << scanEndInd[i] << std::endl;;
  }
  */

  /*
  // Disqualify points
  for (int i = 5; i < cloudSize - 6; i++) {
    float diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x;
    float diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y;
    float diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z;
    float diff = diffX * diffX + diffY * diffY + diffZ * diffZ;

    if (diff > 0.1) {

      float depth1 = sqrt(laserCloud->points[i].x * laserCloud->points[i].x + 
                     laserCloud->points[i].y * laserCloud->points[i].y +
                     laserCloud->points[i].z * laserCloud->points[i].z);

      float depth2 = sqrt(laserCloud->points[i + 1].x * laserCloud->points[i + 1].x + 
                     laserCloud->points[i + 1].y * laserCloud->points[i + 1].y +
                     laserCloud->points[i + 1].z * laserCloud->points[i + 1].z);

      if (depth1 > depth2) {
        diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x * depth2 / depth1;
        diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y * depth2 / depth1;
        diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z * depth2 / depth1;

        if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth2 < 0.1) {
          disqualified[i - 5] = 1;
          disqualified[i - 4] = 1;
          disqualified[i - 3] = 1;
          disqualified[i - 2] = 1;
          disqualified[i - 1] = 1;
          disqualified[i] = 1;
        }
      } else {
        diffX = laserCloud->points[i + 1].x * depth1 / depth2 - laserCloud->points[i].x;
        diffY = laserCloud->points[i + 1].y * depth1 / depth2 - laserCloud->points[i].y;
        diffZ = laserCloud->points[i + 1].z * depth1 / depth2 - laserCloud->points[i].z;

        if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth1 < 0.1) {
          disqualified[i + 1] = 1;
          disqualified[i + 2] = 1;
          disqualified[i + 3] = 1;
          disqualified[i + 4] = 1;
          disqualified[i + 5] = 1;
          disqualified[i + 6] = 1;
        }
      }
    }

    float diffX2 = laserCloud->points[i].x - laserCloud->points[i - 1].x;
    float diffY2 = laserCloud->points[i].y - laserCloud->points[i - 1].y;
    float diffZ2 = laserCloud->points[i].z - laserCloud->points[i - 1].z;
    float diff2 = diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2;

    float dis = laserCloud->points[i].x * laserCloud->points[i].x
              + laserCloud->points[i].y * laserCloud->points[i].y
              + laserCloud->points[i].z * laserCloud->points[i].z;

    if (diff > 0.0002 * dis && diff2 > 0.0002 * dis) {
      disqualified[i] = 1;
    }
  }
  */

  // Select points
  //pcl::PointCloud<PointType> cornerPointsLessSharp;
  //pcl::PointCloud<PointType> surfPointsLessFlat;

  pcl::VoxelGrid<PointType> downSizeFilter;
  downSizeFilter.setInputCloud(cloud_in.makeShared());
  downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
  downSizeFilter.filter(planar_points);

  for (int i = 0; i < N_SCANS; i++) {
    for (int j = 0; j < 6; j++) {
      int sp = (scanStartInd[i] * (6 - j)  + scanEndInd[i] * j) / 6;
      int ep = (scanStartInd[i] * (5 - j)  + scanEndInd[i] * (j + 1)) / 6 - 1;

      // Find the 20 points within sp to ep with the largest curvature (edge points)
      std::sort(cloudCurvature.begin() + sp, cloudCurvature.begin() + ep + 1);
      for (int k = 0; k < 20; k++) {
        if (cloudCurvature[ep-k].first > 0.1) {
          PointType pt = cloud_in[cloudCurvature[ep-k].second];
          edge_points.push_back(pt);
        }
      }

      /*
      int largestPickedNum = 0;
      for (int k = ep; k >= sp; k--) {
        int ind = cloudSortInd[k];
        if (disqualified[ind] == 0 &&
            cloudCurvature[ind] > 0.1) {
        
          largestPickedNum++;
          if (largestPickedNum <= 2) {
            cloudLabel[ind] = 2;
            cornerPointsSharp.push_back(cloud_in[ind]);
            cornerPointsLessSharp.push_back(cloud_in[ind]);
          } else if (largestPickedNum <= 20) {
            cloudLabel[ind] = 1;
            cornerPointsLessSharp.push_back(cloud_in[ind]);
          } else {
            break;
          }

          disqualified[ind] = 1;
          for (int l = 1; l <= 5; l++) {
            float diffX = cloud_in[ind + l].x 
                        - cloud_in[ind + l - 1].x;
            float diffY = cloud_in[ind + l].y 
                        - cloud_in[ind + l - 1].y;
            float diffZ = cloud_in[ind + l].z 
                        - cloud_in[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            disqualified[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            float diffX = cloud_in[ind + l].x 
                        - cloud_in[ind + l + 1].x;
            float diffY = cloud_in[ind + l].y 
                        - cloud_in[ind + l + 1].y;
            float diffZ = cloud_in[ind + l].z 
                        - cloud_in[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            disqualified[ind + l] = 1;
          }
        }
      }

      int smallestPickedNum = 0;
      for (int k = sp; k <= ep; k++) {
        int ind = cloudSortInd[k];
        if (disqualified[ind] == 0 &&
            cloudCurvature[ind] < 0.1) {

          cloudLabel[ind] = -1;
          surfPointsFlat.push_back(cloud_in[ind]);

          smallestPickedNum++;
          if (smallestPickedNum >= 4) {
            break;
          }

          disqualified[ind] = 1;
          for (int l = 1; l <= 5; l++) {
            float diffX = cloud_in[ind + l].x 
                        - cloud_in[ind + l - 1].x;
            float diffY = cloud_in[ind + l].y 
                        - cloud_in[ind + l - 1].y;
            float diffZ = cloud_in[ind + l].z 
                        - cloud_in[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            disqualified[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            float diffX = cloud_in[ind + l].x 
                        - cloud_in[ind + l + 1].x;
            float diffY = cloud_in[ind + l].y 
                        - cloud_in[ind + l + 1].y;
            float diffZ = cloud_in[ind + l].z 
                        - cloud_in[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            disqualified[ind + l] = 1;
          }
        }
      }

      for (int k = sp; k <= ep; k++) {
        if (cloudLabel[k] <= 0) {
          surfPointsLessFlatScan->push_back(cloud_in[k]);
        }
      }
      */
    }

    /*
    pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
    pcl::VoxelGrid<PointType> downSizeFilter;
    pcl::VoxelGrid<pcl::PCLPointCloud2> downSizeFilter;
    downSizeFilter.setInputCloud(surfPointsLessFlatScan);
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.filter(surfPointsLessFlatScanDS);
    surfPointsLessFlat += surfPointsLessFlatScanDS;
    */
  }

  std::cout << "# less flat points = " << planar_points.size() << std::endl;
  std::cout << "# less sharp points = " << edge_points.size() << std::endl;
}

void visualizeMatch(PointCloudType::ConstPtr target, PointCloudType::ConstPtr src) {
  viewer.removeAllPointClouds();
  std::cout << "Visualizing match between " << src->size() << " points and " << target->size() << " points" << std::endl;

  // Make target red, src blue
  pcl::PointCloud<pcl::PointXYZRGB> target_red;
  pcl::PointCloud<pcl::PointXYZRGB> src_blue;
  pcl::copyPointCloud(*target, target_red);
  pcl::copyPointCloud(*src, src_blue);
  for (int i = 0; i < target_red.size(); i++) {
    target_red[i].r = 255;
  }
  for (int i = 0; i < src_blue.size(); i++) {
      src_blue[i].b = 255;
  }
  viewer.addPointCloud(target_red.makeShared(), "target");
  viewer.addPointCloud(src_blue.makeShared(), "source");

  viewer.spinOnce(1); // TODO: What is the correct number?
}

void detect_planar_points(PointCloudType::Ptr cloud_in, PointCloudType::Ptr planar_points) {
  planar_points->clear();

  pcl::VoxelGrid<PointType> downSizeFilter;
  downSizeFilter.setInputCloud(cloud_in);
  downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
  downSizeFilter.filter(*planar_points);
}

// po = R * pi + T
// where R = Ry * Rx * Rz
void transformPointToMap(float transform[6], const PointType& pi, PointType& po) {
  // TODO: Use Eigen::AngleAxisd and Eigen::Quaternion
  // Refer to http://stackoverflow.com/questions/21412169/creating-a-rotation-matrix-with-pitch-yaw-roll-using-eigen
  // Tutorial: https://eigen.tuxfamily.org/dox/group__TutorialGeometry.html
  // X1 = Rz * pi
  float x1 = cos(transform[2]) * pi.x - sin(transform[2]) * pi.y;
  float y1 = sin(transform[2]) * pi.x + cos(transform[2]) * pi.y;
  float z1 = pi.z;

  // X2 = Rx * X1
  float x2 = x1;
  float y2 = cos(transform[0]) * y1 - sin(transform[0]) * z1;
  float z2 = sin(transform[0]) * y1 + cos(transform[0]) * z1;

  // po = Ry * X2 + T
  po.x = cos(transform[1]) * x2 + sin(transform[1]) * z2 + transform[3];
  po.y = y2 + transform[4];
  po.z = -sin(transform[1]) * x2 + cos(transform[1]) * z2 + transform[5];
}

void* render_loop(void* arg) {
  // TODO: Run this on another thread
  //std::cout << "Spinning..." << std::endl;
  //viewer.spinOnce(1); // TODO: What is the correct number?
  //std::cout << "Done spinning..." << std::endl;
  while (!viewer.wasStopped()) {
    viewer.spinOnce(1);
    //boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }

  return NULL;
}

// transform = transform * transformPrev^-1 * transform
// transformPrev = transform
void estimateNextTransform(float transformPrev[6], float transform[6]) {
  float rx = transform[0];
  float ry = transform[1];
  float rz = transform[2];
  float tx = transform[3];
  float ty = transform[4];
  float tz = transform[5];

  // Stuff
  Eigen::Matrix4f Rx;
  Rx << 1,       0,        0, 0,
        0, cos(rx), -sin(rx), 0,
        0, sin(rx),  cos(rx), 0,
        0,       0,        0, 1;
  Eigen::Matrix4f Ry;
  Ry <<  cos(ry), 0, sin(ry), 0,
               0, 1,       0, 0,
        -sin(ry), 0, cos(ry), 0,
               0, 0,       0, 1;
  Eigen::Matrix4f Rz;
  Rz << cos(rz), -sin(rz), 0, 0,
        sin(rz),  cos(rz), 0, 0,
              0,        0, 1, 0,
              0,        0, 0, 1;
  Eigen::Matrix4f T;
  T << 1, 0, 0, tx,
       0, 1, 0, ty,
       0, 0, 1, tz,
       0, 0, 0, 1;
  Eigen::Matrix4f H = T*Ry*Rx*Rz;

  float rx_ = transformPrev[0];
  float ry_ = transformPrev[1];
  float rz_ = transformPrev[2];
  float tx_ = transformPrev[3];
  float ty_ = transformPrev[4];
  float tz_ = transformPrev[5];

  // Stuff
  Eigen::Matrix4f Rx_;
  Rx_ << 1,        0,         0, 0,
         0, cos(rx_), -sin(rx_), 0,
         0, sin(rx_),  cos(rx_), 0,
         0,        0,         0, 1;
  Eigen::Matrix4f Ry_;
  Ry_ << cos(ry_), 0, sin(ry_), 0,
                0, 1,        0, 0,
        -sin(ry_), 0, cos(ry_), 0,
                0, 0,        0, 1;
  Eigen::Matrix4f Rz_;
  Rz_ << cos(rz_), -sin(rz_), 0, 0,
         sin(rz_),  cos(rz_), 0, 0,
                0,         0, 1, 0,
                0,         0, 0, 1;
  Eigen::Matrix4f T_;
  T_ << 1, 0, 0, tx_,
        0, 1, 0, ty_,
        0, 0, 1, tz_,
        0, 0, 0, 1;
  Eigen::Matrix4f H_ = T_*Ry_*Rx_*Rz_;

  Eigen::Matrix4f H_new = H * H_.inverse() * H;

  // Extract rx,ry,rz,tx,ty,tz from H_new;
  // TODO: Extract rx, ry, rz
  transform[0] = asin(-H_new(1,2));
  float cx = cos(transform[0]);
  assert(cx != 0);
  transform[1] = atan2(H_new(0,2) / cx,
                       H_new(2,2) / cx);
  transform[2] = atan2(H_new(1,0) / cx,
                       H_new(1,1) / cx);
  transform[3] = H_new(0,3);
  transform[4] = H_new(1,3);
  transform[5] = H_new(2,3);

  // Copy original transform to transformPrev
  transformPrev[0] = rx;
  transformPrev[1] = ry;
  transformPrev[2] = rz;
  transformPrev[3] = tx;
  transformPrev[4] = ty;
  transformPrev[5] = tz;
}

int main (int argc, char** argv)
{
  pthread_t tid;
  //pthread_create(&tid, NULL, render_loop, NULL);

  float transform[6] = {0};
  float transformPrev[6] = {0};
  assert(transform[0] == 0 &&
         transform[1] == 0 &&
         transform[2] == 0 &&
         transform[3] == 0 &&
         transform[4] == 0 &&
         transform[5] == 0);

  // Add 0th frame to map point cloud
  PointCloudType::Ptr map(new PointCloudType);
  load_point_cloud_file(0, map);

  pcl::VoxelGrid<PointType> downSizeFilter;
  downSizeFilter.setInputCloud(map);
  downSizeFilter.setLeafSize(0.4, 0.4, 0.4);
  downSizeFilter.filter(*map);

  pcl::KdTreeFLANN<PointType>::Ptr kdtree(new pcl::KdTreeFLANN<PointType>());

  std::vector<int> pointSearchInd(5);
  std::vector<float> pointSearchSqDis(5);

  PointType pointOri, pointSel;
  pcl::PointXYZI coeff;

  Eigen::MatrixXf matA0(5,3);
  Eigen::VectorXf matB0(5);
  matB0 << -1, -1, -1, -1, -1;
  Eigen::Vector3f matX0;

  //PointCloudType edge_points_prev, planar_points_prev;
  //detect_edge_and_planar_points(map, edge_points_prev, planar_points_prev);

  //viewer.addPointCloud(planar_points.makeShared());

  // Default to nFrames = 111 if not specfified on command line
  int nFrames = 111; 
  if (argc != 2) std::cout << "Usage: ./loam <number_of_frames> " << std::endl; 
  else nFrames = atoi(argv[1]);
  std::cout << "nFrames = " << nFrames << std::endl;

  PointCloudType::Ptr new_frame(new PointCloudType);
  PointCloudType::Ptr planar_points(new PointCloudType);
  PointCloudType::Ptr transformed_planar_points(new PointCloudType);
  PointCloudType::Ptr laserCloudOri(new PointCloudType);
  pcl::PointCloud<pcl::PointXYZI>::Ptr coeffSel(new pcl::PointCloud<pcl::PointXYZI>);

  for (int i = 1; i < nFrames; i++) {
    // Begin CHUNK 1
    double startTime = CycleTimer::currentSeconds();

    new_frame->clear();
    planar_points->clear();

    // Load point cloud
    load_point_cloud_file(i, new_frame);

    // Detect edge and planar points
    //detect_edge_and_planar_points(new_frame, edge_points, planar_points);
    detect_planar_points(new_frame, planar_points);

    // Visualize previous frame and this frame's planar points together
    //visualizeMatch(map, planar_points);

    kdtree->setInputCloud(map);

    // Estimate transform for this frame
    estimateNextTransform(transformPrev, transform);

    double endTime = CycleTimer::currentSeconds();
    std::cout << "Chunk 1: " << endTime-startTime << std::endl;
    // END CHUNK 1
    
    // BEGIN CHUNK 2
    startTime = CycleTimer::currentSeconds();
    
    // Match planar_points to map
    for (int iter = 0; iter < MAX_ITERS; iter++) {
      laserCloudOri->clear();
      coeffSel->clear();

      int nPointsNotSelected = 0;
      int nPlanarPoints = planar_points->size();

      for (int i = 0; i < nPlanarPoints; i++) {
        pointOri = planar_points->points[i]; // Make sure new_frame has been transformed
        // Transform point to map as transformedPoint
        transformPointToMap(transform, pointOri, pointSel);
        // Find closest points to point
        kdtree->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
        // Fill row of matrix and error vector according to math stuff with closest points
        if (pointSearchSqDis[4] < 1.0) {
          for (int j = 0; j < 5; j++) {
            matA0(j,0) = map->points[pointSearchInd[j]].x;
            matA0(j,1) = map->points[pointSearchInd[j]].y;
            matA0(j,2) = map->points[pointSearchInd[j]].z;
          }
          matX0 = matA0.colPivHouseholderQr().solve(matB0);

          float pa = matX0(0, 0);
          float pb = matX0(1, 0);
          float pc = matX0(2, 0);
          float pd = 1;
          assert(!isnan(pa));
          assert(!isnan(pb));
          assert(!isnan(pc));
 
          float ps = sqrt(pa * pa + pb * pb + pc * pc);
          pa /= ps;
          pb /= ps;
          pc /= ps;
          pd /= ps;

          bool planeValid = true;

          if (planeValid) {
            float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;
            /*
            if (pd2 == 0 || isnan(pd2)) {
              std::cout << "pa = " << pa << ", pb = " << pb << std::endl;
            }
            */

            float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                          + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

            coeff.x = s * pa;
            coeff.y = s * pb;
            coeff.z = s * pc;
            coeff.intensity = s * pd2;

            if (s > 0.1) {
              laserCloudOri->push_back(pointOri);
              coeffSel->push_back(coeff);
            }/* else {
              if (nPointsNotSelected == 0) {
                std::cout << "s = " << s << ", pd2 = " << pd2 << std::endl;
              }
              nPointsNotSelected++;
            }*/
          }
        }
      }

      int n = laserCloudOri->size();
      //std::cout << "# points selected = " << laserCloudOri->size() << std::endl;
      //std::cout << "# points not selected because s was too small = " << nPointsNotSelected << std::endl;
      assert(laserCloudOri->size() > 50);

      Eigen::MatrixXf matA(n, 6);
      Eigen::VectorXf matB(n);
      Eigen::VectorXf matX(6);

      float srx = sin(transform[0]);
      float crx = cos(transform[0]);
      float sry = sin(transform[1]);
      float cry = cos(transform[1]);
      float srz = sin(transform[2]);
      float crz = cos(transform[2]);

      // CAN JUST PUT A PARALLEL FOR HERE
      for (int i = 0; i < n; i++) {
        pointOri = laserCloudOri->points[i];
        coeff = coeffSel->points[i];

        float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                        + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                        + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

        float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                        + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                        + ((-cry*crz - srx*sry*srz)*pointOri.x 
                        + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

        float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                        + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                        + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;

        matA(i, 0) = arx;
        matA(i, 1) = ary;
        matA(i, 2) = arz;
        matA(i, 3) = coeff.x;
        matA(i, 4) = coeff.y;
        matA(i, 5) = coeff.z;
        matB(i, 0) = -coeff.intensity;
      }

      std::cout << "Iteration " << iter << ": Average error = " << matB.norm() / n << std::endl;

      matX = (matA.transpose() * matA).colPivHouseholderQr().solve(matA.transpose() * matB); // TODO: Considering storing matA0.transpose() instead of calling it twice

      transform[0] += matX(0);
      transform[1] += matX(1);
      transform[2] += matX(2);
      transform[3] += matX(3);
      transform[4] += matX(4);
      transform[5] += matX(5);

      /*
      // Visualize iteration
      if (iter % 1 == 0) {
        // Visualize transformed planar points
        transformed_planar_points->resize(planar_points->size());
        for (int ptIdx = 0; ptIdx < planar_points->size(); ptIdx++) {
          transformPointToMap(transform, planar_points->points[ptIdx], transformed_planar_points->points[ptIdx]);
        }
        visualizeMatch(map, transformed_planar_points);
      }
      */

      float deltaR = sqrt(
        pow((matX(0)/PI*180), 2) +
        pow((matX(1)/PI*180), 2) +
        pow((matX(2)/PI*180), 2));
      float deltaT = sqrt(
        pow(matX(3) * 100, 2) +
        pow(matX(4) * 100, 2) +
        pow(matX(5) * 100, 2));

      bool converged = deltaR < 0.05 && deltaT < 0.05;
      if (converged) {
        break;
      }

    }

    endTime = CycleTimer::currentSeconds();
    std::cout << "Chunk 2: " << endTime-startTime << std::endl;
    // END CHUNK 2
    
    // BEGIN CHUNK 3
    startTime = CycleTimer::currentSeconds();

    // Only add points to map every 10 frames
    if (i % 10 == 0) {
      // Add transformed planar_points to map, downsize with downSizeFilter
      for (int ptIdx = 0; ptIdx < planar_points->size(); ptIdx++) {
        transformPointToMap(transform, planar_points->points[ptIdx], pointSel);
        map->push_back(pointSel);
      }
      downSizeFilter.setInputCloud(map);
      downSizeFilter.filter(*map);
    }

    endTime = CycleTimer::currentSeconds();
    std::cout << "Chunk 3: " << endTime-startTime << std::endl;
    // END CHUNK 3

  }

  // Save map to file
  writeMapToFile(map);

  return 0;

}
