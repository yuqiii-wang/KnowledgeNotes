#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>

#include <iostream>

int main(){

  auto imageL = cv::imread("street_view_left.png");
  auto imageR = cv::imread("street_view_right.png");

  // Camera intristic parameter matrix
  // I did not calibration
  cv::Mat K = (cv::Mat_<float>(3,3) <<  500.f,   0.f, imageL.cols / 2.f,
                                          0.f, 500.f, imageL.rows / 2.f,
                                          0.f,   0.f,               1.f);

  // Variables to store keypoints and descriptors
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;

  cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();

  // extract feature points and calculate descriptors
  akaze -> detectAndCompute(imageL, cv::noArray(), keypoints1, descriptors1);
  akaze -> detectAndCompute(imageR, cv::noArray(), keypoints2, descriptors2);

  // Match features.
  std::vector<cv::DMatch> matches;
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
  matcher->match(descriptors1, descriptors2, matches, cv::Mat());

  // Sort matches by score
  std::sort(matches.begin(), matches.end());

  // Remove not so good matches
  const float GOOD_MATCH_PERCENT = 0.15f;
  const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
  matches.erase(matches.begin()+numGoodMatches, matches.end());

  // Draw top matches
  cv::Mat imMatches;
  cv::drawMatches(imageL, keypoints1, imageR, keypoints2, matches, imMatches);
  cv::imshow("picture of matching by AKAZE", imMatches);

  // Extract location of good matches
  std::vector<cv::Point2f> points1, points2;

  for( size_t i = 0; i < matches.size(); i++ )
  {
    points1.push_back( keypoints1[ matches[i].queryIdx ].pt );
    points2.push_back( keypoints2[ matches[i].trainIdx ].pt );
  }

  cv::Mat Kd;
  K.convertTo(Kd, CV_64F);

  cv::Mat mask; // unsigned char array
  cv::Mat E = cv::findEssentialMat(points1, points2, Kd.at<double>(0,0),
                           cv::Point2d(imageL.cols/2., imageL.rows/2.),
                           cv::RANSAC, 0.999, 1.0, mask);
  // E is CV_64F not 32F

  std::vector<cv::Point2f> inlier_match_points1, inlier_match_points2;
  for(int i = 0; i < mask.rows; i++) {
    if(mask.at<unsigned char>(i)){
      inlier_match_points1.push_back(points1[i]);
      inlier_match_points2.push_back(points2[i]);
    }
  }

  cv::Mat src;
  cv::hconcat(imageL, imageR, src);
  for(int i = 0; i < inlier_match_points1.size(); i++) {
    cv::line( src, inlier_match_points1[i],
              cv::Point2f(inlier_match_points2[i].x + imageL.cols, inlier_match_points2[i].y),
              1, 1, 0 );
  }

  mask.release();
  cv::Mat R, t;
  cv::recoverPose(E, 
                  inlier_match_points1,
                  inlier_match_points2, 
                  R, t, Kd.at<double>(0,0), 
                  cv::Point2d(imageL.cols/2., imageL.rows/2.),
                  mask);
  // R,t is CV_64F not 32F

  std::vector<cv::Point2d> triangulation_points1, triangulation_points2;
  for(int i = 0; i < mask.rows; i++) {
    if(mask.at<unsigned char>(i)){
      triangulation_points1.push_back 
                   (cv::Point2d((double)inlier_match_points1[i].x,(double)inlier_match_points1[i].y));
      triangulation_points2.push_back 
                   (cv::Point2d((double)inlier_match_points2[i].x,(double)inlier_match_points2[i].y));
    }
  }

  // if(true) {
  cv::Mat srcTrian;
  cv::hconcat(imageL, imageR, srcTrian);
  for(int i = 0; i < triangulation_points1.size(); i++) {
    cv::line( srcTrian, triangulation_points1[i],
              cv::Point2f((float)triangulation_points2[i].x + (float)imageL.cols,
                          (float)triangulation_points2[i].y),
              1, 1, 0 );
  }

  cv::Mat Rt0 = cv::Mat::eye(3, 4, CV_64FC1);
  cv::Mat Rt1 = cv::Mat::eye(3, 4, CV_64FC1);
  R.copyTo(Rt1.rowRange(0,3).colRange(0,3));
  t.copyTo(Rt1.rowRange(0,3).col(3));


  cv::Mat point3d_homo;
  cv::triangulatePoints(Kd * Rt0, Kd * Rt1, 
                        triangulation_points1, triangulation_points2,
                        point3d_homo);
  //point3d_homo is 64F
  //available input type is here
  //https://stackoverflow.com/questions/16295551/how-to-correctly-use-cvtriangulatepoints

  assert(point3d_homo.cols == triangulation_points1.size());

  // prepare a viewer
	pcl::visualization::PCLVisualizer viewer("Viewer");
  viewer.setBackgroundColor (255, 255, 255);

  // create point cloud
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	cloud->points.resize (point3d_homo.cols);

  for(int i = 0; i < point3d_homo.cols; i++) {
    pcl::PointXYZRGB &point = cloud->points[i];
    cv::Mat p3d;
    cv::Mat _p3h = point3d_homo.col(i);
    convertPointsFromHomogeneous(_p3h.t(), p3d);
    point.x = p3d.at<double>(0);
    point.y = p3d.at<double>(1);
    point.z = p3d.at<double>(2);
    point.r = 0;
    point.g = 0;
    point.b = 255;
  }

  viewer.addPointCloud(cloud, "Triangulated Point Cloud");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                            3,
                                            "Triangulated Point Cloud");
  viewer.addCoordinateSystem (1.0);

  // add the second camera pose 
  Eigen::Matrix4f eig_mat;
  Eigen::Affine3f cam_pose;

  R.convertTo(R, CV_32F);
  t.convertTo(t, CV_32F);

  //this shows how a camera moves
  cv::Mat Rinv = R.t(); 
  cv::Mat T = -Rinv * t;

  eig_mat(0,0) = Rinv.at<float>(0,0);eig_mat(0,1) = Rinv.at<float>(0,1);eig_mat(0,2) = Rinv.at<float>(0,2);
  eig_mat(1,0) = Rinv.at<float>(1,0);eig_mat(1,1) = Rinv.at<float>(1,1);eig_mat(1,2) = Rinv.at<float>(1,2);
  eig_mat(2,0) = Rinv.at<float>(2,0);eig_mat(2,1) = Rinv.at<float>(2,1);eig_mat(2,2) = Rinv.at<float>(2,2);
  eig_mat(3,0) = 0.f; eig_mat(3,1) = 0.f; eig_mat(3,2) = 0.f;
  eig_mat(0, 3) = T.at<float>(0);
  eig_mat(1, 3) = T.at<float>(1);
  eig_mat(2, 3) = T.at<float>(2);
  eig_mat(3, 3) = 1.f;

  cam_pose = eig_mat;

  //cam_pose should be Affine3f, Affine3d cannot be used
  viewer.addCoordinateSystem(1.0, cam_pose, "2nd cam");

  viewer.initCameraParameters ();
  while (!viewer.wasStopped ()) {
    viewer.spin();
  }

    cv::waitKey(0);

    return 0;
}