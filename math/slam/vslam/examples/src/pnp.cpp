#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

int main() {

    // the object is a rectangle parallel to the camera frame at a distance of 5 to the camera optical center.
    // the camera sees the object projection on its frame with a shrinking scale factor 1/5

    // given the homography, there should be no rotation but translation of [-1,-1,0] from the world frame to the camera frame

    cv::Mat cameraMatrixK = cv::Mat::eye (3, 3, CV_64FC1);

    cv::Mat objectPoints = cv::Mat::zeros (4, 3, CV_64FC1);
    objectPoints.at<double>(0,0) = 1;
    objectPoints.at<double>(0,1) = 1;
    objectPoints.at<double>(0,2) = 5;
    objectPoints.at<double>(1,0) = 11;
    objectPoints.at<double>(1,1) = 1;
    objectPoints.at<double>(1,2) = 5;
    objectPoints.at<double>(2,0) = 1;
    objectPoints.at<double>(2,1) = 11;
    objectPoints.at<double>(2,2) = 5;
    objectPoints.at<double>(3,0) = 11;
    objectPoints.at<double>(3,1) = 11;
    objectPoints.at<double>(3,2) = 5;

    cv::Mat imagePoints = cv::Mat::zeros (4, 2, CV_64FC1);
    imagePoints.at<double>(0,0) = 1/5;
    imagePoints.at<double>(0,1) = 1/5;
    imagePoints.at<double>(1,0) = 11/5;
    imagePoints.at<double>(1,1) = 1/5;
    imagePoints.at<double>(2,0) = 1/5;
    imagePoints.at<double>(2,1) = 11/5;
    imagePoints.at<double>(3,0) = 11/5;
    imagePoints.at<double>(3,1) = 11/5;

    cv::Mat distCoeffs = cv::Mat::zeros(1,4,CV_64FC1);

    // here uses the default iterative PnP to solve the [R|t]
    cv::Mat rvec1, tvec1;
    cv::solvePnP(objectPoints, imagePoints, cameraMatrixK, distCoeffs, rvec1, tvec1, false);

    std::cout << "========PnP========" << std::endl;
    std::cout << "R: " << rvec1 << std::endl;
    std::cout << "t: " << tvec1 << std::endl;

    // here uses EPnP to solve the [R|t]
    cv::Mat rvec2, tvec2;
    cv::solvePnP(objectPoints, imagePoints, cameraMatrixK, distCoeffs, rvec2, tvec2, false, cv::SOLVEPNP_EPNP  );

    std::cout << "========EPNP========" << std::endl;
    std::cout << "R: " << rvec2 << std::endl;
    std::cout << "t: " << tvec2 << std::endl;

    // here uses P3P to solve the [R|t]
    cv::Mat rvec3, tvec3;
    cv::solvePnP(objectPoints, imagePoints, cameraMatrixK, distCoeffs, rvec3, tvec3, false, cv::SOLVEPNP_P3P );

    std::cout << "========P3P========" << std::endl;
    std::cout << "R: " << rvec3 << std::endl;
    std::cout << "t: " << tvec3 << std::endl;
}