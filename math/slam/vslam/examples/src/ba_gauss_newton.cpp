#include <iostream>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Cholesky> 

#include <opencv2/opencv.hpp>

#include "sophus/se3.hpp"

void bundleAdjustmentGaussNewton(
        const std::vector<Eigen::Vector3d> &objectPts,
        const std::vector<Eigen::Vector2d> &imagePts,
        const cv::Mat &K,
        Sophus::SE3d &pose) 
{
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    
    const int iterations = 10;
    double cost = 0, lastCost = 0;
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    for (int iter = 0; iter < iterations; iter++) {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        // compute cost
        for (int i = 0; i < objectPts.size(); i++) {

            // pose is the [R|t]
            Eigen::Vector3d pc = pose * objectPts[i];
            double inv_z = 1.0 / pc[2];
            double inv_z2 = inv_z * inv_z;
            Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);
            Eigen::Vector2d e = imagePts[i] - proj;
            cost += e.squaredNorm();

            Eigen::Matrix<double, 2, 6> J;
            J << - fx * inv_z,
            0,
            fx * pc[0] * inv_z2,
            fx * pc[0] * pc[1] * inv_z2,
            - fx -  fx * pc[0] * pc[0] * inv_z2,
            fx * pc[1] * inv_z,
            0,
            - fy * inv_z,
            fy * pc[1] * inv_z,
            fy + fy * pc[1] * pc[1] * inv_z2,
            - fy * pc[0] * pc[1] * inv_z2,
            - fy * pc[0] * inv_z;

            H += J.transpose() * J;
            b += -J.transpose() * e;
        }

        Vector6d dx;

        auto cholesDecomp = H.ldlt();
        dx = cholesDecomp.solve(b);

        if (isnan(dx[0])) {
            std::cout << "result is nan!" << std::endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            // cost increase, update is not good
            std::cout << "Failed convergence for increased cost: cost: " << cost << ", last cost: " << lastCost << std::endl;
            break;
        }

        // update your estimation
        pose = Sophus::SE3d::exp(dx) * pose;
        lastCost = cost;

        std::cout << "iteration " << iter << " cost=" << std::cout.precision(12) << cost << std::endl;
        if (dx.norm() < 1e-6) {
            // converge
            break;
        }
    }

    std::cout << "pose: \n" << pose.matrix() << std::endl;

}

int main()
{

    // the object is a rectangle parallel to the camera frame at a distance of 5 to the camera optical center.
    // the camera sees the object projection on its frame with a shrinking scale factor 1/5

    // given the homography, there should be no rotation but translation of [-1,-1,0] from the world frame to the camera frame

    std::vector<Eigen::Vector3d> objectPts;
    objectPts.push_back(Eigen::Vector3d(1,1,5));
    objectPts.push_back(Eigen::Vector3d(11,1,5));
    objectPts.push_back(Eigen::Vector3d(1,11,5));
    objectPts.push_back(Eigen::Vector3d(11,11,5));

    std::vector<Eigen::Vector2d> imagePts;
    imagePts.push_back(Eigen::Vector2d(1/5,1/5));
    imagePts.push_back(Eigen::Vector2d(11/5,1/5));
    imagePts.push_back(Eigen::Vector2d(1/5,11/5));
    imagePts.push_back(Eigen::Vector2d(11/5,11/5));

    cv::Mat cameraMatrixK = cv::Mat::eye (3, 3, CV_64FC1);
    Sophus::SE3d pose;

    bundleAdjustmentGaussNewton(objectPts, imagePts, cameraMatrixK, pose);

    return 0;
}