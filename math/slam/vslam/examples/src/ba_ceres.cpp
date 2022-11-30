#ifndef SnavelyReprojection_H
#define SnavelyReprojection_H

#include <iostream>
#include <fstream>

#include "ceres/ceres.h"
#include <Eigen/Core>

#include "se3Tools.hpp"


class SnavelyReprojectionError
{
public:
    SnavelyReprojectionError(double observation_x, double observation_y):
    observed_x(observation_x),observed_y(observation_y){}

template<typename T>
    bool operator()(const T* const camera,
                const T* const point,
                T* residuals)const{                  
        // camera[0,1,2] are the angle-axis rotation
        T predictions[2];
        CamProjectionWithDistortion(camera, point, predictions);
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);

        return true;
    }

    static ceres::CostFunction* Create(const double observed_x, const double observed_y){
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError,2,9,3>(
            new SnavelyReprojectionError(observed_x,observed_y))
            );
    }


private:
    double observed_x;
    double observed_y;
};

#endif // SnavelyReprojection.h

int readData(const std::string& filename)
{
    FILE* fptr = fopen(filename.c_str(), "r");
    if (fptr == nullptr) {
        std::cout << "landmarks.txt not existed" << std::endl;
        return -1;
    }

    FscanfOrDie(fptr, "%d", &bal.num_cameras_);
    FscanfOrDie(fptr, "%d", &bal.num_points_);
    FscanfOrDie(fptr, "%d", &bal.num_observations_);

    std::cout << "Header: " << "NumCameras: " << bal.num_cameras_
            << " NumPoints: " << bal.num_points_
            << " NumObservations: " << bal.num_observations_
            << std::endl;

    bal.point_index_ = new int[bal.num_observations_];
    bal.camera_index_ = new int[bal.num_observations_];
    bal.observations_ = new double[2 * bal.num_observations_];

    bal.num_parameters_ = 9 * bal.num_cameras_ + 3 * bal.num_points_;
    bal.parameters_ = new double[bal.num_parameters_];

    // camera_idx, point_idx, obs_x, obs_y
    for (int i = 0; i < bal.num_observations_; ++i) {
        FscanfOrDie(fptr, "%d", bal.camera_index_ + i);
        FscanfOrDie(fptr, "%d", bal.point_index_ + i);
        for (int j = 0; j < 2; ++j) {
            FscanfOrDie(fptr, "%lf", bal.observations_ + 2*i + j);
        }
    }

    // params
    for (int i = 0; i < bal.num_parameters_; ++i) {
        FscanfOrDie(fptr, "%lf", bal.parameters_ + i);
    }

    fclose(fptr);

    return 0;
}

int main() {

    readData("./landmarks.txt");

    return 0;
}