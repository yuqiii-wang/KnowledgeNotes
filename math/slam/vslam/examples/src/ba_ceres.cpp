#ifndef SnavelyReprojection_H
#define SnavelyReprojection_H

#include <iostream>
#include <fstream>

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

int main() {

    BAL bal;

    readData("./landmarks.txt", bal);

    bal.normalize();
    bal.perturb(0.0, 0.0, 0.0);

    const int point_block_size = 3;
    const int camera_block_size = 9;
    double* points = bal.mutable_points();
    double* cameras = bal.mutable_cameras();

    writeToPLYFile("inits.ply", bal);

    // Observations is 2 * num_observations long array observations
    // [u_1, u_2, ... u_n], where each u_i is two dimensional, the x 
    // and y position of the observation. 
    // const std::shared_ptr<double> observations = 
                // std::make_shared<double>(bal.getObservations());
    const double* observations = bal.getObservations();
    ceres::Problem problem;

    for(int i = 0; i < bal.getNumObservations(); ++i){

        // Each Residual block takes a point and a camera as input 
        // and outputs a 2 dimensional Residual
      
        ceres::CostFunction* cost_function = SnavelyReprojectionError::Create((&(*observations))[2*i + 0], (&(*observations))[2*i + 1]);

        // If enabled use Huber's loss function. 
        ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

        // Each observation corresponds to a pair of a camera and a point 
        // which are identified by camera_index()[i] and point_index()[i]
        // respectively.
        double* camera = cameras + camera_block_size * bal.getCameraIndex()[i];
        double* point = points + point_block_size * bal.getPointIndex()[i];

        problem.AddResidualBlock(cost_function, loss_function, camera, point);
    }


    ceres::Solver::Options options;
    options.max_num_iterations = 20;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 12;
    ceres::StringToLinearSolverType("dense_schur", &options.linear_solver_type);
    ceres::StringToTrustRegionStrategyType("dogleg",
                                        &options.trust_region_strategy_type);
    ceres::StringToSparseLinearAlgebraLibraryType("suite_sparse", &options.sparse_linear_algebra_library_type);
    ceres::StringToDenseLinearAlgebraLibraryType("eigen", &options.dense_linear_algebra_library_type);
    options.gradient_tolerance = 1e-8;
    options.function_tolerance = 1e-8;


    ceres::ParameterBlockOrdering* ordering = new ceres::ParameterBlockOrdering;
    // The points come before the cameras
    for(int i = 0; i < bal.getNumPoints(); ++i)
    ordering->AddElementToGroup(points + point_block_size * i, 0);
    for(int i = 0; i < bal.getNumCameras(); ++i)
        ordering->AddElementToGroup(cameras + camera_block_size * i, 1);
    // set this ordering to options
    options.linear_solver_ordering.reset(ordering);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    writeToPLYFile("results.ply", bal);

    return 0;
}