#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/se3.hpp"

int main()
{
    // Rotation matrix with 90 degrees along Z axis (defined as (0,0,1))
    std::cout << "\n********SO(3) - 90********\n" << std::endl;

    // To rotation matrix
    Eigen::Matrix3d R = Eigen::AngleAxisd(
        M_PI / 2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();

    // To quaternion
    Eigen::Quaterniond q(R);

    // Construct SO3 from R
    Sophus::SO3d SO3_R(R);

    // Construct SO3 from quaternion q
    Sophus::SO3d SO3_q(q);

    // SO3_R and SO3_q represent the same matrix
    std::cout << "SO(3) from matrix:\n" << SO3_R.matrix() << std::endl;
    std::cout << "SO(3) from quaternion:\n" << SO3_q.matrix() << std::endl;

    // Lie algebra via logarithmic mapping
    Eigen::Vector3d so3 = SO3_R.log();

    std::cout << "so3 = " << so3.transpose() << std::endl;

    // hat is vec to mat; vee is mat to vec
    std::cout << "so3 hat=\n" << Sophus::SO3d::hat(so3) << std::endl;
    std::cout << "so3 hat vee= " << Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose()
        << std::endl;

    // Perturbation Model
    Eigen::Vector3d update_so3(1e-6, 1e-6, 1e-6); // this is a small update to X

    // Left Perturbation
    Sophus::SO3d SO3_updated_left_perturb = Sophus::SO3d::exp(update_so3) * SO3_R;
    Eigen::Vector3d so3_updated_left_perturb = SO3_updated_left_perturb.log();
    std::cout << "SO3 updated (left perturbation) = \n" << SO3_updated_left_perturb.matrix() << std::endl;   
    std::cout << "so3 updated (left perturbation) = " << so3_updated_left_perturb.transpose() << std::endl;

    // Right Perturbation
    Sophus::SO3d SO3_updated_right_perturb = SO3_R * Sophus::SO3d::exp(update_so3);
    Eigen::Vector3d so3_updated_right_perturb = SO3_updated_right_perturb.log();
    std::cout << "SO3 updated (right perturbation) = \n" << SO3_updated_right_perturb.matrix() << std::endl;   
    std::cout << "so3 updated (right perturbation) = " << so3_updated_right_perturb.transpose() << std::endl;   

    /*
        Result analysis:

        90 degree rotation around Z plus 1 unit translation on X (assumed linear interpolation)
        gives the end point [0.785398 -0.785398         0         0         0    1.5708] from 
        the start point [0  0   0   0   0   0],
        where the tangent of the curve's end point is 90 degree pointing the -Y direction,
        and the length is 0.785398^2 + (-0.785398)^2 = 1.

        |                      *
        |                      *              
        |                     *
        |                    *
        |                  *
        |               *
        |           *   
        |      *         
        |*_______________________
    */

    // Rotation matrix with 90 degrees along X and Y axis
    std::cout << "\n********SO(3) - 90x90x0********\n" << std::endl;

    // To rotation matrix
    auto R_90x90x0 = 
        Eigen::AngleAxisd(0.5*M_PI, Eigen::Vector3d::UnitX())
        * Eigen::AngleAxisd(0.5*M_PI, Eigen::Vector3d::UnitY())
        * Eigen::AngleAxisd(0.0*M_PI, Eigen::Vector3d::UnitZ());

    // To quaternion
    Eigen::Quaterniond q_90x90x0(R_90x90x0);

    // Construct SO3 from R
    Sophus::SO3d SO3_R_90x90x0(R_90x90x0);

    // Construct SO3 from quaternion q
    Sophus::SO3d SO3_q_90x90x0(q_90x90x0);

    // SO3_R and SO3_q represent the same matrix
    std::cout << "SO(3) 90x90x0 from matrix:\n" << SO3_R_90x90x0.matrix() << std::endl;
    std::cout << "SO(3) 90x90x0 from quaternion:\n" << SO3_q_90x90x0.matrix() << std::endl;

    // Lie algebra via logarithmic mapping
    Eigen::Vector3d so3_90x90x0 = SO3_R_90x90x0.log();

    std::cout << "so3 90x90x0 = " << so3_90x90x0.transpose() << std::endl;

    // hat is vec to mat; vee is mat to vec
    std::cout << "so3 90x90x0 hat=\n" << Sophus::SO3d::hat(so3_90x90x0) << std::endl;
    std::cout << "so3 90x90x0 hat vee= " << Sophus::SO3d::vee(Sophus::SO3d::hat(so3_90x90x0)).transpose()
        << std::endl;

    // Perturbation Model
    Eigen::Vector3d update_so3_90x90x0(1e-6, 1e-6, 1e-6); // this is a small update to X

    // Left Perturbation
    Sophus::SO3d SO3_updated_left_perturb_90x90x0 = Sophus::SO3d::exp(update_so3_90x90x0) * SO3_R_90x90x0;
    Eigen::Vector3d so3_updated_left_perturb_90x90x0 = SO3_updated_left_perturb_90x90x0.log();
    std::cout << "SO3 90x90x0 updated (left perturbation) = \n" << SO3_updated_left_perturb_90x90x0.matrix() << std::endl;   
    std::cout << "so3 90x90x0 updated (left perturbation) = " << so3_updated_left_perturb_90x90x0.transpose() << std::endl;   

    // Right Perturbation
    Sophus::SO3d SO3_updated_right_perturb_90x90x0 = SO3_R_90x90x0 * Sophus::SO3d::exp(update_so3_90x90x0);
    Eigen::Vector3d so3_updated_right_perturb_90x90x0 = SO3_updated_right_perturb_90x90x0.log();
    std::cout << "SO3 90x90x0 updated (right perturbation) = \n" << SO3_updated_right_perturb_90x90x0.matrix() << std::endl;   
    std::cout << "so3 90x90x0 updated (right perturbation) = " << so3_updated_right_perturb_90x90x0.transpose() << std::endl;   

    // With the aforementioned obtained rotation, now plus translation of 1 along X
    std::cout << "\n********SE(3) - 90 - 1********\n" << std::endl;
    Eigen::Vector3d t(1, 0, 0);            // translation 1 along X
    Sophus::SE3d SE3_Rt(R, t);             // construct SE3 from R and t
    Sophus::SE3d SE3_qt(q, t);             // or construct SE3 from q and t

    // Lie algebra via logarithmic mapping
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    Vector6d se3 = SE3_Rt.log();
    std::cout << "se3 = " << se3.transpose() << std::endl;

    // se3 hat and vee
    std::cout << "se3 hat = \n" << Sophus::SE3d::hat(se3) << std::endl;
    std::cout << "se3 hat vee = " << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)).transpose()
        << std::endl;

    // Perturbation Model
    Vector6d update_se3;
    update_se3.setZero();
    update_se3(0, 0) = 1e-6;
    Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_Rt;
    Vector6d se3_updated = SE3_updated.log();
    std::cout << "SE3 updated = " << std::endl << SE3_updated.matrix() << std::endl;
    std::cout << "se3 updated = " << se3_updated.transpose() << std::endl;

    // Rotation of 360 degree around Z, meanwhile 1 unit translation along X
    std::cout << "\n********SE(3) - 360 - 4********\n" << std::endl;

    // 360 degree rotation
    Eigen::Matrix3d R360 = Eigen::AngleAxisd(
        M_PI * 2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();

    Eigen::Vector3d t4(4, 0, 0);            // translation 4 along X
    Sophus::SE3d SE3_Rt_360_4(R360, t4);             // construct SE3 from R and t

    // Lie algebra via logarithmic mapping
    Vector6d se3_360_4 = SE3_Rt_360_4.log();
    std::cout << "se3_360_4 = " << se3_360_4.transpose() << std::endl;

    // se3 hat and vee
    std::cout << "se3_360_4 hat = \n" << Sophus::SE3d::hat(se3_360_4) << std::endl;
    std::cout << "se3_360_4 hat vee = " << Sophus::SE3d::vee(Sophus::SE3d::hat(se3_360_4)).transpose()
        << std::endl;

    // Perturbation Model
    Vector6d update_se3_360_4;
    update_se3_360_4.setZero();
    update_se3_360_4(0, 0) = 1e-6;
    Sophus::SE3d SE3_updated_360_4 = Sophus::SE3d::exp(update_se3_360_4) * SE3_Rt_360_4;
    Vector6d se3_updated_360_4 = SE3_updated_360_4.log();
    std::cout << "SE3_360_4 updated = " << std::endl << SE3_updated_360_4.matrix() << std::endl;
    std::cout << "se3_360_4 updated = " << se3_updated_360_4.transpose() << std::endl;
    
    /*
        should give [4  4.89859e-16            0            0            0 -2.44929e-16]
    */

    return 0;
}