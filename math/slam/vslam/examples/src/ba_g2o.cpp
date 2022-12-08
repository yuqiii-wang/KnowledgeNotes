#include <iostream>
#include <fstream>

#include "se3Tools.hpp"

class VertexCameraBAL;
class VertexPointBAL;
class EdgeObservationBAL;

// template <int kNumResiduals,
//           typename ParameterDims,
//           typename Functor,
//           typename T>
// inline bool AutoDifferentiate(const Functor& functor,
//                               T const* const* parameters,
//                               int dynamic_num_outputs,
//                               T* function_value,
//                               T** jacobians) 

struct computeResiduals
{
    computeResiduals(){};

    template<typename T>
    bool operator() ( const T* camera, const T* point, T* residuals ) const
    {
        T predictions[2];
        CamProjectionWithDistortion ( camera, point, predictions );
        residuals[0] = predictions[0] - T ( measurement() ( 0 ) );
        residuals[1] = predictions[1] - T ( measurement() ( 1 ) );

        return true;
    }
} computeResiduals;

// typedef ceres::internal::AutoDifferentiate<2, VertexCameraBAL::Dimension, computeResiduals, double> BalAutoDifferentiate;

class VertexCameraBAL : public g2o::BaseVertex<BAL::camera_block_size,Eigen::VectorXd>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexCameraBAL() {}

    virtual bool read ( std::istream& /*is*/ )    { return false;}

    virtual bool write ( std::ostream& /*os*/ ) const { return false;}

    virtual void setToOriginImpl() {}

    virtual void oplusImpl ( const double* update )
    {
        Eigen::Vector3d::ConstMapType v ( update );
        _estimate += v;
    }
};


class VertexPointBAL : public g2o::BaseVertex<BAL::point_block_size, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexPointBAL() {}

    virtual bool read ( std::istream& /*is*/ )    { return false;}

    virtual bool write ( std::ostream& /*os*/ ) const { return false;}

    virtual void setToOriginImpl() {}

    virtual void oplusImpl ( const double* update )
    {
        Eigen::Vector3d::ConstMapType v ( update );
        _estimate += v;
    }
};

class EdgeObservationBAL : public g2o::BaseBinaryEdge<2, Eigen::Vector2d,VertexCameraBAL, VertexPointBAL>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeObservationBAL(){};

    virtual bool read ( std::istream& /*is*/ )    { return false;}

    virtual bool write ( std::ostream& /*os*/ ) const { return false;}

    virtual void computeError() override   // The virtual function comes from the Edge base class. Must define if you use edge.
    {
        const VertexCameraBAL* cam = static_cast<const VertexCameraBAL*> ( vertex ( 0 ) );
        const VertexPointBAL* point = static_cast<const VertexPointBAL*> ( vertex ( 1 ) );

        // _error is computed and stored.
        computeResiduals( cam->estimate().data(), point->estimate().data(), _error.data() );

    }

    virtual void linearizeOplus() override
    {
        // use numeric Jacobians
        // BaseBinaryEdge<2, Vector2d, VertexCameraBAL, VertexPointBAL>::linearizeOplus();
        // return;
        
        // using autodiff from ceres. Otherwise, the system will use g2o numerical diff for Jacobians

        const VertexCameraBAL* cam = static_cast<const VertexCameraBAL*> ( vertex ( 0 ) );
        const VertexPointBAL* point = static_cast<const VertexPointBAL*> ( vertex ( 1 ) );

        Eigen::Matrix<double, Dimension, VertexCameraBAL::Dimension, Eigen::RowMajor> dError_dCamera;
        Eigen::Matrix<double, Dimension, VertexPointBAL::Dimension, Eigen::RowMajor> dError_dPoint;
        double *parameters[] = { const_cast<double*> ( cam->estimate().data() ), const_cast<double*> ( point->estimate().data() ) };
        double *jacobians[] = { dError_dCamera.data(), dError_dPoint.data() };
        double value[Dimension];

        bool diffState = ceres::internal::AutoDiff<2, VertexCameraBAL::Dimension, computeResiduals, double>( computeResiduals( cam->estimate().data(), point->estimate().data(), _error.data() ), parameters, Dimension, value, jacobians );

        // bool diffState = BalAutoDifferentiate( *this, parameters, Dimension, value, jacobians );

        // copy over the Jacobians (convert row-major -> column-major)
        if ( diffState )
        {
            _jacobianOplusXi = dError_dCamera;
            _jacobianOplusXj = dError_dPoint;
        }
        else
        {
            assert ( 0 && "Error while differentiating" );
            _jacobianOplusXi.setZero();
            _jacobianOplusXj.setZero();
        }
    }
};

int main()
{

    BAL bal;

    readData("./landmarks.txt", bal);

    bal.normalize();
    bal.perturb(0.0, 0.0, 0.0);

    const int point_block_size = 3;
    const int camera_block_size = 9;
    double* points = bal.mutable_points();
    double* cameras = bal.mutable_cameras();

    writeToPLYFile("inits.ply", bal);

    return 0;
}