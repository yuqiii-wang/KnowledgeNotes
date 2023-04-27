#include <algorithm>
#include <cmath>
#include <math.h>
#include <limits>
#include <algorithm>
#include <vector>

#include <Eigen/Core>
#include <ceres/rotation.h>
#include <ceres/ceres.h>
#include "ceres/problem.h"
#include "ceres/internal/autodiff.h"


#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"

// #include <glog/logging.h>


#ifndef SE3_TOOLS
#define SE3_TOOLS

typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;

struct BAL {
private:
    int num_cameras_;
    int num_points_;
    int num_observations_;
    int num_parameters_;

    int* point_index_;
    int* camera_index_;
    double* observations_;
    double* parameters_;

public:
    const static int point_block_size = 3;
    const static int camera_block_size = 9;

    int friend readData(const std::string& filename, BAL& bal);
    int friend writeToPLYFile(const std::string& filename, BAL& bal);

    int getNumCameras()                      {return num_cameras_;}
    int getNumPoints()                       {return num_points_;}
    int getNumObservations()                 {return num_observations_;}

    const double* getObservations()const     { return observations_;             }
    const int* getCameraIndex()const         { return camera_index_;             }
    const int* getPointIndex()const          { return point_index_;             }


    double* mutable_points()                 { return parameters_ + camera_block_size * num_cameras_; }
    double* mutable_cameras()                { return parameters_;               }
    int normalize();
    void CameraToAngelAxisAndCenter(const double* camera, 
                                            double* angle_axis,
                                            double* center) const;
    void AngleAxisAndCenterToCamera(const double* angle_axis,
                                            const double* center,
                                            double* camera) const;
    double Median(std::vector<double>* data);
    void perturb(const double rotation_sigma, 
                         const double translation_sigma,
                         const double point_sigma);
};

// camera : 9 dims array with 
// [0-2] : angle-axis rotation 
// [3-5] : translateion
// [6-8] : camera parameter, [6] focal length, [7-8] second and forth order radial distortion
// point : 3D location. 
// predictions : 2D predictions with center of the image plane. 
template<typename T>
inline bool CamProjectionWithDistortion(const T* camera, const T* point, T* predictions){
    // Rodrigues' formula
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5] are the translation
    p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];

    // Compute the center fo distortion
    T xp = -p[0]/p[2];
    T yp = -p[1]/p[2];

    // Apply second and fourth order radial distortion
    const T& l1 = camera[7];
    const T& l2 = camera[8];

    T r2 = xp*xp + yp*yp;
    T distortion = T(1.0) + r2 * (l1 + l2 * r2);

    const T& focal = camera[6];
    predictions[0] = focal * distortion * xp;
    predictions[1] = focal * distortion * yp;

    return true;
}

template<typename T>
void FscanfOrDie(FILE *fptr, const char *format, T *value) {
  int num_scanned = fscanf(fptr, format, value);
  if (num_scanned != 1) {
      LOG(FATAL) << "Invalid UW data file.";
  }
}

int readData(const std::string& filename, BAL& bal)
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

    bal.num_parameters_ = BAL::camera_block_size * bal.num_cameras_ + BAL::point_block_size * bal.num_points_;
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


double BAL::Median(std::vector<double>* data)
{
  int n = data->size();
  std::vector<double>::iterator mid_point = data->begin() + n/2;
  std::nth_element(data->begin(),mid_point,data->end());
  return *mid_point;
}


int BAL::normalize()
{
  // Compute the marginal median of the geometry
  std::vector<double> tmp(num_points_);
  Eigen::Vector3d median;
  double* points = mutable_points();
  for(int i = 0; i < point_block_size; ++i){
    for(int j = 0; j < num_points_; ++j){
      tmp[j] = points[point_block_size * j + i];      
    }
    median(i) = Median(&tmp);
  }

  for(int i = 0; i < num_points_; ++i){
    VectorRef point(points + point_block_size * i, point_block_size);
    tmp[i] = (point - median).lpNorm<1>();
  }

  const double median_absolute_deviation = Median(&tmp);

  // Scale so that the median absolute deviation of the resulting
  // reconstruction is 100

  const double scale = 100.0 / median_absolute_deviation;

  // X = scale * (X - median)
  for(int i = 0; i < num_points_; ++i){
    VectorRef point(points + point_block_size * i, point_block_size);
    point = scale * (point - median);
  }

  double* cameras = mutable_cameras();
  double angle_axis[point_block_size];
  double center[point_block_size];
  for(int i = 0; i < num_cameras_ ; ++i){
    double* camera = cameras + camera_block_size * i;
    CameraToAngelAxisAndCenter(camera, angle_axis, center);
    // center = scale * (center - median)
    VectorRef(center,point_block_size) = scale * (VectorRef(center,point_block_size)-median);
    AngleAxisAndCenterToCamera(angle_axis, center,camera);
  }
  return 0;
}

int writeToPLYFile(const std::string& filename, BAL& bal) {
  std::ofstream of(filename.c_str());

  of<< "ply"
    << '\n' << "format ascii 1.0"
    << '\n' << "element vertex " << bal.num_cameras_ + bal.num_points_
    << '\n' << "property float x"
    << '\n' << "property float y"
    << '\n' << "property float z"
    << '\n' << "property uchar red"
    << '\n' << "property uchar green"
    << '\n' << "property uchar blue"
    << '\n' << "end_header" << std::endl;

    // Export extrinsic data (i.e. camera centers) as green points.
    double angle_axis[3];
    double center[BAL::point_block_size];
    for(int i = 0; i < bal.num_cameras_ ; ++i){
      const double* camera = bal.parameters_ + BAL::camera_block_size * i;
      bal.CameraToAngelAxisAndCenter(camera, angle_axis, center);
      of << center[0] << ' ' << center[1] << ' ' << center[2]
         << "0 255 0" << '\n';
    }

    // Export the structure (i.e. 3D Points) as white points.
    const double* points = bal.parameters_ + BAL::camera_block_size * bal.num_cameras_;
    for(int i = 0; i < bal.num_points_; ++i){
      const double* point = points + i * BAL::point_block_size;
      for(int j = 0; j < BAL::point_block_size; ++j){
        of << point[j] << ' ';
      }
      of << "255 255 255\n";
    }
    of.close();

    return 0;
}


void BAL::CameraToAngelAxisAndCenter(const double* camera, 
                                            double* angle_axis,
                                            double* center) const{
    VectorRef angle_axis_ref(angle_axis,point_block_size);
    angle_axis_ref = ConstVectorRef(camera,point_block_size);

    // c = -R't
    Eigen::VectorXd inverse_rotation = -angle_axis_ref;
    ceres::AngleAxisRotatePoint(inverse_rotation.data(),
                         camera +  camera_block_size - 6,
                         center);
    VectorRef(center,point_block_size) *= -1.0;
}

void BAL::AngleAxisAndCenterToCamera(const double* angle_axis,
                                            const double* center,
                                            double* camera) const{
    ConstVectorRef angle_axis_ref(angle_axis,point_block_size);
    VectorRef(camera, point_block_size) = angle_axis_ref;

    // t = -R * c 
    ceres::AngleAxisRotatePoint(angle_axis,center,camera+ camera_block_size - 6);
    VectorRef(camera +  camera_block_size - 6,point_block_size) *= -1.0;
}

inline double RandDouble()
{
    double r = static_cast<double>(rand());
    return r / RAND_MAX;
}

inline double RandNormal()
{
    double x1, x2, w;
    do{
        x1 = 2.0 * RandDouble() - 1.0;
        x2 = 2.0 * RandDouble() - 1.0;
        w = x1 * x1 + x2 * x2;
    }while( w >= 1.0 || w == 0.0);

    w = sqrt((-2.0 * log(w))/w);
    return x1 * w;
}

void PerturbPoint3(const double sigma, double* point)
{
  for(int i = 0; i < 3; ++i)
    point[i] += RandNormal()*sigma;
}

void BAL::perturb(const double rotation_sigma, 
                         const double translation_sigma,
                         const double point_sigma){
   assert(point_sigma >= 0.0);
   assert(rotation_sigma >= 0.0);
   assert(translation_sigma >= 0.0);

   double* points = mutable_points();
   if(point_sigma > 0){
     for(int i = 0; i < num_points_; ++i){
       PerturbPoint3(point_sigma, points + point_block_size * i);
     }
   }

   for(int i = 0; i < num_cameras_; ++i){
     double* camera = mutable_cameras() +  camera_block_size * i;

     double angle_axis[point_block_size];
     double center[point_block_size];
     // Perturb in the rotation of the camera in the angle-axis
     // representation
     CameraToAngelAxisAndCenter(camera, angle_axis, center);
     if(rotation_sigma > 0.0){
       PerturbPoint3(rotation_sigma, angle_axis);
     }
     AngleAxisAndCenterToCamera(angle_axis, center,camera);

     if(translation_sigma > 0.0)
        PerturbPoint3(translation_sigma, camera +  camera_block_size - 6);
   }
}

#endif // SE3_TOOLS