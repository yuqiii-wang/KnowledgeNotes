# Ceres Optimization Tools

## The BALProblem Example

```cpp
// read BA data from a file
BALProblem bal_problem(filename);

// take each dimension's median; each point minus median as the deviation
void BALProblem::Normalize(){

    // compute three medians for the three dimensions
    for(int i = 0; i < 3; ++i){
        for(int j = 0; j < num_points_; ++j){
            tmp[j] = points[3 * j + i];      
        }
        median(i) = Median(&tmp);
    }

    // normalize each points
    for(int i = 0; i < num_points_; ++i){
        VectorRef point(points + 3 * i, 3);
        tmp[i] = (point - median).lpNorm<1>(); // L1 norm
    }

    // camera : 9 dims array
    // [0−2] : angle−axis rotation
    // [3−5] : translation
    // [6−8] : camera parameter, [6] focal length, [7−8] second and forth order radial
    // distortion
    for(int i = 0; i < num_cameras_ ; ++i){
        double* camera = cameras + camera_block_size() * i;
        // rotate the camera to origin angle (0,0,0), 
        // the corresponding translation is output to center
        CameraToAngelAxisAndCenter(camera, angle_axis, center);
        // center = scale * (center - median), normalize translation
        VectorRef(center,3) = scale * (VectorRef(center,3)-median);
        // rotate back to where the camera was
        AngleAxisAndCenterToCamera(angle_axis, center,camera);
    }
}
bal_problem.Normalize();

// just add some Gaussian noises
void BALProblem::Perturb(const double rotation_sigma, 
                         const double translation_sigma,
                         const double point_sigma){
    for(int i = 0; i < 3; ++i)
        point[i] += RandNormal()*sigma;
}
bal_problem.Perturb(params.rotation_sigma, params.translation_sigma,
                        params.point_sigma);

// Construct the least squares problem
// defined cost function and loss function
void BuildProblem(BALProblem* bal_problem, Problem* problem, const BundleParams& params)
{
    for(int i = 0; i < bal_problem->num_observations(); ++i){

        // Each Residual block takes a point and a camera as input 
        // and outputs a 2 dimensional Residual
        CostFunction* cost_function = SnavelyReprojectionError::Create(observations[2*i + 0], observations[2*i + 1]);
        LossFunction* loss_function = params.robustify ? new HuberLoss(1.0) : NULL;

        problem->AddResidualBlock(cost_function, loss_function, camera, point);
    }
}
ceres::Problem problem;
BuildProblem(&bal_problem, &problem, params);

// Ceres built-in solver
ceres::Solve(options, &problem, &summary);
```

The most interesting function is how the cost function is designed to display projection error.

* `operator()` defines residual as the discrepancies between predicts and observations
* `CamProjectionWithDistortion` considers how the predicts are computed

```cpp
class SnavelyReprojectionError
{
public:
    SnavelyReprojectionError(double observation_x, double observation_y):observed_x(observation_x),observed_y(observation_y){}

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

    // camera : 9 dims array
    // [0-2] : angle-axis rotation
    // [3-5] : translation
    // [6-8] : camera parameter, [6] focal length, [7-8] second and forth order radial distortion
    // point : 3D location.
    // predictions : 2D predictions with center of the image plane.
    template<typename T>
    static inline bool CamProjectionWithDistortion(
                    const T *camera, const T *point, T *predictions) {
        // Rodrigues' formula
        T p[3];
        AngleAxisRotatePoint(camera, point, p);
        // camera[3,4,5] are the translation
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        
        // Compute the center fo distortion
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];

        // Apply second and fourth order radial distortion
        const T &l1 = camera[7];
        const T &l2 = camera[8];

        T r2 = xp * xp + yp * yp;
        T distortion = T(1.0) + r2 * (l1 + l2 * r2);

        const T &focal = camera[6];
        predictions[0] = focal * distortion * xp;
        predictions[1] = focal * distortion * yp;
        return true;
    }

    // 
    static ceres::CostFunction* Create(const double observed_x, const double observed_y){
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError,2,9,3>(
            new SnavelyReprojectionError(observed_x,observed_y)));
    }


private:
    double observed_x;
    double observed_y;
};
```

## Important Ceres Functions

* `AutoDiffCostFunction` and cost function

The cost function is a custom class that must have `bool operator()(const T* const x , const T* const y, T* e) const` passed to Ceres' `AutoDiffCostFunction` that compute error. 

```cpp
class MyScalarCostFunctor {
  MyScalarCostFunctor(double k): k_(k) {}

  template <typename T>
  bool operator()(const T* const x , const T* const y, T* e) const {
    e[0] = k_ - x[0] * y[0] - x[1] * y[1];
    return true;
  }

 private:
  double k_;
};
```

```cpp
CostFunction* cost_function
    = new AutoDiffCostFunction<MyScalarCostFunctor, 1, 2, 2>(
        new MyScalarCostFunctor(1.0));              ^  ^  ^
                                                    |  |  |
                Dimension of residual --------------+  |  |
                Dimension of x (params to adjust) -----+  |
                Dimension of y (observations) ------------+
```

* `Problem::AddResidualBlock()`

It adds a CostFunction, an optional LossFunction and connects the CostFunction to a set of parameter block.

Typically, there are four args to `AddResidualBlock`.
```cpp
AddResidualBlock(cost_function, loss_function, x, y);
```

* `ceres::Problem` and `ceres::Solve`

`ceres::Problem` has `AddResidualBlock` that is required to construct the least squares problem.


`ceres::Solve` takes some config and solve the least squares problem. 
Typically, the below options are set.

```cpp
ceres::Solver::Options options;
options.max_num_iterations = 100;
options.minimizer_progress_to_stdout = true;
options.num_threads = 12;
ceres::StringToLinearSolverType("dense_schur", &options.linear_solver_type);
ceres::StringToTrustRegionStrategyType("levenberg_marquardt",
                                    &options.trust_region_strategy_type);
ceres::StringToSparseLinearAlgebraLibraryType("suite_sparse", &options.sparse_linear_algebra_library_type);
ceres::StringToDenseLinearAlgebraLibraryType("eigen", &options.dense_linear_algebra_library_type);
options.gradient_tolerance = 1e-8;
options.function_tolerance = 1e-8;

ceres::Solver::Summary summary;
ceres::Solve(options, &problem, &summary);
```

Variable elimination ordering can boost computation speed. 
For example, consider the below two equations:
$$
\begin{align*}
x+y&=3 &\quad (1) \\\\
3x+2y&=5 &\quad (2)
\end{align*}
$$

Apparently, eliminating one variable in $(1)$ then back substituting it in $(2)$ for $y$, is less costly than the opposite way's.

In SLAM, points come before the cameras in variable elimination.
```cpp
ceres::ParameterBlockOrdering* ordering = new ceres::ParameterBlockOrdering;

// The points come before the cameras
for(int i = 0; i < num_points; ++i)
   ordering->AddElementToGroup(points + point_block_size * i, 0);
for(int i = 0; i < num_cameras; ++i)
    ordering->AddElementToGroup(cameras + camera_block_size * i, 1);

// set this ordering to options
options->linear_solver_ordering.reset(ordering);
```