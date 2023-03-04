# Lidar Mapping Google Cartographer

## Local Mapping

### Scans

Laser scans are recorded and are transformed by a static homogeneous transformation matrix to a robot's origin.

Scan point set: $H=\{h_k\}_{k=1,2,3,...,K}, h_k \in R^2$

The pose $\xi$ of the scan frame in the
submap frame transformation: $T_{\xi}$

### Submaps

A few consecutive scans are used to build a submap. 
These submaps take the form of probability grids $M : rZ × rZ \rightarrow [p_{min} , p_{max} ]$ which map from discrete grid points at a given resolution $r$, for example $5$ cm, to values. $Z$ is the height/width of the submap image.

$p_{hit}$ and $p_{miss}$ are computed by the number of hits and misses over total laser scannings on a particular grid point.

If $M(x)$ has not yet been observed, it is assigned $p_{hit}$ or $p_{miss}$; if $M(x)$ is observed, it can be updated.

<div style="display: flex; justify-content: center;">
      <img src="imgs/google_cart_scan.png" width="40%" height="40%" alt="google_cart_scan">
</div>
</br>

$$
\text{odds}(p)=\frac{p}{1-p}
$$

Define the formula to compute the new grid point:

$$
M_{new}(x) = \text{clamp}\Big(
    \frac{1}{
    \text{odds} \big( \text{odds}\big(M_{old}(x)\big) \cdot \text{odds}(p_{hit}) \big)}
\Big)
$$
where $\text{odds}\big(M_{old}(x)\big) \cdot \text{odds}(p_{hit})$ is the update to the old grid point,
and $\text{clamp}$ is a function that contains a value $x$ within an interval $[a,b]$
$$
\text{clamp}(x) = \text{max}
\big( a, \text{min}(x, b) \big) \in [a, b]
$$

### Scan Matching

The scan matcher is responsible for
finding a scan pose $\xi$ that maximizes the probabilities at the scan points in the submap.

$$
arg \space \underset{\xi}{min} \sum_{k=1}^K \big(1-M_{smooth}(T_\xi h_k)\big)^2
$$
where $T_\xi$ transforms $h_k$ from the scan frame to the submap
frame according to the scan pose.

$M_{smooth}: R^2 \rightarrow R$ by nature is Bicubic Interpolation whose output is in the range $(0, 1)$ (when near to one, the matching result is considered good). It is used to "smooth" the generated grid map.

## Closing Loops

### Closing Loop Optimization

Local closing loop problem refers to optimizing submap poses $\Xi^m = \{\xi^m_0, \xi^m_1, \xi^m_2, ..., \xi^m_k\}$ and scan poses $\Xi^s = \{\xi^s_0, \xi^s_1, \xi^s_2, ..., \xi^s_k\}$ over a route.

Closing loop optimization:

$$
arg\space \underset{\Xi^m， \Xi^s}{min} \frac{1}{2} \underset{i,j}{\sum} \rho \big( E^2(\xi^m_i, \xi^s_j; \Sigma_{i,j}, \xi_{i,j}) \big)
$$
where constraints take the form of relative poses $\xi_{i,j}$ (describes where in the submap coordinate frame the scan was matched), 
and associated covariance matrices $\Sigma_{i,j}$, for input pair $\xi^m_i, \xi^s_j$. The employed loss $\rho$ is *Huber Loss*.

In detail,
$$
E^2(\xi^m_i, \xi^s_j; \Sigma_{i,j}, \xi_{i,j}) 
\\ =
\bold{e}(\xi^m_i, \xi^s_j; \xi_{i,j})^T \Sigma_{i,j} \bold{e}(\xi^m_i, \xi^s_j; \xi_{i,j})
$$
in which, $\bold{e}(\xi^m_i, \xi^s_j; \xi_{i,j})$ describes the error of robot one step pose against its scans and generated submap, such that

$$
\bold{e}(\xi^m_i, \xi^s_j; \xi_{i,j}) =
 \xi_{i,j} - 
\begin{bmatrix}
    R^{-1}_{\xi^m_i} (\bold{t}_{\xi^m_i} - \bold{t}_{\xi^s_j}) \\
    \xi^m_{i;\theta} - \xi^s_{j;\theta}
\end{bmatrix}
$$
where $R^{-1}_{\xi^m_i} (\bold{t}_{\xi^m_i} - \bold{t}_{\xi^s_j})$ describes the translation differences between $\xi^m_i$ and $\xi^s_j$ aligned to the submap coordinates by $R^{-1}_{\xi^m_i}$, 
and $\xi^m_{i;\theta} - \xi^s_{j;\theta}$ describes the gap of the two poses' orientations.

Recall that relative poses $\xi_{i,j}$ (describes where in the submap coordinate frame the scan was matched)

### Search Window $W$ For Fine-Tuning

Define a pose search window $W$,
and $M_{nearest}(T_{\xi}h_k)$ is the $T_{\xi \in W}$ transformed grid.
$\scriptsize{nearest}$ has its name for the search window $W$ does not deviate a lot from where it begins, and the change increment is small as well. 

$$
\xi^* = \argmax_{\xi \in W} \sum_{k=1}^K M_{nearest}(T_{\xi}h_k)
$$
where $\xi \in W$ means employment of a discrete search window. Define resolution $r$ and $\delta_\theta$ for the pose search window boundary $W_x=7 m, W_y=7m, W_\theta=30^\circ$, the discrete search step is

$$
w_x = \frac{W_x}{r}
, \qquad
w_y = \frac{W_y}{r}
, \qquad
w_\theta = \frac{W_\theta}{\delta_\theta}
$$

Accordingly, a naive approach for optimal $\xi^*$ can be computed by iterating all possible discrete steps such as

$$
\begin{align*}
\\& 
\rule{10cm}{1pt}
\\&
\text{Naive Search}
\\& 
\rule{10cm}{0.5pt}
\\& 
    bestScore \leftarrow -\infty
\\& 
    \bold{\text{for }} i_x = -w_x \bold{\text{ to }} w_x \bold{\text{ do }}
\\& \quad
    \bold{\text{for }} i_y = -w_y \bold{\text{ to }} w_y \bold{\text{ do }}
\\& \quad\quad
    \bold{\text{for }} i_\theta = -w_\theta \bold{\text{ to }} w_\theta \bold{\text{ do }}
\\& \quad\quad\quad
    score \leftarrow \sum_{k=1}^K M_{nearest}(T_{\xi_0+[ri_x,ri_y,\delta_\theta]}h_k)
\\& \quad\quad\quad
    \bold{\text{if }} score > bestScore \bold{\text{ then }}
\\& \quad\quad\quad\quad
    match \leftarrow \xi_0+[ri_x,ri_y,\delta_\theta]
\\& \quad\quad\quad\quad
    bestScore \leftarrow score
\\& \quad\quad\quad
    \bold{\text{end if }}
\\& \quad\quad
    \bold{\text{end for}}
\\& \quad
    \bold{\text{end for}}
\\&
    \bold{\text{end for}}
\\ &
\bold{\text{return }} match     \bold{\text{ and }} bestScore
\\& 
\rule{10cm}{0.5pt}
\end{align*}
$$

### Branch-And-Bound Scan Matching

Full search window traversal is time-consuming, hence employing-Branch and-Bound.
It uses a tree structure to represent the search space:
* root - all possibilities
* children nodes - form a
partition of their parent (a subset of parent possibilities)
* leaf nodes - singletons; each
represents a single feasible solution.

The search problem then becomes traversing the nodes of the tree.
To facilitate the traversal, should trim this tree and split branches by the condition $score$:

All nodes of a parent should NOT see their $score$ greater than their parent's $score$ (called *bound*).
So that if a inner node's $score$ is small, all its subset branches/nodes can be trimmed/removed (called *branch*).

##  Code

### GRPC

In Google Cartographer,
requests/responses are sent/received through gRPC (gRPC Remote Procedure Calls, a google high performance remote procedure call (RPC) framework) `grpc` (to replace ROS for cross-process communication).
`grpc.Write(request)` and `grpc.Read(response)` are the `grpc`-defined APIs to sending and receiving data across processes.

Below is an example of adding odometry sensor data from a client `add_odometry_client_` to master node `MapBuilderServer`.

First, should launch the master node `MapBuilderServer::MapBuilderServer(...)`, where `server_builder.registerHandler<handlers::AddOdometryDataHandler>();` registers the client odometry handler.

When there is new odometry data arrived, client serializes sensor data in a proto format then sends it to master by `add_odometry_client_->Write(request);`, and
`DEFINE_HANDLER_SIGNATURE` is used to define "topic" to which the master node should subscribe.
Inside `OnSensorData`, master adds data by queuing the data `EnqueueSensorData(std::move(sensor_data))`.

Finally, the master node `ProcessSensorDataQueue` by popping out sensor data.

```cpp
// cartographer/cloud/internal/map_builder_server.cc
MapBuilderServer::MapBuilderServer(...)
{
    ... 
    server_builder.RegisterHandler<handlers::AddOdometryDataHandler>();
    ...
    grpc_server_ = server_builder.Build();
    ...
}

// cartographer/cloud/internal/client/trajectory_builder_stub.cc
void TrajectoryBuilderStub::AddSensorData(
    const std::string& sensor_id, const sensor::OdometryData& odometry_data) {
  if (!add_odometry_client_) {
    add_odometry_client_ = absl::make_unique<
        async_grpc::Client<handlers::AddOdometryDataSignature>>(
        client_channel_);
  }
  proto::AddOdometryDataRequest request;
  CreateAddOdometryDataRequest(sensor_id, trajectory_id_, client_id_,
                               sensor::ToProto(odometry_data), &request);
  add_odometry_client_->Write(request);
}

// cartographer/cloud/internal/handlers/add_odometry_data_handler.h
DEFINE_HANDLER_SIGNATURE(
    AddOdometryDataSignature, async_grpc::Stream<proto::AddOdometryDataRequest>,
    google::protobuf::Empty,
    "/cartographer.cloud.proto.MapBuilderService/AddOdometryData")

class AddOdometryDataHandler
    : public AddSensorDataHandlerBase<AddOdometryDataSignature> {
 public:
  void OnSensorData(const proto::AddOdometryDataRequest& request) override;
};

// cartographer/cloud/internal/handlers/add_odometry_data_handler.cc
void AddOdometryDataHandler::OnSensorData(
    const proto::AddOdometryDataRequest& request) {
  // The 'BlockingQueue' returned by 'sensor_data_queue()' is already
  // thread-safe. Therefore it suffices to get an unsynchronized reference to
  // the 'MapBuilderContext'.
  GetUnsynchronizedContext<MapBuilderContextInterface>()->EnqueueSensorData(
      request.sensor_metadata().trajectory_id(),
      sensor::MakeDispatchable(request.sensor_metadata().sensor_id(),
                               sensor::FromProto(request.odometry_data())));

  // The 'BlockingQueue' in 'LocalTrajectoryUploader' is thread-safe.
  // Therefore it suffices to get an unsynchronized reference to the
  // 'MapBuilderContext'.
  if (GetUnsynchronizedContext<MapBuilderContextInterface>()
          ->local_trajectory_uploader()) {
    auto sensor_data = absl::make_unique<proto::SensorData>();
    *sensor_data->mutable_sensor_metadata() = request.sensor_metadata();
    *sensor_data->mutable_odometry_data() = request.odometry_data();
    GetUnsynchronizedContext<MapBuilderContextInterface>()
        ->local_trajectory_uploader()
        ->EnqueueSensorData(std::move(sensor_data));
  }
}

// cartographer/cloud/internal/map_builder_server.cc
void MapBuilderServer::ProcessSensorDataQueue() {
  LOG(INFO) << "Starting SLAM thread.";
  while (!shutting_down_) {
    kIncomingDataQueueMetric->Set(incoming_data_queue_.Size());
    std::unique_ptr<MapBuilderContextInterface::Data> sensor_data =
        incoming_data_queue_.PopWithTimeout(kPopTimeout);
    if (sensor_data) {
      grpc_server_->GetContext<MapBuilderContextInterface>()
          ->AddSensorDataToTrajectory(*sensor_data);
    }
  }
}
```

### The Master Node Map Server 

Start from `map_server` that acts as a master node/process managing/receiving requests from client nodes/processes.

```cpp
// cartographer/cloud/map_builder_server_main.cc
int main(int argc, char** argv) {
    ...
    cartographer::cloud::Run(FLAGS_configuration_directory,
                               FLAGS_configuration_basename);
}

void Run(const std::string& configuration_directory,
         const std::string& configuration_basename) {
    ...
    std::unique_ptr<MapBuilderServerInterface> map_builder_server =
      CreateMapBuilderServer(map_builder_server_options,
                             std::move(map_builder));

    // Starts the gRPC server, the 'LocalTrajectoryUploader' and the SLAM thread.
    map_builder_server->Start();
    map_builder_server->WaitForShutdown();

}

// cartographer/cloud/internal/map_builder_server.cc
void MapBuilderServer::Start() {
  shutting_down_ = false;
  if (local_trajectory_uploader_) {
    local_trajectory_uploader_->Start();
  }
  StartSlamThread();
  grpc_server_->Start();
}

void MapBuilderServer::StartSlamThread() {
  CHECK(!slam_thread_);

  // Start the SLAM processing thread.
  slam_thread_ = absl::make_unique<std::thread>(
      [this]() { this->ProcessSensorDataQueue(); });
}

void MapBuilderServer::ProcessSensorDataQueue() {
  while (!shutting_down_) {
    ... // popping out sensor data from queue
    if (sensor_data) {
      grpc_server_->GetContext<MapBuilderContextInterface>()
          ->AddSensorDataToTrajectory(*sensor_data);
    }
  }
}

// cartographer/cloud/internal/map_builder_context_impl.h
template <class SubmapType>
void MapBuilderContext<SubmapType>::AddSensorDataToTrajectory(
    const Data& sensor_data) {
  sensor_data.data->AddToTrajectoryBuilder(
      map_builder_server_->map_builder_->GetTrajectoryBuilder(
          sensor_data.trajectory_id));
}

// cartographer/sensor/internal/dispatchable.h
void AddToTrajectoryBuilder(
    mapping::TrajectoryBuilderInterface *const trajectory_builder) override {
  trajectory_builder->AddSensorData(sensor_id_, data_);
}
```

By now, trajectory builder has finished associated coming sensor data with callbacks.
When there is new sensor data arrives, `AddSensorDataToTrajectory(*sensor_data);` runs in a loop adding sensor data to the corresponding trajectory.

`LocalSlamResultCallback` is invoked when having accumulated enough sensor/range data.

```cpp
class TrajectoryBuilderInterface {
 public:
  struct InsertionResult {
    NodeId node_id;
    std::shared_ptr<const TrajectoryNode::Data> constant_data;
    std::vector<std::shared_ptr<const Submap>> insertion_submaps;
  };

  // A callback which is called after local SLAM processes an accumulated
  // 'sensor::RangeData'. If the data was inserted into a submap, reports the
  // assigned 'NodeId', otherwise 'nullptr' if the data was filtered out.
  using LocalSlamResultCallback =
      std::function<void(int /* trajectory ID */, common::Time,
                         transform::Rigid3d /* local pose estimate */,
                         sensor::RangeData /* in local frame */,
                         std::unique_ptr<const InsertionResult>)>;

  struct SensorId {
    enum class SensorType {
      RANGE = 0,
      IMU,
      ODOMETRY,
      FIXED_FRAME_POSE,
      LANDMARK,
      LOCAL_SLAM_RESULT
    };
  };
};

// `LocalSlamResultCallback` is defined as a lambda
// `OnLocalSlamResult` just serializes result data to proto format
mapping::TrajectoryBuilderInterface::LocalSlamResultCallback = 
[this](int trajectory_id, common::Time time,
              transform::Rigid3d local_pose, sensor::RangeData range_data,
              std::unique_ptr<
                  const mapping::TrajectoryBuilderInterface::InsertionResult>
                  insertion_result) {
  auto it = client_ids_.find(trajectory_id);
  map_builder_server_->OnLocalSlamResult(trajectory_id, it->second, time,
                                         local_pose, std::move(range_data),
                                         std::move(insertion_result));
};
```

`TrajectoryBuilderStub` is the class that gradually takes sensor data and builds a trajectory.
In construction, if `local_slam_result_callback` is not a `nullptr`, it (the client node `receive_local_slam_results_client_.Write(request);`) launches a thread that runs `RunLocalSlamResultsReader` 

```cpp
// cartographer/cloud/internal/client/trajectory_builder_stub.cc
TrajectoryBuilderStub::TrajectoryBuilderStub(
    std::shared_ptr<::grpc::Channel> client_channel, const int trajectory_id,
    const std::string& client_id,
    LocalSlamResultCallback local_slam_result_callback)
    : client_channel_(client_channel),
      trajectory_id_(trajectory_id),
      client_id_(client_id),
      receive_local_slam_results_client_(client_channel_) {
  if (local_slam_result_callback) {
    proto::ReceiveLocalSlamResultsRequest request;
    request.set_trajectory_id(trajectory_id);
    receive_local_slam_results_client_.Write(request);
    auto* receive_local_slam_results_client_ptr =
        &receive_local_slam_results_client_;
    receive_local_slam_results_thread_ = absl::make_unique<std::thread>(
        [receive_local_slam_results_client_ptr, local_slam_result_callback]() {
          RunLocalSlamResultsReader(receive_local_slam_results_client_ptr,
                                    local_slam_result_callback);
        });
  }
}


void TrajectoryBuilderStub::RunLocalSlamResultsReader(
    async_grpc::Client<handlers::ReceiveLocalSlamResultsSignature>* client,
    LocalSlamResultCallback local_slam_result_callback) {
  proto::ReceiveLocalSlamResultsResponse response;
  while (client->StreamRead(&response)) {
    int trajectory_id = response.trajectory_id();
    common::Time time = common::FromUniversal(response.timestamp());
    transform::Rigid3d local_pose = transform::ToRigid3(response.local_pose());
    sensor::RangeData range_data = sensor::FromProto(response.range_data());
    auto insertion_result =
        response.has_insertion_result()
            ? absl::make_unique<InsertionResult>(
                  InsertionResult{mapping::NodeId{
                      response.insertion_result().node_id().trajectory_id(),
                      response.insertion_result().node_id().node_index()}})
            : nullptr;
    local_slam_result_callback(trajectory_id, time, local_pose, range_data,
                               std::move(insertion_result));
  }
  client->StreamFinish();
}
```

### Scan Match and Pose Optimization

Basically, just use laser scan cloud points' world frame position `transform * point` matching against the contour `grid_.limits()` of the existing map.
The matching error (smoothed by bicubic interpolation) is the residual that is fed to ceres optimization that adjusts the laser radiation center pose `pose_estimate`/`transform` to reduce the residual.

Bicubic interpolation would see small interpolated value/error `grid_.GetCorrespondenceCost( Eigen::Array2i(column - kPadding, row - kPadding ))` for elements/cloud points within the existing map; 
large error `*value = kMaxCorrespondenceCost;` for elements/cloud points outside the map boundary.

* The use of `ceres::BiCubicInterpolator`:

```cpp
class GridArrayAdapter {
    enum { DATA_DIMENSION = 1 };
    void GetValue(int row, int col, double* f) const;
};
```
where `GetValue` gives the value of a function (possibly vector valued) for any pair of integers `row` and `col` and the `enum DATA_DIMENSION` indicates the dimensionality of the function being interpolated. 
In the below case, `DATA_DIMENSION = 1` indicates there is only 1d error (correspondence cost) to measure.

The bicubic interpolator produces a smooth approximation to evaluate the $f(row,col), \frac{\partial f(row,col)}{\partial \space row}$, and $\frac{\partial f(row,col)}{\partial \space col}$ at any any point in the real plane.
In other words, `interpolator.Evaluate(x,y,cost)` interpolates what cost should be at $(x,y)$.

```cpp
void CeresScanMatcher2D::Match(const Eigen::Vector2d& target_translation,
                               const transform::Rigid2d& initial_pose_estimate,
                               const sensor::PointCloud& point_cloud,
                               const Grid2D& grid,
                               transform::Rigid2d* const pose_estimate,
                               ceres::Solver::Summary* const summary) const {

    ceres::Problem problem;

    problem.AddResidualBlock(
          CreateOccupiedSpaceCostFunction2D(
              options_.occupied_space_weight() /
                  std::sqrt(static_cast<double>(point_cloud.size())),
              point_cloud, grid),
          nullptr /* loss function */, ceres_pose_estimate);

    problem.AddResidualBlock(
      TranslationDeltaCostFunctor2D::CreateAutoDiffCostFunction(
          options_.translation_weight(), target_translation),
      nullptr /* loss function */, ceres_pose_estimate);

    problem.AddResidualBlock(
      RotationDeltaCostFunctor2D::CreateAutoDiffCostFunction(
          options_.rotation_weight(), ceres_pose_estimate[2]),
      nullptr /* loss function */, ceres_pose_estimate);

  ceres::Solve(ceres_solver_options_, &problem, summary);

  *pose_estimate = transform::Rigid2d(
      {ceres_pose_estimate[0], ceres_pose_estimate[1]}, ceres_pose_estimate[2]);

}

class OccupiedSpaceCostFunction2D {
    
    bool operator()(const T* const pose, T* residual) const {
        const GridArrayAdapter adapter(grid_);
        ceres::BiCubicInterpolator<GridArrayAdapter> interpolator(adapter);
        const MapLimits& limits = grid_.limits();

        for (size_t i = 0; i < point_cloud_.size(); ++i) {
            // Note that this is a 2D point. The third component is a scaling factor.
            const Eigen::Matrix<T, 3, 1> point((T(point_cloud_[i].position.x())),
                                               (T(point_cloud_[i].position.y())),
                                               T(1.));
            const Eigen::Matrix<T, 3, 1> world = transform * point;
            interpolator.Evaluate(
                (limits.max().x() - world[0]) / limits.resolution() - 0.5 +
                    static_cast<double>(kPadding),
                (limits.max().y() - world[1]) / limits.resolution() - 0.5 +
                    static_cast<double>(kPadding),
                &residual[i]);
            residual[i] = scaling_factor_ * residual[i];
        }
        return true;
    }

    class GridArrayAdapter {
        enum { DATA_DIMENSION = 1 };
        void GetValue(const int row, const int column, double* const value) const {
            if (row < kPadding || column < kPadding || row >= NumRows() - kPadding ||
                column >= NumCols() - kPadding) {
              *value = kMaxCorrespondenceCost;
            } else {
              *value = static_cast<double>(grid_.GetCorrespondenceCost(
                  Eigen::Array2i(column - kPadding, row - kPadding)));
            }
        }
    };

};
```

### Laser Ray Process

Lidar data `RangeData` is defined as below.
Simply, `returns` is of `Eigen::Vector2i` representing the points hitting an obstacle.

```cpp
// Rays begin at 'origin'. 'returns' are the points where obstructions were
// detected. 'misses' are points in the direction of rays for which no return
// was detected, and were inserted at a configured distance. It is assumed that
// between the 'origin' and 'misses' is free space.
struct RangeData {
  Eigen::Vector3f origin;
  PointCloud returns;
  PointCloud misses;
};
```

`hit_table_` and `miss_table_` are `std::vector<uint16>` that map between probability and cost value.

`CastRays` takes care of range data `sensor::RangefinderPoint& hit : range_data.returns` that is of `Eigen::Vector3f position;` for each ray hit point (a hit point is where the laser ends at hitting an obstacle).
It pushes back `update_indices_.push_back(cell);` that will later be updated by `FinishUpdate`.

```cpp
ProbabilityGridRangeDataInserter2D::ProbabilityGridRangeDataInserter2D(
    const proto::ProbabilityGridRangeDataInserterOptions2D& options)
    : options_(options),
      hit_table_(ComputeLookupTableToApplyCorrespondenceCostOdds(
          Odds(options.hit_probability()))), // options.hit_probability() = 0.55
      miss_table_(ComputeLookupTableToApplyCorrespondenceCostOdds(
          Odds(options.miss_probability()))) {} // options.hit_probability() = 0.49

void ProbabilityGridRangeDataInserter2D::Insert(
    const sensor::RangeData& range_data, GridInterface* const grid) const {
  ProbabilityGrid* const probability_grid = static_cast<ProbabilityGrid*>(grid);
  CHECK(probability_grid != nullptr);
  // By not finishing the update after hits are inserted, we give hits priority
  // (i.e. no hits will be ignored because of a miss in the same cell).
  CastRays(range_data, hit_table_, miss_table_, options_.insert_free_space(),
           probability_grid);
  probability_grid->FinishUpdate();
}


void CastRays(const sensor::RangeData& range_data,
              const std::vector<uint16>& hit_table,
              const std::vector<uint16>& miss_table,
              const bool insert_free_space, ProbabilityGrid* probability_grid) {
  GrowAsNeeded(range_data, probability_grid);

  const MapLimits& limits = probability_grid->limits();
  const double superscaled_resolution = limits.resolution() / kSubpixelScale;
  const MapLimits superscaled_limits(
      superscaled_resolution, limits.max(),
      CellLimits(limits.cell_limits().num_x_cells * kSubpixelScale,
                 limits.cell_limits().num_y_cells * kSubpixelScale));
  const Eigen::Array2i begin =
      superscaled_limits.GetCellIndex(range_data.origin.head<2>());

  // Compute and add the end points.
  std::vector<Eigen::Array2i> ends;
  ends.reserve(range_data.returns.size());
  for (const sensor::RangefinderPoint& hit : range_data.returns) {
    ends.push_back(superscaled_limits.GetCellIndex(hit.position.head<2>()));
    probability_grid->ApplyLookupTable(ends.back() / kSubpixelScale, hit_table);
  }

  if (!insert_free_space) {
    return;
  }

  // Now add the misses.
  for (const Eigen::Array2i& end : ends) {
    std::vector<Eigen::Array2i> ray =
        RayToPixelMask(begin, end, kSubpixelScale);
    for (const Eigen::Array2i& cell_index : ray) {
      probability_grid->ApplyLookupTable(cell_index, miss_table);
    }
  }

  // Finally, compute and add empty rays based on misses in the range data.
  for (const sensor::RangefinderPoint& missing_echo : range_data.misses) {
    std::vector<Eigen::Array2i> ray = RayToPixelMask(
        begin, superscaled_limits.GetCellIndex(missing_echo.position.head<2>()),
        kSubpixelScale);
    for (const Eigen::Array2i& cell_index : ray) {
      probability_grid->ApplyLookupTable(cell_index, miss_table);
    }
  }
}
```

For each laser hit point, there generates a ray from the radiation center to the hit end point.

Then iterate all cells sitting in between such as `const Eigen::Array2i& cell_index : ray`, where `probability_grid->ApplyLookupTable(cell_index, miss_table);` is called to update cell's cost by looking the `miss_table`.
Here `*cell = table[*cell];` means cell's cost is adjusted by looking at the table.

```cpp
bool ApplyLookupTable(const Eigen::Array3i& index,
                        const std::vector<uint16>& table) {
  uint16* const cell = mutable_value(index);  // `mutable_value` returns the cell's cost
  update_indices_.push_back(cell);
  *cell = table[*cell];
  return true;
}

void FinishUpdate() {
  while (!update_indices_.empty()) {
    *update_indices_.back() -= kUpdateMarker; // kUpdateMarker = (cartographer::uint16)32768U
    update_indices_.pop_back();
  }
}
```

### Probability Grid

The cell in a probability grid is located by `const Eigen::Array2i& cell_index` (for 2d) setting to value `const float probability`.

To save storage space, cell probability uses `uint16` mapping $[1,32767]$ to $[b_{lower}, b_{upper}]$, where $0 \le b_{lower} \le b_{upper} \le 1$.

In fast mode, cell probability even uses `uint8` that only has $256=2^8$ resolutions. 

In the code below, `BoundedFloatToValue` and `SlowValueToBoundedFloat` are used to convert between `int` and `float` to represent probability.

A lookup table `kValueToCorrespondenceCost` is precomputed to do direct mapping between `uint16 value` and `float cost`.
Cost can be used to measure a cell's score if this cell meet a good scan match: $probability = 1.0 - cost$.



```cpp
void ProbabilityGrid::SetProbability(const Eigen::Array2i& cell_index,
                                     const float probability) {
    uint16& cell =
        (*mutable_correspondence_cost_cells())[ToFlatIndex(cell_index)];
    cell =
        CorrespondenceCostToValue(ProbabilityToCorrespondenceCost(probability));

    mutable_known_cells_box()->extend(cell_index.matrix());
}

inline uint16 BoundedFloatToValue(const float float_value,
                                  const float lower_bound,
                                  const float upper_bound) {
    const int value =
        common::RoundToInt(
          (common::Clamp(float_value, lower_bound, upper_bound) - lower_bound) *
          (32766.f / (upper_bound - lower_bound))) + 1;
  return value;
}

// Converts a uint16 (which may or may not have the update marker set) to a
// correspondence cost in the range [kMinCorrespondenceCost,
// kMaxCorrespondenceCost].
inline float ValueToCorrespondenceCost(const uint16 value) {
  return (*kValueToCorrespondenceCost)[value];
}

// 0 is unknown, [1, 32767] maps to [lower_bound, upper_bound].
float SlowValueToBoundedFloat(const uint16 value, const uint16 unknown_value,
                              const float unknown_result,
                              const float lower_bound,
                              const float upper_bound) {
  if (value == unknown_value) return unknown_result;
  const float kScale = (upper_bound - lower_bound) / (kValueCount - 2.f);
  return value * kScale + (lower_bound - kScale);
}
```

### Trajectory

```cpp
std::unique_ptr<LocalTrajectoryBuilder2D::MatchingResult>
LocalTrajectoryBuilder2D::AddRangeData(...)
{
    auto synchronized_data =
      range_data_collator_.AddRangeData(sensor_id, unsynchronized_data);

    // from other sensor such as IMU to get a pose estimate prior
    std::vector<transform::Rigid3f> range_data_poses;
    for (const auto& range : synchronized_data.ranges) {
        range_data_poses.push_back(
          extrapolator_->ExtrapolatePose(time_point).cast<float>());
    }

    // Drop any returns below the minimum range and convert returns beyond the
    // maximum range into misses.
    for (size_t i = 0; i < synchronized_data.ranges.size(); ++i) {
        const Eigen::Vector3f delta = hit_in_local.position - origin_in_local;
        const float range = delta.norm(); // `norm` is the 3d vector length
        if (range >= options_.min_range()) {
            if (range <= options_.max_range()) {
              accumulated_range_data_.returns.push_back(hit_in_local);
            } else {
              hit_in_local.position =
                  origin_in_local +
                  options_.missing_data_ray_length() / range * delta;
              accumulated_range_data_.misses.push_back(hit_in_local);
            }
        }
    }
    ++num_accumulated_;

    // when having accumulated enough data, add them by `AddAccumulatedRangeData`
    if (num_accumulated_ >= options_.num_accumulated_range_data()) {
        return AddAccumulatedRangeData(
          time,
          TransformToGravityAlignedFrameAndFilter(
              gravity_alignment.cast<float>() * range_data_poses.back().inverse(),
              accumulated_range_data_),
          gravity_alignment, sensor_duration);
    }
}

std::unique_ptr<LocalTrajectoryBuilder2D::MatchingResult>
LocalTrajectoryBuilder2D::AddAccumulatedRangeData(...)
{
    // Computes a gravity aligned pose prediction.
    const transform::Rigid3d non_gravity_aligned_pose_prediction =
        extrapolator_->ExtrapolatePose(time);
    const transform::Rigid2d pose_prediction = transform::Project2D(
        non_gravity_aligned_pose_prediction * gravity_alignment.inverse());

    // point cloud to voxel
    const sensor::PointCloud& filtered_gravity_aligned_point_cloud =
      sensor::AdaptiveVoxelFilter(gravity_aligned_range_data.returns,
                                  options_.adaptive_voxel_filter_options());

    // local map frame <- gravity-aligned frame
    std::unique_ptr<transform::Rigid2d> pose_estimate_2d =
            ScanMatch(time, pose_prediction, filtered_gravity_aligned_point_cloud);
    const transform::Rigid3d pose_estimate =
          transform::Embed3D(*pose_estimate_2d) * gravity_alignment;
      extrapolator_->AddPose(time, pose_estimate);

    // `InsertIntoSubmap` inserts range data to the current trajectory attached active submap 
    // ->`LocalTrajectoryBuilder2D::InsertIntoSubmap(...)`
    // ->`active_submaps_.InsertRangeData(range_data_in_local);` 
    // ->`submap->InsertRangeData(range_data, range_data_inserter_.get());`
    sensor::RangeData range_data_in_local =
        TransformRangeData(gravity_aligned_range_data,
                           transform::Embed3D(pose_estimate_2d->cast<float>()));
    std::unique_ptr<InsertionResult> insertion_result = InsertIntoSubmap(
        time, range_data_in_local, filtered_gravity_aligned_point_cloud,
        pose_estimate, gravity_alignment.rotation());

    return absl::make_unique<MatchingResult>(
      MatchingResult{time, pose_estimate, std::move(range_data_in_local),
                     std::move(insertion_result)});
}
```

`InsertionResult` and `MatchingResult` are part of trajectory.

```cpp
// Wires up the local SLAM stack (i.e. pose extrapolator, scan matching, etc.)
// without loop closure.
class LocalTrajectoryBuilder2D {
 public:
  struct InsertionResult {
    std::shared_ptr<const TrajectoryNode::Data> constant_data;
    std::vector<std::shared_ptr<const Submap2D>> insertion_submaps;
  };
  struct MatchingResult {
    common::Time time;
    transform::Rigid3d local_pose;
    sensor::RangeData range_data_in_local;
    // 'nullptr' if dropped by the motion filter.
    std::unique_ptr<const InsertionResult> insertion_result;
  };
  ...
};
```

Trajectory builder collectively computes range data and inserts the matching results into submaps.
It builds pose graph alongside taking range and other sensor data (e.g., IMU).

One trajectory is made up of one loop closure.

```cpp
int MapBuilderStub::AddTrajectoryBuilder(
    const std::set<SensorId>& expected_sensor_ids,
    const mapping::proto::TrajectoryBuilderOptions& trajectory_options,
    LocalSlamResultCallback local_slam_result_callback) {

    proto::AddTrajectoryRequest request;
    request.set_client_id(client_id_);
    *request.mutable_trajectory_builder_options() = trajectory_options;
    for (const auto& sensor_id : expected_sensor_ids) {
      *request.add_expected_sensor_ids() = cloud::ToProto(sensor_id);
    }

    async_grpc::Client<handlers::AddTrajectorySignature> client(
      client_channel_, common::FromSeconds(kChannelTimeoutSeconds),
      async_grpc::CreateLimitedBackoffStrategy(
          common::FromMilliseconds(kRetryBaseDelayMilliseconds),
          kRetryDelayFactor, kMaxRetries));
    CHECK(client.Write(request));

    // Construct trajectory builder stub.
    trajectory_builder_stubs_.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(client.response().trajectory_id()),
        std::forward_as_tuple(make_unique<TrajectoryBuilderStub>(
            client_channel_, client.response().trajectory_id(), client_id_,
            local_slam_result_callback)));
    return client.response().trajectory_id();

  }
```

```cpp
// An individual submap, which has a 'local_pose' in the local map frame, keeps
// track of how many range data were inserted into it, and sets
// 'insertion_finished' when the map no longer changes and is ready for loop
// closing.
class Submap {
public:
  Submap(const transform::Rigid3d& local_submap_pose)
      : local_pose_(local_submap_pose) {}

  bool insertion_finished() const { return insertion_finished_; }
  void set_insertion_finished(bool insertion_finished) {
    insertion_finished_ = insertion_finished;
  }

  void InsertRangeData(
    const sensor::RangeData& range_data,
    const RangeDataInserterInterface* range_data_inserter) {
      CHECK(!insertion_finished());
      range_data_inserter->Insert(range_data, grid_.get()); // further calls `castRay`
      set_num_range_data(num_range_data() + 1);
  }
};

// Once a certain number of range data have been inserted, the new submap is
// considered initialized: the old submap is no longer changed, the "new" submap
// is now the "old" submap and is used for scan-to-map matching. Moreover, a
// "new" submap gets created. The "old" submap is forgotten by this object.
class ActiveSubmaps2D {
    ...
};


std::vector<std::shared_ptr<const Submap2D>> ActiveSubmaps2D::InsertRangeData(
    const sensor::RangeData& range_data) {
  if (submaps_.empty() ||
      submaps_.back()->num_range_data() == options_.num_range_data()) {
    AddSubmap(range_data.origin.head<2>());
  }
  for (auto& submap : submaps_) {
    submap->InsertRangeData(range_data, range_data_inserter_.get());
  }
  if (submaps_.front()->num_range_data() == 2 * options_.num_range_data()) {
    submaps_.front()->Finish();
  }
  return submaps();
}
```

### Branch and Bound

Branch and Bound method uses a precomputed grid where each cell's score (`score = 1.f - cost`) is computed.

Recall that in a search window of size `wide_limits_.num_x_cells * wide_limits_.num_y_cells`, want to compute max scores in sub-windows: 
1. iterate all cells in a sub-window (sub-window partitioned by `stride`), and add them into `current_values`
2. find the max score/probability cell and store it in `intermediate`
3. remove cells from `current_values`

Sub-window cell iterations are first centered about the y-axis then about the x-axis.

Scores are served as bound that in branch and Bound method, children branches whose parent node's score is lower than the parent's counterpart node's score, will not grow.

```cpp
PrecomputationGrid2D::PrecomputationGrid2D(...)
{
    // intermediate is used to store sub-window max probability result
    const int stride = wide_limits_.num_x_cells;
    std::vector<float>& intermediate = *reusable_intermediate_grid;
    intermediate.resize(stride * limits.num_y_cells);

    // compute max probability x0 <= x < x0 + width centered around (x0, y)
    for (int y = 0; y != limits.num_y_cells; ++y) {
        SlidingWindowMaximum current_values;
        for (int x = -width + 1; x != 0; ++x) {
            current_values.AddValue(
              1.f - std::abs(grid.GetCorrespondenceCost(Eigen::Array2i(0, y))));
        }
        for (int x = 0; x < limits.num_x_cells - width; ++x) {
              intermediate[x + width - 1 + y * stride] = current_values.GetMaximum();
              current_values.RemoveValue(
                  1.f - std::abs(grid.GetCorrespondenceCost(Eigen::Array2i(x, y))));
              current_values.AddValue(1.f - std::abs(grid.GetCorrespondenceCost(
                                                Eigen::Array2i(x + width, y))));
        }    
    }

    // compute max probability y0 <= y < y0 + width centered around (x, y0)
    for (int x = 0; x != wide_limits_.num_x_cells; ++x) {
        SlidingWindowMaximum current_values;
        current_values.AddValue(intermediate[x]);
        for (int y = -width + 1; y != 0; ++y) {
            cells_[x + (y + width - 1) * stride] =
              ComputeCellValue(current_values.GetMaximum());
        }
        for (int y = 0; y < limits.num_y_cells - width; ++y) {
          cells_[x + (y + width - 1) * stride] =
              ComputeCellValue(current_values.GetMaximum());
          current_values.RemoveValue(intermediate[x + y * stride]);
          current_values.AddValue(intermediate[x + (y + width) * stride]);
        }
    }
    
}
```