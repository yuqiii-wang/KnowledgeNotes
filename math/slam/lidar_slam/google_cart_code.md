# Google Cartographer  Code

## GRPC

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

## The Master Node Map Server 

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

`RangeData` is used to record how rays are radiated from one `origin` (just a 3d pose) and ended at `returns` (of a `PointCloud` type used to record many reflected laser rays, and where these rays are reflected are treated as the poses of points) when hitting obstacles.
`PointCloud` is composed of `std::vector<PointType> points_;` and its corresponding point intensities `std::vector<float> intensities_;`.
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

// Stores 3D positions of points together with some additional data, e.g.
// intensities.
class PointCloud {
public:
  using PointType = RangefinderPoint;

  PointCloud();
  explicit PointCloud(std::vector<PointType> points);
  PointCloud(std::vector<PointType> points, std::vector<float> intensities);

  ...

private:
  // For 2D points, the third entry is 0.f.
  std::vector<PointType> points_;
  // Intensities are optional. If non-empty, they must have the same size as
  // points.
  std::vector<float> intensities_;
};
```


## Trajectory

Trajectory first receives range data by `range_data_collator_` that sorts each laser ray by (linear approximated) timestamp.
Then for each laser ray, add a corresponding pose by `extrapolator_->ExtrapolatePose(...)`, where extrapolated poses can be from IMU or odometry.

`(hit_in_local.position - origin_in_local).norm()` computes the 3d Euclidean distance of a point from its origin, and will be discarded for distance being too short or too long.

Finally, range data is added by `AddAccumulatedRangeData`, where `ScanMatch` takes the current active submap, in which pose optimization is performed (will be discussed later).
The ceres optimization residual is recorded for performance measurement.

Also in the final stage `AddAccumulatedRangeData`, with the obtained estimated pose from `ScanMatch`, 
the range data as well as the pose are inserted into the current active submap by `InsertIntoSubmap(...)` that returns the result of insertion `std::unique_ptr<InsertionResult> insertion_result`.

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

std::unique_ptr<transform::Rigid2d> LocalTrajectoryBuilder2D::ScanMatch(...)
{
    std::shared_ptr<const Submap2D> matching_submap =
      active_submaps_.submaps().front();

    if (options_.use_online_correlative_scan_matching()) {
    const double score = real_time_correlative_scan_matcher_.Match(...);
    } 

    auto pose_observation = absl::make_unique<transform::Rigid2d>();
    ceres::Solver::Summary summary;
    ceres_scan_matcher_.Match(pose_prediction.translation(), initial_ceres_pose,
                              filtered_gravity_aligned_point_cloud,
                              *matching_submap->grid(), pose_observation.get(),
                              &summary);

    ...

    return pose_observation;
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

In the above `ScanMatch`, there is an option that fast produces a matching score.
This serves as precomputed scores for `PrecomputationGrid2D` that is used in branch-and-bound method during loop closure.
```cpp
const double score = real_time_correlative_scan_matcher_.Match(...)
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

`Submap` takes some range data and forms a local map.
One submap has one local pose `local_pose_`, which is `transform::Rigid3d local_pose = transform::ToRigid3(response.local_pose());` then `local_slam_result_callback(..., local_pose, range_data, std::move(insertion_result));` in `void TrajectoryBuilderStub::RunLocalSlamResultsReader(...)` reading client's data `client->StreamRead(&response)` containing range and pose results.
Remember that `local_slam_result_callback(...)` is invoked when accumulated enough range data.

An active submap continues receiving range data and performs pose optimization based on scan match.

A finished submap produces a probability grid `grid_ = grid_->ComputeCroppedGrid();`

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


void Submap2D::Finish() {
  CHECK(grid_);
  CHECK(!insertion_finished());
  grid_ = grid_->ComputeCroppedGrid();
  set_insertion_finished(true);
}
```


## Scan Match and Pose Optimization

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

### Fast Scan Match and Score Computation for Precomputed Probability Grid

`GenerateExhaustiveSearchCandidates(search_parameters);` takes config from manually set parameters to see by what steps and bound to generate some scan candidates.

Then `ScoreCandidates(grid, discrete_scans, search_parameters, &candidates);` is executed to  get the score of a candidate.

The scoring mechanism is basically 

```c++
double RealTimeCorrelativeScanMatcher2D::Match(...)
{
    ...

    std::vector<Candidate2D> candidates =
      GenerateExhaustiveSearchCandidates(search_parameters);
    ScoreCandidates(grid, discrete_scans, search_parameters, &candidates);

    const Candidate2D& best_candidate =
        *std::max_element(candidates.begin(), candidates.end());
    *pose_estimate = transform::Rigid2d(
        {initial_pose_estimate.translation().x() + best_candidate.x,
         initial_pose_estimate.translation().y() + best_candidate.y},
        initial_rotation * Eigen::Rotation2Dd(best_candidate.orientation));
    return best_candidate.score;
}


std::vector<Candidate2D>
RealTimeCorrelativeScanMatcher2D::GenerateExhaustiveSearchCandidates(...)
{
  std::vector<Candidate2D> candidates;
  candidates.reserve(num_candidates);
  for (int scan_index = 0; scan_index != search_parameters.num_scans;
       ++scan_index) {
    for (int x_index_offset = search_parameters.linear_bounds[scan_index].min_x;
         x_index_offset <= search_parameters.linear_bounds[scan_index].max_x;
         ++x_index_offset) {
      for (int y_index_offset =
               search_parameters.linear_bounds[scan_index].min_y;
           y_index_offset <= search_parameters.linear_bounds[scan_index].max_y;
           ++y_index_offset) {
        candidates.emplace_back(scan_index, x_index_offset, y_index_offset,
                                search_parameters);
      }
    }
  }
  CHECK_EQ(candidates.size(), num_candidates);
  return candidates;
}

void RealTimeCorrelativeScanMatcher2D::ScoreCandidates(...)
{
    candidate.score = ComputeCandidateScore(
            static_cast<const ProbabilityGrid&>(grid),
            discrete_scans[candidate.scan_index], candidate.x_index_offset,
            candidate.y_index_offset);
    candidate.score *=
        std::exp(-common::Pow2(std::hypot(candidate.x, candidate.y) *
                                   options_.translation_delta_cost_weight() +
                               std::abs(candidate.orientation) *
                                   options_.rotation_delta_cost_weight()));
}

float ComputeCandidateScore(...) 
{
  float candidate_score = 0.f;
  for (const Eigen::Array2i& xy_index : discrete_scan) {
    const Eigen::Array2i proposed_xy_index(xy_index.x() + x_index_offset,
                                           xy_index.y() + y_index_offset);
    const float probability =
        probability_grid.GetProbability(proposed_xy_index);
    candidate_score += probability;
  }
  candidate_score /= static_cast<float>(discrete_scan.size());
  CHECK_GT(candidate_score, 0.f);
  return candidate_score;
}
```

## Laser Ray Process

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

## Probability Grid

The probability grid is first init with cells' probability set to some manually config values, such as 
```lua
probability_grid_range_data_inserter = {
  insert_free_space = true,
  hit_probability = 0.55,
  miss_probability = 0.49,
}
```

Then the hit and miss probabilities are transformed to correspondence cost as well as some look-up table init.

```cpp
ProbabilityGridRangeDataInserter2D::ProbabilityGridRangeDataInserter2D(
    const proto::ProbabilityGridRangeDataInserterOptions2D& options)
    : options_(options),
      hit_table_(ComputeLookupTableToApplyCorrespondenceCostOdds(
          Odds(options.hit_probability()))),
      miss_table_(ComputeLookupTableToApplyCorrespondenceCostOdds(
          Odds(options.miss_probability()))) {}

// Sets the probability of the cell at 'cell_index' to the given
// 'probability'. Only allowed if the cell was unknown before.
void ProbabilityGrid::SetProbability(const Eigen::Array2i& cell_index,
                                     const float probability) {
    uint16& cell =
        (*mutable_correspondence_cost_cells())[ToFlatIndex(cell_index)];
    cell =
        CorrespondenceCostToValue(ProbabilityToCorrespondenceCost(probability));

    mutable_known_cells_box()->extend(cell_index.matrix());
}
```


The probability update is in accordance to 
$$
M_{new}(x) = \text{clamp}\Big(
    \frac{1}{
    \text{odds} \big( \text{odds}\big(M_{old}(x)\big) \cdot \text{odds}(p_{hit}) \big)}
\Big)
$$

```cpp
inline float Odds(float probability) {
  return probability / (1.f - probability);
}

// Clamps 'value' to be in the range ['min', 'max'].
template <typename T>
T Clamp(const T value, const T min, const T max) {
  if (value > max) {
    return max;
  }
  if (value < min) {
    return min;
  }
  return value;
}
```

The cell in a probability grid is located by `const Eigen::Array2i& cell_index` (for 2d) setting to value `const float probability`.

To save storage space, cell probability uses `uint16` mapping $[1,32767]$ to $[b_{lower}, b_{upper}]$, where $0 \le b_{lower} \le b_{upper} \le 1$.

In fast mode, cell probability even uses `uint8` that only has $256=2^8$ resolutions. 

In the code below, `BoundedFloatToValue` and `SlowValueToBoundedFloat` are used to convert between `int` and `float` to represent probability.

A lookup table `kValueToCorrespondenceCost` is precomputed to do direct mapping between `uint16 value` and `float cost`.
Cost can be used to measure a cell's score if this cell meet a good scan match: $probability = 1.0 - cost$.


```cpp
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

### Probability Grid Construction

A probability takes a resolution defining a cell size and a `conversion_tables`.

A `conversion_tables` is a lookup table for mapping from a uint16 value to a float/probability in $[$ `lower_bound`, `upper_bound` $]$ (the conversion is simply for storage convenience that uint16 occupies less space than float).

```cpp
/* ------------------- ProbabilityGrid ------------------*/

mapping::ProbabilityGrid CreateProbabilityGrid(
    const double resolution,
    mapping::ValueConversionTables* conversion_tables) {
  constexpr int kInitialProbabilityGridSize = 100;
  Eigen::Vector2d max =
      0.5 * kInitialProbabilityGridSize * resolution * Eigen::Vector2d::Ones();
  return mapping::ProbabilityGrid(
      mapping::MapLimits(resolution, max,
                         mapping::CellLimits(kInitialProbabilityGridSize,
                                             kInitialProbabilityGridSize)),
      conversion_tables);
}

// Returns the probability of the cell with 'cell_index'.
float ProbabilityGrid::GetProbability(const Eigen::Array2i& cell_index) const {
  if (!limits().Contains(cell_index)) return kMinProbability;
  return CorrespondenceCostToProbability(ValueToCorrespondenceCost(
      correspondence_cost_cells()[ToFlatIndex(cell_index)]));
}

ProbabilityGrid::ProbabilityGrid(const MapLimits& limits,
                                 ValueConversionTables* conversion_tables)
    : Grid2D(limits, kMinCorrespondenceCost, kMaxCorrespondenceCost,
             conversion_tables),
      conversion_tables_(conversion_tables) {}


std::unique_ptr<Grid2D> ProbabilityGrid::ComputeCroppedGrid() const {
  Eigen::Array2i offset;
  CellLimits cell_limits;
  ComputeCroppedLimits(&offset, &cell_limits);
  const double resolution = limits().resolution();
  const Eigen::Vector2d max =
      limits().max() - resolution * Eigen::Vector2d(offset.y(), offset.x());
  std::unique_ptr<ProbabilityGrid> cropped_grid =
      absl::make_unique<ProbabilityGrid>(
          MapLimits(resolution, max, cell_limits), conversion_tables_);
  for (const Eigen::Array2i& xy_index : XYIndexRangeIterator(cell_limits)) {
    if (!IsKnown(xy_index + offset)) continue;
    cropped_grid->SetProbability(xy_index, GetProbability(xy_index + offset));
  }

  return std::unique_ptr<Grid2D>(cropped_grid.release());
}

/* ------------------- ValueConversionTables ------------------*/

// Performs lazy computations of lookup tables for mapping from a uint16 value
// to a float in ['lower_bound', 'upper_bound']. The first element of the table
// is set to 'unknown_result'.
class ValueConversionTables {
 public:
  const std::vector<float>* GetConversionTable(float unknown_result,
                                               float lower_bound,
                                               float upper_bound);

 private:
  std::map<const std::tuple<float /* unknown_result */, float /* lower_bound */,
                            float /* upper_bound */>,
           std::unique_ptr<const std::vector<float>>>
      bounds_to_lookup_table_;
};

const std::vector<float>* ValueConversionTables::GetConversionTable(
    float unknown_result, float lower_bound, float upper_bound) {
  std::tuple<float, float, float> bounds =
      std::make_tuple(unknown_result, lower_bound, upper_bound);
  auto lookup_table_iterator = bounds_to_lookup_table_.find(bounds);
  if (lookup_table_iterator == bounds_to_lookup_table_.end()) {
    auto insertion_result = bounds_to_lookup_table_.emplace(
        bounds, PrecomputeValueToBoundedFloat(0, unknown_result, lower_bound,
                                              upper_bound));
    return insertion_result.first->second.get();
  } else {
    return lookup_table_iterator->second.get();
  }
}
```


## Branch and Bound

Branch and Bound is used in loop closure to fine-tune the pose graph.
Basically, just launch a candidate pose search window containing many candidate poses;
then find the candidate pose with the highest score.

The score is precomputed by the below function and stored for submaps.
```cpp
const double score = real_time_correlative_scan_matcher_.Match(...)
```

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

The actual scoring function that takes `const PrecomputationGrid2D& precomputation_grid` and sums the score is
```cpp
void FastCorrelativeScanMatcher2D::ScoreCandidates(...)
{
 for (Candidate2D& candidate : *candidates) {
    int sum = 0;
    for (const Eigen::Array2i& xy_index :
         discrete_scans[candidate.scan_index]) {
      const Eigen::Array2i proposed_xy_index(
          xy_index.x() + candidate.x_index_offset,
          xy_index.y() + candidate.y_index_offset);

      // Returns a value between 0 and 255 to represent probabilities between
      // min_score and max_score.
      sum += precomputation_grid.GetValue(proposed_xy_index);     
    }
    // Maps values from [0, 255] to [min_score, max_score].
    candidate.score = precomputation_grid.ToScore(
        sum / static_cast<float>(discrete_scans[candidate.scan_index].size()));   

  }
  std::sort(candidates->begin(), candidates->end(),
            std::greater<Candidate2D>());
}
```