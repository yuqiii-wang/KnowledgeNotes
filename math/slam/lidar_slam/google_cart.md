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

From `main()` starts `ros`
```cpp
int main() {
    ...
    ::ros::init(argc, argv, "cartographer_node"); // ros node: cartographer_node
    ::ros::start();

    cartographer_ros::ScopedRosLogSink ros_log_sink;
    cartographer_ros::Run(); // actual execution
    ::ros::shutdown();
}
```

From `Run`
```cpp
void Run() {
  constexpr double kTfBufferCacheTimeInSeconds = 10.;
  tf2_ros::Buffer tf_buffer{::ros::Duration(kTfBufferCacheTimeInSeconds)};
  tf2_ros::TransformListener tf(tf_buffer);//subscribes to a appropriate topics to receive the transformation.
  NodeOptions node_options;//NodeOptions : ros node about tf publish time, etc.
  TrajectoryOptions trajectory_options; 
  std::tie(node_options, trajectory_options) =
      LoadOptions(FLAGS_configuration_directory, FLAGS_configuration_basename);// config assignment to node_options, trajectory_options

  auto map_builder =
      cartographer::common::make_unique<cartographer::mapping::MapBuilder>(
          node_options.map_builder_options);// cartographer::common::make_unique定义在common文件夹下的make_unique.h文件中。
  Node node(node_options, std::move(map_builder), &tf_buffer);// node construction: defined many topics to subscribe
  if (!FLAGS_load_state_filename.empty()) {
    node.LoadState(FLAGS_load_state_filename, FLAGS_load_frozen_state);  // load ros bag
  }

  if (FLAGS_start_trajectory_with_default_topics) {
    node.StartTrajectoryWithDefaultTopics(trajectory_options);
  }

  ::ros::spin();// spin handles ros topic data

  node.FinishAllTrajectories();
  node.RunFinalOptimization();

  if (!FLAGS_save_state_filename.empty()) {
    node.SerializeState(FLAGS_save_state_filename);
  }
}
```
Topics:
* `kSubmapListTopic`:
* `kTrajectoryNodeListTopic`
* `kLandmarkPosesListTopic`
* `kConstraintListTopic`
* `kScanMatchedPointCloudTopic`

Services:
* `kSubmapQueryServiceName`

```cpp
for (const auto& texture_proto : response_proto.textures()) {
  response.textures.emplace_back();
  auto& texture = response.textures.back();
  texture.cells.insert(texture.cells.begin(), texture_proto.cells().begin(),
                       texture_proto.cells().end());
  texture.width = texture_proto.width();
  texture.height = texture_proto.height();
  texture.resolution = texture_proto.resolution();
  texture.slice_pose = ToGeometryMsgPose(
      cartographer::transform::ToRigid3(texture_proto.slice_pose()));
}
response.status.message = "Success.";
response.status.code = cartographer_ros_msgs::StatusCode::OK;
```

* `kStartTrajectoryServiceName`

One trajectory means one closed loop operations. 
For robust mapping, can allow a robot repeat the existing closed loop route traveling.

```cpp
response.trajectory_id = AddTrajectory(options, request.topics);
response.status.code = cartographer_ros_msgs::StatusCode::OK;
response.status.message = "Success.";
```

In `AddTrajectory`, extrapolator is used to estimate a robot's pose, typically from IMU.
`LaunchSubscribers` has many sensor publisher info, depending on hardware (config in lua files), subscribes sensor topics.
```cpp
int Node::AddTrajectory(const TrajectoryOptions& options,
                        const cartographer_ros_msgs::SensorTopics& topics) {
  const std::set<cartographer::mapping::TrajectoryBuilderInterface::SensorId>
      expected_sensor_ids = ComputeExpectedSensorIds(options, topics);
  const int trajectory_id =
      map_builder_bridge_.AddTrajectory(expected_sensor_ids, options);
  AddExtrapolator(trajectory_id, options);
  AddSensorSamplers(trajectory_id, options);
  LaunchSubscribers(options, topics, trajectory_id);
  is_active_trajectory_[trajectory_id] = true;
  for (const auto& sensor_id : expected_sensor_ids) {
    subscribed_topics_.insert(sensor_id.id);
  }
  return trajectory_id;
}
```

* `kFinishTrajectoryServiceName`

Checks if trajectory work is finished. First check some statuses, then do some trajectory-finishing clean up work
```cpp
// some err checking
if (...)
  status_response.code = cartographer_ros_msgs::StatusCode::INVALID_ARGUMENT;
else if (...)
  status_response.code = cartographer_ros_msgs::StatusCode::NOT_FOUND;
else if (...)
  status_response.code = cartographer_ros_msgs::StatusCode::RESOURCE_EXHAUSTED;

// shutdown topics
for (auto& entry : subscribers_[trajectory_id]) {
    entry.subscriber.shutdown();
    subscribed_topics_.erase(entry.topic);
}

// set trajectory finish status
map_builder_bridge_.FinishTrajectory(trajectory_id);
is_active_trajectory_[trajectory_id] = false;
status_response.code = cartographer_ros_msgs::StatusCode::OK;
```

* `kWriteStateServiceName`

Used to write map data to a file
```cpp
map_builder_bridge_.SerializeState(request.filename);
response.status.code = cartographer_ros_msgs::StatusCode::OK;
response.status.message = "State written to '" + request.filename + "'.";
```

```cpp
Node::Node(
    const NodeOptions& node_options,
    std::unique_ptr<cartographer::mapping::MapBuilderInterface> map_builder,
    tf2_ros::Buffer* const tf_buffer)
    : node_options_(node_options),
      map_builder_bridge_(node_options_, std::move(map_builder), tf_buffer) 
{
  carto::common::MutexLocker lock(&mutex_);
  submap_list_publisher_ =
      node_handle_.advertise<::cartographer_ros_msgs::SubmapList>(
          kSubmapListTopic, kLatestOnlyPublisherQueueSize);
  trajectory_node_list_publisher_ =
      node_handle_.advertise<::visualization_msgs::MarkerArray>(
          kTrajectoryNodeListTopic, kLatestOnlyPublisherQueueSize);
  landmark_poses_list_publisher_ =
      node_handle_.advertise<::visualization_msgs::MarkerArray>(
          kLandmarkPosesListTopic, kLatestOnlyPublisherQueueSize);
  constraint_list_publisher_ =
      node_handle_.advertise<::visualization_msgs::MarkerArray>(
          kConstraintListTopic, kLatestOnlyPublisherQueueSize);


  service_servers_.push_back(node_handle_.advertiseService(
      kSubmapQueryServiceName, &Node::HandleSubmapQuery, this));
  service_servers_.push_back(node_handle_.advertiseService(
      kStartTrajectoryServiceName, &Node::HandleStartTrajectory, this));
  service_servers_.push_back(node_handle_.advertiseService(
      kFinishTrajectoryServiceName, &Node::HandleFinishTrajectory, this));
  service_servers_.push_back(node_handle_.advertiseService(
      kWriteStateServiceName, &Node::HandleWriteState, this));

  scan_matched_point_cloud_publisher_ =
      node_handle_.advertise<sensor_msgs::PointCloud2>(
          kScanMatchedPointCloudTopic, kLatestOnlyPublisherQueueSize); 

  wall_timers_.push_back(node_handle_.createWallTimer(
      ::ros::WallDuration(node_options_.submap_publish_period_sec),
      &Node::PublishSubmapList, this));
  wall_timers_.push_back(node_handle_.createWallTimer(
      ::ros::WallDuration(node_options_.pose_publish_period_sec),
      &Node::PublishTrajectoryStates, this));
  wall_timers_.push_back(node_handle_.createWallTimer(
      ::ros::WallDuration(node_options_.trajectory_publish_period_sec),
      &Node::PublishTrajectoryNodeList, this));
  wall_timers_.push_back(node_handle_.createWallTimer(
      ::ros::WallDuration(node_options_.trajectory_publish_period_sec),
      &Node::PublishLandmarkPosesList, this));
  wall_timers_.push_back(node_handle_.createWallTimer(
      ::ros::WallDuration(kConstraintPublishPeriodSec),
      &Node::PublishConstraintList, this));
}
```