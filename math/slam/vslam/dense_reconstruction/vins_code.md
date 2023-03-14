# VINS SLAM Source Code Review

## Compile and Launch

### Error `[pose_graph-4] process has died` and `[feature_tracker-2] process has died`

Error `[pose_graph-4] process has died` and `[feature_tracker-2] process has died` may occur after running `roslaunch vins_estimator euroc.launch`.
The root cause is `cv_bridge` having a wrong version.

`cv_bridge` is used to convert between ros images and OpenCV images.

First, prepare `OpenCV 3.4` by `git checkout 3.4`, and compile the 3.4 version OpenCV.

In Ubuntu20.04, go to `/opt/ros/noetic/share/cv_bridge/cmake/` and edit the config `sudo gedit cv_bridgeConfig.cmake`, changing the relevant OpenCV uses to using 3.4 version.
In detail, there are

```bash
# if(NOT "include;/usr/include/opencv4 " STREQUAL " ")
if(NOT "include;/home/yuqi/code/code/opencv/include/opencv" STREQUAL " ")
...

#  set(_include_dirs "include;/usr/include/opencv4")
   set(_include_dirs "include;/home/yuqi/code/code/opencv/include/opencv")
...

# set(libraries "cv_bridge;/usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_video.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_face.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_text.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_core.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.2.0;/usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.2.0")
set(libraries "cv_bridge;/home/yuqi/code/code/opencv/build/lib/libopencv_calib3d.so.3.4;/home/yuqi/code/code/opencv/build/lib/libopencv_imgproc.so.3.4;/home/yuqi/code/code/opencv/build/lib/libopencv_core.so.3.4;/home/yuqi/code/code/opencv/build/lib/libopencv_highgui.so.3.4;")
```

## Overall Structure

Start from `estimator_node`, VINS first registers publishing/listening messages, then processes IMU and image data in a loop.


```cpp
int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");

    registerPub(n);

    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    std::thread measurement_process{process};
    ros::spin();
}


void registerPub(ros::NodeHandle &n)
{
    pub_latest_odometry = n.advertise<nav_msgs::Odometry>("imu_propagate", 1000);
    pub_path = n.advertise<nav_msgs::Path>("path", 1000);
    pub_relo_path = n.advertise<nav_msgs::Path>("relocalization_path", 1000);
    pub_odometry = n.advertise<nav_msgs::Odometry>("odometry", 1000);
    pub_point_cloud = n.advertise<sensor_msgs::PointCloud>("point_cloud", 1000);
    pub_margin_cloud = n.advertise<sensor_msgs::PointCloud>("history_cloud", 1000);
    pub_key_poses = n.advertise<visualization_msgs::Marker>("key_poses", 1000);
    pub_camera_pose = n.advertise<nav_msgs::Odometry>("camera_pose", 1000);
    pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 1000);
    pub_keyframe_pose = n.advertise<nav_msgs::Odometry>("keyframe_pose", 1000);
    pub_keyframe_point = n.advertise<sensor_msgs::PointCloud>("keyframe_point", 1000);
    pub_extrinsic = n.advertise<nav_msgs::Odometry>("extrinsic", 1000);
    pub_relo_relative_pose=  n.advertise<nav_msgs::Odometry>("relo_relative_pose", 1000);

    cameraposevisual.setScale(1);
    cameraposevisual.setLineWidth(0.05);
    keyframebasevisual.setScale(0.1);
    keyframebasevisual.setLineWidth(0.01);
}


// thread: visual-inertial odometry
void process()
{
    while (true)
    {
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        for (auto &measurement : measurements)
        {    
            // measurement.first is the IMU data
            for (auto &imu_msg : measurement.first)
            {
                dx = imu_msg->linear_acceleration.x;
                dy = imu_msg->linear_acceleration.y;
                dz = imu_msg->linear_acceleration.z;
                rx = imu_msg->angular_velocity.x;
                ry = imu_msg->angular_velocity.y;
                rz = imu_msg->angular_velocity.z;
                estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
            }

            // measurement.second is the Image feature point data
            auto img_msg = measurement.second;
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }
            estimator.processImage(image, img_msg->header);

            // after loading and processing data, publish them
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";
            pubOdometry(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubPointCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
        }
    }
}
```


* The main work in `processIMU` is about pre-integration taking into account of IMU data to a corresponding keyframe interval.
```cpp
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    ...
    pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
    ...
}
```

* The main work in `processImage`

In previous `process`, image feature points are  
```cpp
auto img_msg = measurement.second;
for (auto& point : img_msg) {
    xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
    image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
}
estimator.processImage(image, img_msg->header);
```

```cpp
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header)
{
    
}
```

## Optimization

```cpp
void Estimator::optimization()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
}
```