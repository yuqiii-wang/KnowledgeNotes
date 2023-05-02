# Point Cloud Library (PCL)

The Point Cloud Library (PCL) is a standalone, large scale, open project for 2D/3D image and point cloud processing. 

## Segmentation Example

```cpp
// ros msg to pcl msg setup
ros::sensor_msgs::PointCloud2::Ptr cloud_blob (new ros::sensor_msgs::PointCloud2), cloud_filtered_blob (new ros::sensor_msgs::PointCloud2);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>), cloud_p (new pcl::PointCloud<pcl::PointXYZ>);

reader.read ("table_scene_lms400.pcd", *cloud_blob);

// Create the filtering object: down-sample the dataset using a leaf size of 1cm
pcl::VoxelGrid<sensor_msgs::PointCloud2> sor;
sor.setInputCloud (cloud_blob);
sor.setLeafSize (0.01f, 0.01f, 0.01f);
sor.filter (*cloud_filtered_blob);

// Convert to the templated PointCloud
pcl::fromROSMsg (*cloud_filtered_blob, *cloud_filtered);

pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
// Create the segmentation object
pcl::SACSegmentation<pcl::PointXYZ> seg;
// Optional
seg.setOptimizeCoefficients (true);
// Mandatory
seg.setModelType (pcl::SACMODEL_PLANE);
seg.setMethodType (pcl::SAC_RANSAC);
seg.setMaxIterations (1000);
seg.setDistanceThreshold (0.01);

// Create the filtering object
pcl::ExtractIndices<pcl::PointXYZ> extract;

// Segment the largest planar component from the remaining cloud
seg.setInputCloud (cloud_filtered);
seg.segment (*inliers, *coefficients);

// Extract the inliers
extract.setInputCloud (cloud_filtered);
extract.setIndices (inliers);
extract.setNegative (false);
extract.filter (*cloud_p); // copy inliers to cloud_p

// Create the filtering object
// inlier points are removed from `*cloud_filtered`
extract.setNegative (true);
extract.filter (*cloud_filtered);
```

## Bounding Box

Bounding box attempts to find a minimum-perimeter enclosing rectangle over some point clouds.