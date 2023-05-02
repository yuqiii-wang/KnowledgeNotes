# Some Personal Projects

## IMU and Wheel Odometry Data Fusion by Kalman Filter

By the image frame retrieval timestamp (5 Hz), collect all IMU and wheel odometry data starting from the last image frame retrieval timestamp.
Then, use linear interpolation of rotation and translation as the model motion prediction results assigned to each IMU and wheel odometry reading.

## Hand Eye Calibration

The hand eye calibration problem estimates the transformation between a camera ("eye") mounted on a robot gripper ("hand").

It computes Hand-Eye calibration $\space^{g}T_{c}$ (`R_cam2gripper` and `t_cam2gripper`) given some $\space^{b}T_{g}$ (`R_gripper2base` and `t_gripper2base`) and $\space^{c}T_{t}$ (`R_target2cam` and `t_target2cam`).

There should be at least two transformations $i \ne j$ for both $i,j \ge 2$ to compute $\space^{g}T_{c}$.

<div style="display: flex; justify-content: center;">
      <img src="imgs/hand_eye_cali.png" width="40%" height="40%" alt="hand_eye_cali" />
</div>
</br>

```cpp
void cv::calibrateHandEye	(	InputArrayOfArrays 	R_gripper2base,
                            InputArrayOfArrays 	t_gripper2base,
                            InputArrayOfArrays 	R_target2cam,
                            InputArrayOfArrays 	t_target2cam,
                            OutputArray 	R_cam2gripper,
                            OutputArray 	t_cam2gripper,
                            HandEyeCalibrationMethod 	method = CALIB_HAND_EYE_TSAI 
                            )	
```

### Camera

To estimate the pose of camera, chessboard can be used to estimate $\space^{c}T_{t}$,
and IMU and odometry can be used to estimate $\space^{b}T_{g}$.
Having collected many $\space^{c}T_{t}$ and $\space^{b}T_{g}$, run `cv::calibrateHandEye(...)` to compute $\space^{g}T_{c}$.

### Lidar

By ICP, the lidar device movements can be computed $\space^{c}T_{t}$, then use IMU and odometry can be used to estimate $\space^{b}T_{g}$.
Finally by `cv::calibrateHandEye(...)` compute $\space^{g}T_{c}$.

## Lidar and ORB Feature Data Fusion

### YOLOv3 Human Body Detection and ORB Extraction and Tracking

YOLOv3 tiny model outputs human body detection rectangles.

Within the rectangles, search for ORB features.

```cpp
static Ptr<ORB> cv::ORB::create	(	int 	nfeatures = 500,
                                    float 	scaleFactor = 1.2f,
                                    int 	nlevels = 8,
                                    int 	edgeThreshold = 31,
                                    int 	firstLevel = 0,
                                    int 	WTA_K = 2,
                                    int 	scoreType = ORB::HARRIS_SCORE,
                                    int 	patchSize = 31,
                                    int 	fastThreshold = 20 
                                    )

// Detect ORB features and compute descriptors.
Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
orb->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
```

Optical flow tracking 

```cpp
void cv::calcOpticalFlowPyrLK	(	InputArray 	prevImg,
                                InputArray 	nextImg,
                                InputArray 	prevPts,
                                InputOutputArray 	nextPts,
                                OutputArray 	status,
                                OutputArray 	err,
                                Size 	winSize = Size(21, 21),
                                int 	maxLevel = 3,
                                TermCriteria 	criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                                int 	flags = 0,
                                double 	minEigThreshold = 1e-4 
                                )	

// calculate optical flow
Mat old_frame, old_gray;
vector<Point2f> p0, p1;
vector<uchar> status;
vector<float> err;
TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15,15), 2, criteria);
```

### Triangulation and K-Means Clustering to ORB Features

Given `cam0` and `cam1` that are $3 \times 4$ camera matrices (intrinsic and extrinsic parameters) for two camera poses, 
opencv provides the below method to triangulate two sets of feature points `cam0pnts` and `cam1pnts` with the results stored in `pnts3D`.
```cpp
cv::Mat pnts3D(1, N, CV_64FC4);
cv::Mat cam0pnts(1, N, CV_64FC2);
cv::Mat cam1pnts(1, N, CV_64FC2);

cv::triangulatePoints(cam0,cam1,cam0pnts,cam1pnts,pnts3D);
```

Having computed the 3d triangulated points, also compute the K-means clustering.
```cpp
double cv::kmeans	(	InputArray 	data,
                    int 	K,
                    InputOutputArray 	bestLabels,
                    TermCriteria 	criteria,
                    int 	attempts,
                    int 	flags,
                    OutputArray 	centers = noArray() 
                    )	

Mat points(clusterCount, 1, CV_32FC3), labels;
std::vector<Point3f> centers;
double compactness = kmeans(points, clusterCount, labels,
                        TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
                        3, KMEANS_PP_CENTERS, centers);
```

### ORB Feature Cluster Centroids as Initial Guesses to Lidar Point Cloud Segmentation

By the obtained bounding box from YOLOv3 as well as estimated cluster centroids, 
set up a projective filter whose opening angle is contained by the human body centroid and the YOLO predicted bounding box size.
Point clouds outside the projective filter are filtered out.

<div style="display: flex; justify-content: center;">
      <img src="imgs/point_cloud_filter_by_yolo_angle.png" width="40%" height="40%" alt="point_cloud_filter_by_yolo_angle" />
</div>
</br>

Then, perform segmentation on the filtered point clouds (random walk of 50 cm threshold), and by PCL point cloud bounding boxes to enclose the segmented point clouds (by picking the leftmost and rightmost points' x, and topmost and bottommost points' y).

### Data Fusion by Intersection of Union (IoU) Matching ORB Point Cloud and Lidar Point Cloud

By 2D laser points, the projected lidar bounding box would converge to a horizontal line against the YOLO bounding box that is also compressed to a horizontal line (by setting the $z$ to the height of the lidar device).

The two horizontal lines are matched by their closet centroids, and compared against their reprojection overlapping line percentage.

### Final Rectangle Enclosure Output

Output the fused point cloud enclosure rectangle/bounding box (most likely the bounding box of lidar point cloud since it is most stable).

If failed, only output visual ORB point cloud if they ORB features were successfully tracked (optical flow) in the last three/five frames, and the tracked OBR's triangulated 3d representations do not deviate a lot (compute covariance) in the last three/five frames.