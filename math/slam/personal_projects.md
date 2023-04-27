# Some Personal Projects

## IMU and Wheel Odometry Data Fusion

## Lidar Device Extrinsic Alignment

## Hand Eye Calibration

Visual features are extracted by ORB.

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
orb->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);

// Match features.
std::vector<DMatch> matches;
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
matcher->match(descriptors1, descriptors2, matches, Mat());

// Sort matches by score
std::sort(matches.begin(), matches.end());

// Remove not so good matches
const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
matches.erase(matches.begin()+numGoodMatches, matches.end());

// Extract location of good matches
std::vector<Point2f> points1, points2;

for( size_t i = 0; i < matches.size(); i++ )
{
  points1.push_back( keypoints1[ matches[i].queryIdx ].pt );
  points2.push_back( keypoints2[ matches[i].trainIdx ].pt );
}
```

Having obtained `keypoints1` and `keypoints2` from two camera shots, perform triangulation.

## Lidar and ORB Feature Data Fusion

### YOLOv3 Human Body Detection and ORB Extraction

### K-Means Clustering to ORB Features

### Cluster Centroids as Initial Guesses to Lidar Point Cloud Segmentation

### Enclosure of Lidar Point Cloud and Projection to 2D Image

### Data Fusion by Intersection of Union (IoU) Matching ORB Point Cloud and Lidar Point Cloud

Assign more weights to Lidar point cloud.

Penalty to Enclosure size (to a human being body size) that should not exceed $1$ meter in width.

Secondary weight by covariance of 

### Final Rectangle Enclosure Output

Output the fused point cloud enclosure rectangle.

If failed, only output visual ORB point cloud with enclosure rectangle.