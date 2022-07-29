# Iterative Closest Point (ICP)

Given some source points, with applied transformation (rotation and translation), algorithm iteratively revises the transformation to minimize an error metric (typically the sum of squared differences) between output point cloud and reference (regarded as ground truth) point cloud.