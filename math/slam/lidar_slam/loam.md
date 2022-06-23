# LOAM: Lidar Odometry and Mapping

## Iterative Closest Point (ICP)

Given some source points, with applied transformation (rotation and translation), algorithm iteratively revises the transformation to minimize an error metric (typically the sum of squared differences) between output point cloud and reference (regarded as ground truth) point cloud.

## Lidar Odometry

Define every scan sweep as $k, k \in Z^+$.

Define Lidar coordinate system in which each point $i, i \in P_k$ is part of point cloud $P_k$, $i$ in this coordinate is denoted as $\bold{X}^L_{(k,i)}$

Accordingly, in the world coordinate, $i$ is denoted as $\bold{X}^W_{(k,i)}$

Let $S$ be the set of consecutive points of $i$ returned by the laser scanner in
the same scan.

### Point Cloud Feature Extraction

This step is to find planes and edges of scanned obstacles.

Define a term $c$ to evaluate the smoothness of local surface
$$
c=\frac{1}{|S|\cdot||\bold{X}^L_{(k,i)}||}
\big|\big|
\sum_{j \in S, j\ne i} (\bold{X}^L_{(k,i)}-\bold{X}^L_{(k,j)})
\big|\big|
$$
where large/max values of $c$ indicate edges, small/minimum values of $c$ indicate planes.

### Feature Point Correspondance

Let $t_k$ be the starting time of a sweep $k$. By the end of the sweep, received point cloud $P_k$ is reprojected (here denoted as $\overline{P}_k$) to timestamp $t_{k+1}$.

Let $\varepsilon_k$ and $H_k$ be the sets of edge points
and planar points, respectively.

The difference between $P_k$ and $\overline{P}_k$ is that during this sweep $k$, robot is moving. Robot's movement should be corrected to 