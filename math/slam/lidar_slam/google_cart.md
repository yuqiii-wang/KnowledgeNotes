# Lidar Mapping Google Cartographer

## Local Mapping

### Scans

Laser scans are recorded and are transformed by a static homogeneous transformation matrix to a robot's origin.

Scan point set: $H=\{h_k\}_{k=1,2,3,...,K}, h_k \in R^2$

The pose $\xi$ of the scan frame in the
submap frame transformation: $T_{\xi}$

### Submaps

A few consecutive scans are used to build a submap. 
These submaps take the form of probability grids $M : rZ × rZ \rightarrow [p_{min} , p_{max} ]$ which map from discrete grid points at a given resolution $r$, for example 5 cm, to values. $Z$ is the height/width of the submap image.

The probability represents a grid cell $M(x)$ is empty or obstructed. If $M(x)$ has not yet been observed, it is assigned $p_{hit}$ or $p_{miss}$; if $M(x)$ is observed, it can be updated.

$$
odds(p)=\frac{p}{1-p}
$$

$$
M_{new}(x) = clamp\big(\frac{1}{odds(odds(M_{old}(x)) \cdot odds(p_{hit}))}\big)
$$
where
$$
clamp(x) = max(a, min(x, b)) \in [a, b]
$$

### Scan Matching

The scan matcher is responsible for
finding a scan pose $\xi$ that maximizes the probabilities at the scan points in the submap.

$$
arg \space \underset{\xi}{min} \sum_{k=1}^n \big(1-M_{smooth}(T_\xi h_k)\big)^2
$$
where $T_\xi$ transforms $h_k$ from the scan frame to the submap
frame according to the scan pose.

$M_{smooth}: R^2 \rightarrow R$ by nature is Bicubic Interpolation whose output is in the range $(0, 1)$ (when near to one, the matching result is considered good). It is used to "smooth" the generated grid map.

## Closing Loops

### Closing Loop Optimization

Local closing loop problem refers to optimizing submap poses $\Xi^m = \{\xi^m_0, \xi^m_1, \xi^m_2, ..., \xi^m_k\}$ and scan poses $\Xi^s = \{\xi^s_0, \xi^s_1, \xi^s_2, ..., \xi^s_k\}$ over a route.

A closed route means that a robot returns to the place where it starts travelling; the robot should travel as much as possible to visit all places of an environment, such as a closed door museum, school.

Closing loop optimization:

$$
arg\space \underset{\Xi^m， \Xi^s}{min} \frac{1}{2} \underset{i,j}{\sum} p \big( E^2(\xi^m_i, \xi^s_j; \Sigma_{i,j}, \xi_{i,j}) \big)
$$
where constraints take the form of relative poses $\xi_{i,j}$ and associated covariance matrices $\Sigma_{i,j}$, for input pair $\xi^m_i, \xi^s_j$

In detail,
$$
E^2(\xi^m_i, \xi^s_j; \Sigma_{i,j}, \xi_{i,j}) 
\\ =
e(\xi^m_i, \xi^s_j; \xi_{i,j})^T \Sigma_{i,j} e(\xi^m_i, \xi^s_j; \xi_{i,j})
$$
in which, $e(\xi^m_i, \xi^s_j; \xi_{i,j})$ describes the error of robot one step pose against its scans and generated submap.

*Huber Loss* is the loss function $p$.

### Branch-and-bound scan matching

In order to retreive optimal scan match, we want to fill a map with 
$$
\xi^* = arg\space \underset{\xi \in W}{max} \sum_{k=1}^K M_{nearest}(T_{\xi}h_k)
$$
where, $\xi \in W$ means employment of a discrete search window:

1. find the longest scan by
$$
d_{max}=\underset{k=1,2,...,K}{max} ||h_k||
$$
2. calculate angular step $\delta_\theta$
$$
\delta_\theta = arccos(1-\frac{r^2}{2d_{max}^2})
$$
3. compute search window translational move step with pre-defined max window size (such as $7$ m) $W_x, W_y$, and max window rotation (such as $30^\circ$ ) $W_\theta$
$$
w_x=\frac{W_x}{r}, w_y=\frac{W_y}{r}, w_\theta=\frac{W_\theta}{\delta_\theta}
$$

Iterating all possible windows can be time-consuming, here introduces *branch-and-bound*.

The goal of a *branch-and-bound* algorithm is to find a value $x$ that maximizes or minimizes the value of a real-valued function $f(x)$, called an objective function, among some set $S$ of admissible, or candidate solutions. The set $S$ is called the search space, or feasible region. 

* branch: splits the search space into smaller spaces, then minimizing $f(x)$ on these smaller spaces;
* bound: set lower and upper bounds of regions/branches of the search space. If no bounds are available, the algorithm degenerates to an exhaustive search.

In practice of scan matching, branching heappens when a candidate solution is not a leaf node (there is sub solution search space) and not hit bound (the split node represents a promissing solution search space).