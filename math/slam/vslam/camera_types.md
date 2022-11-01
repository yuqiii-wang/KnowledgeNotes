# Camera Types 

## Monocular camera

Mono camera computes depth info by selecting at least two key image frames from constant motion.

### Visual Feature Mapping

When the camera is monocular, we only know the 2D pixel coordinates, so the
problem is to estimate the motion according to two sets of 2D points. This
problem is solved by *epipolar geometry*.

### Pros and Cons

* Pros

Easy in computation and cheap

* Cons

Requiring camera constant movement/translation to find multiple key frames to detect disparity.

Scale Ambiguity: in figure (a), a camera moved forward and observed a pair of feature dots spanning two meter wide; in figure (b), a camera moved forward and observed an one-meter wide two dot feature pair. However, both scenarios see the same correspondance points on the monocular camera view, and camera itself cannot determine actual width of the feature pair, hence defined the scale ambiguity issue.

![monocular_scale_ambiguity](imgs/monocular_scale_ambiguity.png   "monocular_scale_ambiguity")

It is usually dealt with by adding other information such as sensors' input (IMU) or defining road as a plane.

## Binocular/Stereo Camera

Binocular camera uses triangulation from two physical camera positions to find object point depth.

![stereo_cam_depth](imgs/stereo_cam_depth.png "stereo_cam_depth")

The distance vector from object point $X$ to baseline $B$ is $z$, such that
$$
Bf=z(x-x')
$$

Hence, if we know the two optical center vector $B$, focus length (vector from camera view plane to optical center) $f$, two object correspondance points (left and right) on camera views relative to epipoles, we can compute the real world object point $X$ represented by $z$.

### Visual Feature Mapping

When the camera is binocular, RGB-D, or the distance is obtained by some
method, then the problem is to estimate the motion according to two sets of
3D points. This problem is usually solved by *ICP*.

### Pros and Cons

* Pros

Not dependant on other sensors' input.

* Cons

High computation cost; complicated calibration.

## RGB-D camera

RGB-D camera uses infrared light or Time-of-Flight (ToF) that a camera actively radiates light and accordingly receives light signals. It computes the elapsed time between two infrared light signals, by light speed, gives the results of depth/length to the camera.

### Visual Feature Mapping

Same as that of Stereo Camera, use ICP.

### Pros and Cons

* Pros

Low depth computation resources, since a depth camera gives depth information directly.

* Cons

Highly dependant on infrared results, that infrared might be prone to errors for strong ambient light disturbance, incapability/difficult of/to penetrating some materials (such as dust and mist) to give depth results of distant object.

## 3D and 2D Visual Feature Mapping

If one set is 3D and one set is 2D, we get some 3D points and their projection
positions on the camera, and we can also estimate the camera’s pose. This
problem is solved by PnP.