# ROS

Remember to call `source ./devel/setup.bash` before compilation or execution.

* ROS Error：-- Could NOT find PY_em (missing: PY_EM)

run with a builtin python env:
`catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3`

* Ceres solver install

First install dependencies
```bash
# CMake
sudo apt-get install cmake
# google-glog + gflags
sudo apt-get install libgoogle-glog-dev libgflags-dev
# Use ATLAS for BLAS & LAPACK
sudo apt-get install libatlas-base-dev
# Eigen3
sudo apt-get install libeigen3-dev
# SuiteSparse (optional)
sudo apt-get install libsuitesparse-dev
```

The version of ceres for VINS should be old; set git repo to an old commit
```bash
git clone https://ceres-solver.googlesource.com/ceres-solver

git checkout facb199f3eda902360f9e1d5271372b7e54febe1

mkdir build & cd build
cmake ..
make -j12
sudo make install
```

* OpenCV Install

By default, `apt-get` installs OpenCV 4.2; VINS uses version 3 (OpenCV version 4 has all macros `#define` replaced with `cv::`, hence version 3 required).

Run `sudo apt list |grep -i openCV` to check OpenCV version.

```bash
git clone https://github.com/opencv/opencv.git

# version 3.4
git checkout cb2052dbfef821f8fbdcc7ecf5780ab712d4e5dc
```