# ROS

## Installation Guides

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
git switch 3.4
```

* `absl` not found

Use `abseil`

```bash
git clone https://github.com/abseil/abseil-cpp.git
```

* `lua`

```cpp
sudo apt-get install liblua5.3-dev
```
Comments: should have the suffix `-dev` to include libraries and header files.

## ROS API

A simple single-thread ros program looks like this.
```cpp
Foo foo_object;

ros::init(argc, argv, "my_node");
ros::NodeHandle nh;
ros::Subscriber sub = nh.subscribe("my_topic", 1, &Foo::callback, foo_object);
...
ros::spin();
```

`ros::Subscriber` registers listening to a topic and processes this topic data by handling `void(T::*)(M) 	fp`.
The actual handling happens in each `ros::spin();`. 
```cpp
template<class M , class T >
Subscriber ros::NodeHandle::subscribe (const std::string & 	topic,
                                uint32_t 	queue_size,
                                void(T::*)(M) 	fp,
                                T * 	obj,
                                const TransportHints & 	transport_hints = TransportHints() 
                                );
```

Topic data is generated via `advertise`.
```cpp
template<class M >
Publisher ros::NodeHandle::advertise (const std::string & 	topic,
                                uint32_t 	queue_size,
                                bool 	latch = false 
                                );
```

### Topics vs Services vs Actions

* Topic:

continuous data streams (sensor data, robot state, …)

* Service:

a simple blocking call, used to execute short functions such as querying specific data

For example, in server code (assumed already defined ros service request and response `beginner_tutorials::AddTwoInts::Request` and `beginner_tutorials::AddTwoInts::Response`)
```cpp
bool add(beginner_tutorials::AddTwoInts::Request  &req,
         beginner_tutorials::AddTwoInts::Response &res)
{
  res.sum = req.a + req.b;
  ROS_INFO("request: x=%ld, y=%ld", (long int)req.a, (long int)req.b);
  ROS_INFO("sending back response: [%ld]", (long int)res.sum);
  return true;
}

ros::ServiceServer service = n.advertiseService("add_two_ints", add);
```

In client code
```cpp
ros::NodeHandle n;
ros::ServiceClient client = n.serviceClient<beginner_tutorials::AddTwoInts>("add_two_ints");

beginner_tutorials::AddTwoInts srv;
srv.request.a = atoll(argv[1]);
srv.request.b = atoll(argv[2]);
if (client.call(srv))
{
  ROS_INFO("Sum: %ld", (long int)srv.response.sum);
}
```

* Action

Long execution function.

Can be preempted and preemption should always be implemented cleanly by action servers.

For example, in server code
```cpp
actionlib::SimpleActionClient<actionlib_tutorials::FibonacciAction> ac("fibonacci", true);

ac.waitForServer(); //will wait for tserver to star

actionlib_tutorials::FibonacciGoal goal;
goal.order = 20;
ac.sendGoal(goal);

//wait for the action to return, set 30 sec rimeout
bool finished_before_timeout = ac.waitForResult(ros::Duration(30.0));
```