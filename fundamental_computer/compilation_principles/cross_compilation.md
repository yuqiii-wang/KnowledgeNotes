# Cross Compilation on x86 and Arm

Cross compilation means， for example, compiling code on x86 but actually expecting the code being run on arm.

A cross-compiler is a compiler that runs on one platform/architecture but generates code for another platform/architecture.

In practice, should download cross cimpiling extension package.
In CMAKE, such extension packages are set as the C/CXX compilor.
```cmake
set(tools /home/devel/gcc-4.7-linaro-rpi-gnueabihf)
set(CMAKE_C_COMPILER ${tools}/bin/arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER ${tools}/bin/arm-linux-gnueabihf-g++)
```