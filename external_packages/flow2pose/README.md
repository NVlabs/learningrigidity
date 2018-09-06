# Flow2Pose Lib

The flow2pose library is an algorithm that takes 3D ego-motion flow fields as input and estimate the pose transformation. 

## Dependencies

* gtsam
* opencv3

## Install

Use the following command manually

```
mkdir build
cd build
cmake ..
make -j8
```

Or use '../../setup/install_for_refinement.sh' for an all-in-one install.