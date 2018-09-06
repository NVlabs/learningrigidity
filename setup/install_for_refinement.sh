# system dependencies
sudo apt-get install libboost-all-dev cmake libtbb-dev libopencv-dev

# compile and install gtsam as dependency for the refinment module
git clone --depth 1 -b develop git@bitbucket.org:gtborg/gtsam.git external_packages/gtsam
cd external_packages/gtsam
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../../../tmp/
make install -j8

# compile the refinement module
cd ../../external_packages/flow2pose
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../../../tmp/
make -j8
