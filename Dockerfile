FROM nvidia/cuda:8.0-devel-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    cmake \
    libboost-all-dev \
    libtbb-dev \
    libopencv-dev \
    && sudo rm -rf /var/lib/apt/lists/*
 
# Create a working directory
RUN mkdir /rigidity
WORKDIR /rigidity

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN mkdir /home/user
RUN chmod 777 /home/user

COPY ./setup /rigidity/setup

# Install Miniconda
 ENV CONDA_AUTO_UPDATE_CONDA=false

RUN curl https://repo.anaconda.com/miniconda/Miniconda2-4.5.4-Linux-x86_64.sh -o ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc 


# Create a Python 2.7 environment with all the required packages
RUN . /opt/conda/etc/profile.d/conda.sh \
    && conda env create -f ./setup/rigidity.yml \
    && echo "conda activate rigidity" >> ~/.bashrc \
    && conda activate rigidity

# COPY External Packages
COPY ./external_packages /rigidity/external_packages

# Build GTSAM
RUN cd /rigidity/external_packages/gtsam \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make install -j8 \
    && cd ../.. 

# Build correlation layer
RUN . /opt/conda/etc/profile.d/conda.sh \
    && conda activate rigidity \
    && cd /rigidity/external_packages/correlation-pytorch-master/correlation-pytorch/correlation_package/src \
    && nvcc -c -o corr_cuda_kernel.cu.o corr_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52 \
    && nvcc -c -o corr1d_cuda_kernel.cu.o corr1d_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52 \
    && cd ../../ \
    && python setup.py build install

# compile the refinement module
RUN cd /rigidity/external_packages/flow2pose \
    && mkdir build \
    && cd build \
    && cmake .. \
         -DPYTHON_INCLUDE_DIR:PATH=/opt/conda/envs/rigidity/include/python2.7 \
         -DPYTHON_LIBRARY:PATH=/opt/conda/envs/rigidity/lib/python2.7 \
         -DPYTHON_EXECUTABLE:FILEPATH=/opt/conda/envs/rigidity/bin/python \
    && make install -j8 \
    && cp pyFlow2Pose.so /opt/conda/envs/rigidity/lib/python2.7/site-packages \
    && cp libpyboostcv_bridge.so /usr/local/lib

RUN rm -rf /rigidity

# Set the default cmd`
CMD ["/bin/bash"]