# Start with NVIDIA L4T PyTorch base image
FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libdc1394-22-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Special handling for OpenCV on Jetson
RUN apt-get update && apt-get install -y python3-opencv
RUN pip3 uninstall -y opencv-python
RUN pip3 install opencv-python-headless

# Install Open3D from source for ARM architecture
RUN git clone --depth 1 https://github.com/intel-isl/Open3D && \
    cd Open3D && \
    util/install_deps_ubuntu.sh && \
    mkdir build && \
    cd build && \
    cmake -DBUILD_CUDA_MODULE=ON \
          -DBUILD_SHARED_LIBS=ON \
          -DGLIBCXX_USE_CXX11_ABI=ON \
          -DPYTHON_EXECUTABLE=$(which python3) \
          .. && \
    make -j$(nproc) && \
    make install && \
    make install-pip-package && \
    cd ../.. && \
    rm -rf Open3D

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:0

# Command to run the application
CMD ["python3", "main.py"] 