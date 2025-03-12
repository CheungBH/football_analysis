# Use an official PyTorch image with CUDA 11.3
FROM nvcr.io/nvidia/pytorch:22.06-py3

# Set a working directory inside the container
WORKDIR /whole

ARG DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/Asia/Hong_Kong /etc/localtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    vim \
    wget \
    libgl1 \
    libglib2.0-0 \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libxcb-xinput0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-xfixes0 \
    libxcb-shm0 \
    libxcb-randr0 \
    libxcb-shape0 \
    libxcb-glx0 \
    software-properties-common \
    x11-apps \
    mesa-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

#RUN add-apt-repository universe && apt-get update

# Remove existing cv2 if it's installed from another source
RUN rm -rf /opt/conda/lib/python3.8/site-packages/cv2

# Copy the requirements into the container
COPY . .
#COPY assets/ /whole/assets/

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt
# EXPOSE 8088

# Set the default command to Python
# CMD ["python3", "yolov9_bytetrack_pth_4_cropping.py", "--video_path", "assets/sample", "--output_path", "output"]
