# Use a complete and supported base image
FROM python:3.9-bullseye

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by OpenCV, ffmpeg, and wget
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip and install the Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download the YuNet model from the official OpenCV GitHub repository
RUN wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# Copy the face tracker script into the container
COPY face_tracker.py .

# The entrypoint for the container. This will run the face tracker script.
ENTRYPOINT ["python", "face_tracker.py"]
