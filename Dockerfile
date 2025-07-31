FROM python:3.9-slim 
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/* 
RUN pip install opencv-python numpy mediapipe ffmpeg-python
COPY video_to_shorts.py /app/video_to_shorts.py 
WORKDIR /app 
ENTRYPOINT ["python", "video_to_shorts.py"]