docker build -t video-to-shorts2 .
docker run --rm -v $(pwd):/data video-to-shorts2 /data/short3.mp4 /data/output3.mp4