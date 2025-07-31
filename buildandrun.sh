docker build -t video-to-shorts2 .
docker run --rm -v $(pwd):/data video-to-shorts2 /data/short1.mp4 /data/output1.mp4 --debug --inertia=0.5 --scene-cut-threshold=0.95
docker run --rm -v $(pwd):/data video-to-shorts2 /data/short2.mp4 /data/output2.mp4 --debug --inertia=0.5 --scene-cut-threshold=0.95
docker run --rm -v $(pwd):/data video-to-shorts2 /data/short3.mp4 /data/output3.mp4 --debug --inertia=0.5 --scene-cut-threshold=0.95
docker run --rm -v $(pwd):/data video-to-shorts2 /data/short4.mp4 /data/output4.mp4 --debug --inertia=0.5 --scene-cut-threshold=0.95
