# ./preprocessVideos.sh
docker build -t face-tracker .
rm output_short* -rf
docker run -v $(pwd):/app/videos face-tracker  /app/videos/x_short1.mp4 /app/videos/output_short1.mp4 --debug --conf_threshold 0.8 --scene_threshold 10 --skip_frames 15 --iou_threshold 0.1 --merge_iou_threshold 0.5 --q_ar_noise 0.5 --q_vel_noise 0.2
docker run -v $(pwd):/app/videos face-tracker /app/videos/x_short2.mp4 /app/videos/output_short2.mp4 --debug --conf_threshold 0.8 --scene_threshold 10 --skip_frames 15 --iou_threshold 0.1 --merge_iou_threshold 0.5 --q_ar_noise 0.5 --q_vel_noise 0.2
docker run -v $(pwd):/app/videos face-tracker /app/videos/x_short3.mp4 /app/videos/output_short3.mp4 --debug --conf_threshold 0.8 --scene_threshold 10 --skip_frames 15 --iou_threshold 0.1 --merge_iou_threshold 0.5 --q_ar_noise 0.5 --q_vel_noise 0.2
docker run -v $(pwd):/app/videos face-tracker /app/videos/x_short4.mp4 /app/videos/output_short4.mp4 --debug --conf_threshold 0.8 --scene_threshold 10 --skip_frames 15 --iou_threshold 0.1 --merge_iou_threshold 0.5 --q_ar_noise 0.5 --q_vel_noise 0.2
docker run -v $(pwd):/app/videos face-tracker /app/videos/x_short5.mp4 /app/videos/output_short5.mp4 --debug --conf_threshold 0.8 --scene_threshold 10 --skip_frames 15 --iou_threshold 0.1 --merge_iou_threshold 0.5 --q_ar_noise 0.5 --q_vel_noise 0.2
