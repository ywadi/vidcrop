#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
import tempfile

import cv2
import numpy as np
import ffmpeg
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# Scenedetect imports
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

# --- Configuration ---
# This is the name of the model file we downloaded in the Dockerfile.
YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"

# --- Logging Setup ---

def setup_logging():
    """Configures the logging format for the entire script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

# --- Core Tracking Logic (Kalman Filter + Hungarian Algorithm) ---

def iou(bb_test, bb_gt):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o

class KalmanBoxTracker:
    """
    This class represents the state of a single tracked object.
    """
    count = 0

    def __init__(self, bbox, q_vel_noise=0.01, q_ar_noise=0.01):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= q_ar_noise
        self.kf.Q[4:,4:] *= q_vel_noise
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.last_seen_bbox = bbox
        self.last_confidence = bbox[4]

    def convert_bbox_to_z(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h) if h != 0 else 0
        return np.array([x, y, s, r]).reshape((4, 1))

    def convert_x_to_bbox(self, x, score=None):
        w = np.sqrt(x[2] * x[3]) if x[2] * x[3] >= 0 else 0
        h = x[2] / w if w != 0 else 0
        if score is None:
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
        else:
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))
        self.last_seen_bbox = bbox
        self.last_confidence = bbox[4]

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

class Sort:
    """
    A simple online and realtime tracking algorithm.
    """
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3, q_vel_noise=0.01, q_ar_noise=0.01):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.q_vel_noise = q_vel_noise
        self.q_ar_noise = q_ar_noise
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], q_vel_noise=self.q_vel_noise, q_ar_noise=self.q_ar_noise)
            self.trackers.append(trk)
        
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if trk.time_since_update == 0:
                status = 1
                ret.append(np.concatenate((d, [trk.id + 1], [trk.last_confidence], [status])).reshape(1, -1))
            elif trk.hits >= self.min_hits:
                status = 2
                ret.append(np.concatenate((d, [trk.id + 1], [trk.last_confidence], [status])).reshape(1, -1))

            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 7))

    def associate_detections_to_trackers(self, detections, trackers):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = iou(det, trk)
        
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        matched_indices = []
        for r,c in zip(row_ind, col_ind):
            if iou_matrix[r,c] >= self.iou_threshold:
                matched_indices.append(np.array([r,c]).reshape(1,2))

        if len(matched_indices) > 0:
            matched_indices = np.concatenate(matched_indices, axis=0)
        else:
            matched_indices = np.empty((0,2),dtype=int)

        unmatched_detections = []
        for d, det in enumerate(detections):
            if(d not in matched_indices[:,0]):
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if(t not in matched_indices[:,1]):
                unmatched_trackers.append(t)
        
        return matched_indices, np.array(unmatched_detections), np.array(unmatched_trackers)

# --- Video and Scene Processing Utilities ---

def detect_scenes(video_path, threshold):
    logging.info(f"Starting scene detection with threshold: {threshold}...")
    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        scene_manager.detect_scenes(video=video, show_progress=True)
        scene_list = scene_manager.get_scene_list()
        
        if not scene_list:
            logging.warning("No scenes detected. Treating the whole video as a single scene.")
            total_frames = int(video.frame_rate * video.duration.get_seconds())
            return [(0, total_frames)]

        logging.info(f"Detected {len(scene_list)} scenes.")
        return [(s[0].get_frames(), s[1].get_frames()) for s in scene_list]

    except Exception as e:
        logging.error(f"Failed to process scenes: {e}")
        return None


def mux_audio(video_file, audio_source_file, output_file):
    logging.info("Muxing audio into final video...")
    try:
        input_video = ffmpeg.input(video_file)
        input_audio = ffmpeg.input(audio_source_file)
        ffmpeg.output(input_video.video, input_audio.audio, output_file, vcodec='copy', acodec='copy', loglevel="warning").run(overwrite_output=True)
        logging.info(f"Successfully created final output file: {output_file}")
    except ffmpeg.Error as e:
        logging.error("FFmpeg error during audio muxing:")
        if e.stderr:
            logging.error(e.stderr.decode())


# --- Main Processing Function ---

def run_detection(frame, detector):
    detections = []
    _, faces = detector.detect(frame)
    if faces is not None:
        if faces.shape[0] > 0:
            detections = np.array([f[:4].tolist() + [f[-1]] for f in faces])
            detections[:, 2] += detections[:, 0]
            detections[:, 3] += detections[:, 1]
    return detections

def merge_overlapping_tracks(buffer, total_frames, merge_iou_threshold):
    logging.info(f"Starting overlapping track merge pass with IoU threshold: {merge_iou_threshold}...")
    all_ids = sorted(list({obj[4] for frame_data in buffer.values() for obj in frame_data if len(frame_data) > 0}))
    if not all_ids:
        logging.info("No tracks to merge.")
        return buffer
        
    collision_pairs = set()
    for frame_idx in range(total_frames):
        if frame_idx not in buffer or len(buffer[frame_idx]) < 2:
            continue
        
        objects = buffer[frame_idx]
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                box_i, box_j = objects[i], objects[j]
                if iou(box_i, box_j) > merge_iou_threshold:
                    id_i, id_j = sorted((int(box_i[4]), int(box_j[4])))
                    if id_i != id_j:
                        collision_pairs.add((id_i, id_j))

    if not collision_pairs:
        logging.info("No overlapping tracks found to merge.")
        return buffer

    parent = {i: i for i in all_ids}
    def find_set(v):
        if v == parent[v]: return v
        parent[v] = find_set(parent[v])
        return parent[v]
    def unite_sets(a, b):
        a, b = find_set(a), find_set(b)
        if a != b: parent[max(a, b)] = min(a, b)

    for id_i, id_j in collision_pairs:
        unite_sets(id_i, id_j)

    id_map = {i: find_set(i) for i in parent}
    logging.info(f"Generated merge map: {id_map}")

    for frame_idx in range(total_frames):
        if frame_idx in buffer and len(buffer[frame_idx]) > 0:
            for obj in buffer[frame_idx]:
                obj[4] = id_map.get(obj[4], obj[4])

            seen_ids_in_frame = set()
            new_frame_data = []
            sorted_objects = sorted(buffer[frame_idx], key=lambda x: x[5], reverse=True)
            
            for obj in sorted_objects:
                if obj[4] not in seen_ids_in_frame:
                    new_frame_data.append(obj)
                    seen_ids_in_frame.add(obj[4])
            
            buffer[frame_idx] = np.array(new_frame_data) if new_frame_data else np.empty((0, 7))
            
    return buffer

def refine_track_boundaries(buffer, frame_buffer, detector, total_frames, detection_frames):
    logging.info("Starting track boundary refinement pass (Binary Search)...")
    all_ids = sorted(list({obj[4] for frame_data in buffer.values() for obj in frame_data if len(frame_data) > 0}))
    
    detection_frames_sorted = sorted(list(detection_frames))

    for track_id in all_ids:
        track_frames = [i for i, data in buffer.items() if any(obj[4] == track_id for obj in data)]
        if not track_frames: continue
        
        first_seen_frame = min(track_frames)
        last_seen_frame = max(track_frames)

        # --- Refine Start Frame ---
        prev_det_boundary = 0
        for det_frame in reversed(detection_frames_sorted):
            if det_frame < first_seen_frame:
                prev_det_boundary = det_frame
                break

        if first_seen_frame >= prev_det_boundary:
            first_instance = next(obj for obj in buffer[first_seen_frame] if obj[4] == track_id)
            low, high = prev_det_boundary, first_seen_frame - 1
            true_first = first_seen_frame
            best_det_found = None
            while low <= high:
                mid = (low + high) // 2
                frame = frame_buffer.get(mid)
                if frame is None:
                    low = mid + 1
                    continue
                detections = run_detection(frame, detector)
                matching_dets = [d for d in detections if iou(first_instance, d) > 0.4]
                if matching_dets:
                    true_first = mid
                    best_det_found = matching_dets[0]
                    high = mid - 1
                else:
                    low = mid + 1
            
            if true_first < first_seen_frame and best_det_found is not None:
                logging.info(f"Refined start for track ID {track_id} from {first_seen_frame} to {true_first}")
                
                dummy_tracker = KalmanBoxTracker(best_det_found)
                for i in range(true_first, first_seen_frame):
                    status = 1 if i == true_first else 2
                    pos = dummy_tracker.get_state()[0]
                    new_obj = np.concatenate((pos, [track_id], [best_det_found[4]], [status])).reshape(1, 7)
                    if i in buffer and buffer[i].size > 0:
                        buffer[i] = np.vstack([buffer[i], new_obj])
                    else:
                        buffer[i] = new_obj
                    dummy_tracker.predict()


        # --- Refine End Frame ---
        last_instance_in_buffer = next((obj for obj in buffer.get(last_seen_frame, []) if obj[4] == track_id), None)
        if last_instance_in_buffer is not None and last_instance_in_buffer[6] == 2:
            last_detection_frame = -1
            for i in range(last_seen_frame, -1, -1):
                if i in buffer:
                    instance = next((obj for obj in buffer[i] if obj[4] == track_id), None)
                    if instance is not None and instance[6] == 1:
                        last_detection_frame = i
                        break
            
            if last_detection_frame != -1:
                logging.info(f"Terminating predictions for track ID {track_id} after frame {last_detection_frame}.")
                for i in range(last_detection_frame + 1, last_seen_frame + 1):
                    if i in buffer:
                        updated_frame_data = np.array([obj for obj in buffer[i] if obj[4] != track_id])
                        buffer[i] = updated_frame_data if updated_frame_data.size > 0 else np.empty((0, 7))
                last_seen_frame = last_detection_frame

        if last_seen_frame < total_frames - 1:
            last_instance = next((obj for obj in buffer.get(last_seen_frame, []) if obj[4] == track_id), None)
            if last_instance is None: continue

            next_det_boundary = total_frames - 1
            for det_frame in detection_frames_sorted:
                if det_frame > last_seen_frame:
                    next_det_boundary = det_frame
                    break
            
            low, high = last_seen_frame + 1, next_det_boundary
            true_last = last_seen_frame
            best_det_found = None
            while low <= high:
                mid = (low + high) // 2
                frame = frame_buffer.get(mid)
                if frame is None:
                    high = mid - 1
                    continue
                detections = run_detection(frame, detector)
                matching_dets = [d for d in detections if iou(last_instance, d) > 0.4]
                if matching_dets:
                    true_last = mid
                    best_det_found = matching_dets[0]
                    low = mid + 1
                else:
                    high = mid - 1
            
            if true_last > last_seen_frame and best_det_found is not None:
                logging.info(f"Refined end for track ID {track_id} from {last_seen_frame} to {true_last}")
                for i in range(last_seen_frame + 1, true_last + 1):
                    new_obj = np.array([*best_det_found[:4], track_id, best_det_found[4], 1.0]).reshape(1, 7)
                    if i in buffer and buffer[i].size > 0:
                        buffer[i] = np.vstack([buffer[i], new_obj])
                    else:
                        buffer[i] = new_obj
    return buffer


def process_video(args):
    # 1. Initial Setup
    if not os.path.exists(YUNET_MODEL_PATH):
        logging.error(f"FATAL: YuNet model not found at '{YUNET_MODEL_PATH}'.")
        sys.exit(1)
    if not os.path.exists(args.input):
        logging.error(f"FATAL: Input file not found at '{args.input}'.")
        sys.exit(1)

    logging.info(f"Processing video: {args.input}")
    logging.info(f"Arguments: {vars(args)}")

    # 2. Scene Detection
    scenes = detect_scenes(args.input, args.scene_threshold)
    if scenes is None:
        logging.error("Could not perform scene detection. Aborting.")
        return
    scene_map = {}
    for i, (start, end) in enumerate(scenes):
        for frame_num in range(start, end):
            scene_map[frame_num] = i + 1

    # 3. Video I/O
    cap = cv2.VideoCapture(args.input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    face_detector = cv2.FaceDetectorYN.create(YUNET_MODEL_PATH, "", (width, height))
    face_detector.setScoreThreshold(args.conf_threshold)

    tracker = Sort(max_age=args.skip_frames, min_hits=3, iou_threshold=args.iou_threshold, q_vel_noise=args.q_vel_noise, q_ar_noise=args.q_ar_noise)
    
    tracking_buffer = {}
    frame_buffer = {} 

    temp_video_file = None
    out = None
    if args.debug:
        temp_fd, temp_video_file = tempfile.mkstemp(suffix=".mp4")
        os.close(temp_fd)
        logging.info(f"Debug mode enabled. Temporary video at: {temp_video_file}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_file, fourcc, fps, (width, height))

    # 4. Main Processing Loop
    logging.info("Buffering all frames into memory...")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_buffer[frame_idx] = frame
        frame_idx += 1
    actual_total_frames = len(frame_buffer)
    logging.info(f"Frame buffering complete. Actual frames read: {actual_total_frames}")

    logging.info("Starting tracking pass...")
    
    detection_frames = set()
    frame_idx = 0
    while frame_idx < actual_total_frames:
        if frame_idx % args.skip_frames == 0:
            for i in range(args.detection_burst):
                if frame_idx + i < actual_total_frames:
                    detection_frames.add(frame_idx + i)
        frame_idx += 1
    
    for frame_idx in range(actual_total_frames):
        frame = frame_buffer[frame_idx]
        detections = []
        if frame_idx in detection_frames:
            detections = run_detection(frame, face_detector)
        
        tracked_objects = tracker.update(detections)
        if len(tracked_objects) > 0:
            tracking_buffer[frame_idx] = tracked_objects

    logging.info("Initial tracking pass complete.")

    # 5. Refinement Passes - CORRECT ORDER
    tracking_buffer = merge_overlapping_tracks(tracking_buffer, actual_total_frames, args.merge_iou_threshold)
    tracking_buffer = refine_track_boundaries(tracking_buffer, frame_buffer, face_detector, actual_total_frames, detection_frames)
    logging.info("All refinement passes complete.")

    # 6. Debug Video Rendering
    if args.debug and out:
        logging.info("Rendering debug video...")
        for frame_idx in range(actual_total_frames):
            frame = frame_buffer.get(frame_idx)
            if frame is None: continue
            
            if frame_idx in tracking_buffer:
                objects_to_draw = tracking_buffer[frame_idx]
                for obj in objects_to_draw:
                    x1, y1, x2, y2, obj_id, conf, status = obj
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    color = (0, 255, 0) if status == 1 else (0, 255, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID: {int(obj_id)} C: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            current_scene = scene_map.get(frame_idx, 'N/A')
            scene_text = f"Scene: {current_scene}"
            cv2.putText(frame, scene_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            out.write(frame)
        
        out.release()
        logging.info("Debug video render complete.")

    # 7. Finalization
    cap.release()
    if args.debug and temp_video_file:
        mux_audio(temp_video_file, args.input, args.output)
        os.remove(temp_video_file)
        logging.info("Process complete.")
    else:
        logging.info("Process complete. No output file generated as --debug was not specified.")

# --- Entrypoint ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Face Detector with Tracking and Refinement")
    parser.add_argument('input', type=str, help="Path to the input video file.")
    parser.add_argument('output', type=str, help="Path to the output video file (only used with --debug).")
    parser.add_argument('--skip_frames', type=int, default=30, help="Number of frames to skip between detection bursts.")
    parser.add_argument('--detection_burst', type=int, default=5, help="Number of consecutive frames to run detection on during a burst.")
    parser.add_argument('--conf_threshold', type=float, default=0.85, help="Confidence threshold for face detection.")
    parser.add_argument('--scene_threshold', type=int, default=30, help="Threshold for scene change detection (lower is more sensitive).")
    parser.add_argument('--iou_threshold', type=float, default=0.3, help="Intersection over Union threshold for associating detections to trackers.")
    parser.add_argument('--merge_iou_threshold', type=float, default=0.85, help="Intersection over Union threshold for merging overlapping tracks in post-processing.")
    parser.add_argument('--q_vel_noise', type=float, default=0.01, help="Process noise for velocity in the Kalman filter. Higher values reduce drift.")
    parser.add_argument('--q_ar_noise', type=float, default=0.01, help="Process noise for aspect ratio in the Kalman filter.")
    parser.add_argument('--debug', action='store_true', help="If set, outputs a video with bounding boxes and tracking info.")
    
    args = parser.parse_args()
    
    setup_logging()
    
    process_video(args)
