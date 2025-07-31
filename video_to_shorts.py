import cv2
import numpy as np
import mediapipe as mp
import ffmpeg
import sys
import argparse
import logging
import time

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoProcessor:
    """
    A class to handle the core logic of detecting faces and cropping a video
    to a 9:16 aspect ratio, with smoothed, stabilized camera motion and glitch-free layout changes.
    """

    def __init__(self, input_path, output_path, smoothing_factor=0.07, inertia_zone=0.7, enter_threshold=5, exit_threshold=15, debug_mode=False, upscale_factor=1.0, confidence=0.5, scene_cut_threshold=0.99):
        """
        Initializes the VideoProcessor.

        Args:
            input_path (str): Path to the input video file.
            output_path (str): Path to save the output video file.
            smoothing_factor (float): Factor for smoothing camera motion (0-1). Lower is smoother.
            inertia_zone (float): Percentage of the frame to use as a 'dead zone' for movement (0-1).
            enter_threshold (int): Frames to confirm a new mode before switching.
            exit_threshold (int): Frames to wait before exiting a mode after faces are lost.
            debug_mode (bool): If True, draws bounding boxes around detected faces.
            upscale_factor (float): Factor to upscale frames before detection (e.g., 1.5).
            confidence (float): Minimum detection confidence for MediaPipe.
            scene_cut_threshold (float): Correlation threshold for scene cut detection (0.0-1.0). Lower is more sensitive.
        """
        if not input_path or not output_path:
            raise ValueError("Input and output paths cannot be None.")
            
        self.input_path = input_path
        self.output_path = output_path
        self.smoothing_factor = smoothing_factor
        self.inertia_zone = inertia_zone
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.debug_mode = debug_mode
        self.upscale_factor = upscale_factor
        self.scene_cut_threshold = scene_cut_threshold
        
        # --- Video Properties ---
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise IOError(f"Error opening video file: {input_path}")
            
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # --- Output Configuration ---
        self.output_aspect_ratio = 9 / 16
        self.output_height = 1920
        self.output_width = int(self.output_height * self.output_aspect_ratio)

        # --- MediaPipe Initialization ---
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=confidence
        )

        # --- State for Logic ---
        self.active_mode = 'general'
        self.candidate_mode = 'general'
        self.mode_confirmation_counter = 0
        self.last_face_count = 0
        self.last_general_crop_box = None
        self.last_top_crop_box = None
        self.last_bottom_crop_box = None
        self.prev_frame_hist = None # For scene cut detection

        logging.info("VideoProcessor initialized.")
        logging.info(f"Smoothing: {self.smoothing_factor}, Inertia: {self.inertia_zone}, Confidence: {confidence}")
        if self.scene_cut_threshold < 1.0:
            logging.info(f"Scene cut detection enabled with threshold: {self.scene_cut_threshold}")
        if self.upscale_factor > 1.0:
            logging.info(f"Detection frame upscale factor: {self.upscale_factor}")
        if self.debug_mode:
            logging.warning("DEBUG MODE ENABLED: Bounding boxes will be drawn on the output video.")

    def _detect_scene_cut(self, frame, frame_number):
        """Detects a scene cut by comparing color histograms of consecutive frames."""
        if frame_number == 0:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            self.prev_frame_hist = cv2.calcHist([hsv_frame], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(self.prev_frame_hist, self.prev_frame_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            return False

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        current_hist = cv2.calcHist([hsv_frame], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(current_hist, current_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        correlation = cv2.compareHist(self.prev_frame_hist, current_hist, cv2.HISTCMP_CORREL)
        self.prev_frame_hist = current_hist

        return correlation < self.scene_cut_threshold

    def _is_bbox_in_crop(self, bbox, crop_box):
        """Checks if the center of a bounding box is inside a crop box."""
        if not bbox or not crop_box:
            return False
        
        face_cx = bbox[0] + bbox[2] / 2
        face_cy = bbox[1] + bbox[3] / 2
        
        crop_cx, crop_cy, crop_w, crop_h = crop_box
        crop_xmin = crop_cx - crop_w / 2
        crop_xmax = crop_cx + crop_w / 2
        crop_ymin = crop_cy - crop_h / 2
        crop_ymax = crop_cy + crop_h / 2
        
        return (crop_xmin < face_cx < crop_xmax) and \
               (crop_ymin < face_cy < crop_ymax)

    def _get_face_bboxes(self, detections, source_width, source_height):
        """Helper to get absolute pixel bounding boxes from detection results."""
        if not detections: return []
        bboxes = []
        for d in detections:
            box = d.location_data.relative_bounding_box
            x = int(box.xmin * source_width)
            y = int(box.ymin * source_height)
            w = int(box.width * source_width)
            h = int(box.height * source_height)
            bboxes.append((x, y, w, h))
        return bboxes
    
    def _create_non_stretched_background(self, frame):
        """Creates a blurred background that covers the output dimensions without stretching."""
        # Calculate scale factor, using the larger of the two to ensure coverage (overflow)
        scale_w = self.output_width / self.original_width
        scale_h = self.output_height / self.original_height
        scale = max(scale_w, scale_h)

        # New dimensions of the scaled frame
        scaled_w = int(self.original_width * scale)
        scaled_h = int(self.original_height * scale)

        # Resize the frame preserving aspect ratio
        resized_frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

        # Calculate coordinates for a center crop
        x_offset = (scaled_w - self.output_width) // 2
        y_offset = (scaled_h - self.output_height) // 2

        # Perform the center crop to get the final background size
        background = resized_frame[y_offset:y_offset + self.output_height, x_offset:x_offset + self.output_width]
        
        return background

    def _calculate_target_crop(self, bboxes):
        """Calculates the ideal target crop box for the current frame based on face bboxes."""
        if not bboxes:
            return (self.original_width / 2, self.original_height / 2, self.original_width, self.original_height)

        if len(bboxes) == 1:
            x, y, w, h = bboxes[0]
            face_cx, face_cy = x + w / 2, y + h / 2
            crop_h = self.original_height
            crop_w = int(crop_h * self.output_aspect_ratio)
            if crop_w > self.original_width:
                crop_w = self.original_width
                crop_h = int(crop_w / self.output_aspect_ratio)
            return (face_cx, face_cy, crop_w, crop_h)

        all_x = [b[0] for b in bboxes] + [b[0] + b[2] for b in bboxes]
        all_y = [b[1] for b in bboxes] + [b[1] + b[3] for b in bboxes]
        
        faces_xmin, faces_xmax = min(all_x), max(all_x)
        faces_ymin, faces_ymax = min(all_y), max(all_y)
        
        cx, cy = (faces_xmin + faces_xmax) / 2, (faces_ymin + faces_ymax) / 2
        w, h = faces_xmax - faces_xmin, faces_ymax - faces_ymin

        if h == 0 or w == 0: return self._calculate_target_crop(None)

        if w / h > self.output_aspect_ratio:
            final_h = w / self.output_aspect_ratio
            final_w = w
        else:
            final_w = h * self.output_aspect_ratio
            final_h = h

        return (cx, cy, final_w * 1.4, final_h * 1.4)

    def _apply_smoothing_and_inertia(self, target_box, last_box_state):
        """Applies inertia and EMA smoothing to a crop box."""
        if last_box_state is None:
            return target_box

        last_cx, last_cy, last_w, last_h = last_box_state
        target_cx, target_cy, target_w, target_h = target_box
        
        dead_zone_w = last_w * self.inertia_zone
        dead_zone_h = last_h * self.inertia_zone
        
        x_in_zone = abs(target_cx - last_cx) < dead_zone_w / 2
        y_in_zone = abs(target_cy - last_cy) < dead_zone_h / 2

        if x_in_zone and y_in_zone:
            return last_box_state
        
        alpha = self.smoothing_factor
        smooth_cx = alpha * target_cx + (1 - alpha) * last_cx
        smooth_cy = alpha * target_cy + (1 - alpha) * last_cy
        smooth_w = alpha * target_w + (1 - alpha) * last_w
        smooth_h = alpha * target_h + (1 - alpha) * last_h
        
        return (smooth_cx, smooth_cy, smooth_w, smooth_h)

    def _place_sub_frame(self, main_frame, sub_frame, region_box):
        """Places a sub-frame into a region, resizing to fit with aspect ratio preserved."""
        if sub_frame is None or sub_frame.shape[0] == 0 or sub_frame.shape[1] == 0: return
        rx, ry, rw, rh = region_box
        sh, sw, _ = sub_frame.shape

        scale = min(rw / sw, rh / sh)
        new_w, new_h = int(sw * scale), int(sh * scale)
        if new_w <= 0 or new_h <= 0: return
            
        resized_sub = cv2.resize(sub_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x_offset = rx + (rw - new_w) // 2
        y_offset = ry + (rh - new_h) // 2
        main_frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_sub

    def _get_crop_from_box(self, frame, box):
        """Safely crops the frame using a (cx, cy, w, h) box."""
        if box is None: return None
        cx, cy, w, h = box
        x_min = max(0, int(cx - w / 2))
        y_min = max(0, int(cy - h / 2))
        x_max = min(self.original_width, int(cx + w / 2))
        y_max = min(self.original_height, int(cy + h / 2))
        if x_max <= x_min or y_max <= y_min: return None
        return frame[y_min:y_max, x_min:x_max]

    def _calculate_zoomed_out_face_target(self, bbox):
        """Calculates a WIDE, zoomed-out crop target for a single face."""
        x, y, w, h = bbox
        half_screen_ar = (self.output_width) / (self.output_height / 2)
        
        padding_multiplier = 3.0 
        
        if w / h > half_screen_ar:
            crop_w = w * padding_multiplier
            crop_h = crop_w / half_screen_ar
        else:
            crop_h = h * padding_multiplier
            crop_w = crop_h * half_screen_ar
        
        return (x + w / 2, y + h / 2, crop_w, crop_h)

    def process_video(self):
        """Main processing loop."""
        logging.info("Starting video processing...")
        start_time = time.time()
        
        input_audio = ffmpeg.input(self.input_path).audio
        process = (
            ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{self.output_width}x{self.output_height}', r=self.fps)
            .output(input_audio, self.output_path, pix_fmt='yuv420p', vcodec='libx264', acodec='copy')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        frame_count = 0
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success: break

            # --- Pre-processing for Detection ---
            frame_for_detection = frame
            detection_width, detection_height = self.original_width, self.original_height
            if self.upscale_factor > 1.0:
                d_w = int(self.original_width * self.upscale_factor)
                d_h = int(self.original_height * self.upscale_factor)
                frame_for_detection = cv2.resize(frame, (d_w, d_h), interpolation=cv2.INTER_CUBIC)
                sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                frame_for_detection = cv2.filter2D(frame_for_detection, -1, sharpen_kernel)
                detection_width, detection_height = d_w, d_h

            # --- Face Detection ---
            rgb_frame = cv2.cvtColor(frame_for_detection, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            bboxes = self._get_face_bboxes(results.detections, detection_width, detection_height)
            if self.upscale_factor > 1.0:
                bboxes = [(int(x / self.upscale_factor), int(y / self.upscale_factor), 
                           int(w / self.upscale_factor), int(h / self.upscale_factor)) 
                          for x, y, w, h in bboxes]

            if self.debug_mode:
                for (x, y, w, h) in bboxes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

            # --- Scene Cut Detection and Mode Logic ---
            is_scene_cut = self._detect_scene_cut(frame, frame_count) if self.scene_cut_threshold < 1.0 else False
            detected_mode = '2_face' if len(bboxes) == 2 else 'general'

            if is_scene_cut:
                # Check if we should override the reset
                should_override_reset = False
                if len(bboxes) == self.last_face_count:
                    if len(bboxes) == 1 and self._is_bbox_in_crop(bboxes[0], self.last_general_crop_box):
                        should_override_reset = True
                    elif len(bboxes) == 2:
                        sorted_bboxes = sorted(bboxes, key=lambda b: b[0])
                        if self._is_bbox_in_crop(sorted_bboxes[0], self.last_top_crop_box) and \
                           self._is_bbox_in_crop(sorted_bboxes[1], self.last_bottom_crop_box):
                            should_override_reset = True
                
                if not should_override_reset:
                    logging.info(f"SCENE CUT DETECTED at frame {frame_count}. Resetting camera and forcing mode to '{detected_mode}'.")
                    self.active_mode = detected_mode
                    self.candidate_mode = detected_mode
                    self.mode_confirmation_counter = 1
                    self.last_general_crop_box = None
                    self.last_top_crop_box = None
                    self.last_bottom_crop_box = None
                else:
                    logging.info(f"Scene cut detected at frame {frame_count}, but overriding reset as subjects are stable.")
            
            # Standard temporal filtering for mode changes within a scene
            if not is_scene_cut:
                if detected_mode == self.candidate_mode:
                    self.mode_confirmation_counter += 1
                else:
                    self.candidate_mode = detected_mode
                    self.mode_confirmation_counter = 1

                threshold = self.exit_threshold if self.active_mode == '2_face' else self.enter_threshold
                if self.candidate_mode != self.active_mode and self.mode_confirmation_counter >= threshold:
                    logging.info(f"CONFIRMED mode change from '{self.active_mode}' to '{self.candidate_mode}' at frame {frame_count}")
                    self.active_mode = self.candidate_mode
                    self.last_general_crop_box = None
                    self.last_top_crop_box = None
                    self.last_bottom_crop_box = None

            # --- Create final frame based on the STABLE active mode ---
            background = self._create_non_stretched_background(frame)
            background = cv2.GaussianBlur(background, (51, 51), 0)
            background = cv2.addWeighted(background, 0.4, np.zeros_like(background), 0.6, 0)

            if self.active_mode == '2_face':
                if len(bboxes) == 2:
                    bboxes.sort(key=lambda b: b[0])
                    self.current_left_bbox, self.current_right_bbox = bboxes
                if hasattr(self, 'current_left_bbox'):
                    top_target = self._calculate_zoomed_out_face_target(self.current_left_bbox)
                    self.last_top_crop_box = self._apply_smoothing_and_inertia(top_target, self.last_top_crop_box)
                    top_crop = self._get_crop_from_box(frame, self.last_top_crop_box)
                    
                    bottom_target = self._calculate_zoomed_out_face_target(self.current_right_bbox)
                    self.last_bottom_crop_box = self._apply_smoothing_and_inertia(bottom_target, self.last_bottom_crop_box)
                    bottom_crop = self._get_crop_from_box(frame, self.last_bottom_crop_box)

                    top_region_h = self.output_height // 2
                    self._place_sub_frame(background, top_crop, (0, 0, self.output_width, top_region_h))
                    self._place_sub_frame(background, bottom_crop, (0, top_region_h, self.output_width, self.output_height - top_region_h))
            
            if self.active_mode == 'general':
                target_box = self._calculate_target_crop(bboxes)
                self.last_general_crop_box = self._apply_smoothing_and_inertia(target_box, self.last_general_crop_box)
                final_crop = self._get_crop_from_box(frame, self.last_general_crop_box)
                self._place_sub_frame(background, final_crop, (0, 0, self.output_width, self.output_height))

            process.stdin.write(background.tobytes())
            frame_count += 1
            self.last_face_count = len(bboxes) # Update face count for the next frame's check
            if frame_count % 150 == 0:
                logging.info(f"Processed frame {frame_count}/{self.total_frames}. Active mode: '{self.active_mode}'")

        logging.info("Cleaning up resources.")
        self.cap.release()
        process.stdin.close()
        process.wait()
        self.face_detection.close()
        logging.info(f"Video processing complete. Output saved to {self.output_path}")

def main():
    parser = argparse.ArgumentParser(description="Crop a video to a 9:16 aspect ratio with smooth, stabilized camera motion.")
    parser.add_argument("input_video", help="Path to the input video file.")
    parser.add_argument("output_video", help="Path for the processed output video file.")
    parser.add_argument("--smoothing", type=float, default=0.07, help="Smoothing factor (0-1). Lower is smoother. Default: 0.07")
    parser.add_argument("--inertia", type=float, default=0.7, help="Inertia zone (0-1). Higher means less reactive. Default: 0.7")
    parser.add_argument("--enter-threshold", type=int, default=5, help="Frames to confirm a new mode before switching. Default: 5")
    parser.add_argument("--exit-threshold", type=int, default=15, help="Frames to wait before exiting a mode after faces are lost. Default: 15")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to draw face bounding boxes on the output video.")
    parser.add_argument("--upscale", type=float, default=1.0, help="Factor to upscale frames before detection for small faces (e.g., 1.5). Default: 1.0 (no upscale).")
    parser.add_argument("--confidence", type=float, default=0.5, help="Minimum face detection confidence (0.0-1.0). Lower to detect more distant faces. Default: 0.5")
    parser.add_argument("--scene-cut-threshold", type=float, default=0.99, help="Threshold for scene cut detection (0.0-1.0). Lower is more sensitive. Try 0.6. Default: 0.99 (mostly disabled).")
    
    args = parser.parse_args()
    try:
        processor = VideoProcessor(args.input_video, args.output_video, 
                                   smoothing_factor=args.smoothing, 
                                   inertia_zone=args.inertia,
                                   enter_threshold=args.enter_threshold,
                                   exit_threshold=args.exit_threshold,
                                   debug_mode=args.debug,
                                   upscale_factor=args.upscale,
                                   confidence=args.confidence,
                                   scene_cut_threshold=args.scene_cut_threshold)
        processor.process_video()
    except (IOError, ValueError) as e:
        logging.error(f"An error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()

