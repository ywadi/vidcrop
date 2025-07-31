import cv2
import numpy as np
import mediapipe as mp
import argparse
import sys
import logging
import os
import ffmpeg

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/data/video_to_shorts.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.4)

# Constants
ASPECT_RATIO = 9 / 16  # 9:16 for shorts
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
MARGIN_FACTOR = 0.1  # 10% margin around faces
SMOOTHING_ALPHA = 0.3  # Smoothing factor for crop transitions
BITRATE = "5000k"
DETECTION_INTERVAL = 3  # Detect faces every 3rd frame
BLUR_KERNEL = (21, 21)  # Gaussian blur kernel
DARKEN_FACTOR = 0.5  # Darken blurred background to 50% brightness

def preprocess_frame(frame):
    """Apply histogram equalization to improve face detection."""
    logger.debug("Preprocessing frame for face detection")
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)

def get_face_bboxes(frame):
    """Detect faces in a frame and return bounding boxes in pixel coordinates."""
    logger.debug("Detecting faces in frame")
    rgb_frame = preprocess_frame(frame)
    results = face_detection.process(rgb_frame)
    bboxes = []
    if results.detections:
        height, width = frame.shape[:2]
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x_min = int(bbox.xmin * width)
            y_min = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)
            x_max = x_min + w
            y_max = y_min + h
            # Clamp coordinates
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width, x_max)
            y_max = min(height, y_max)
            confidence = detection.score[0]
            bboxes.append((x_min, y_min, x_max, y_max))
            logger.debug(f"Face detected with confidence {confidence:.2f}: ({x_min}, {y_min}, {x_max}, {y_max})")
    logger.info(f"Detected {len(bboxes)} faces")
    return bboxes

def compute_crop_1_face(bbox, frame_shape, prev_crops, prev_face_count):
    """Compute crop for a single face, resetting if previous crop was for no faces."""
    logger.debug("Computing crop for 1 face")
    frame_w, frame_h = frame_shape
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2

    # Calculate crop dimensions (target 9:16 but preserve input aspect for scaling)
    crop_h = min(frame_h, int(frame_w / ASPECT_RATIO))
    crop_w = int(crop_h * ASPECT_RATIO)
    margin_w = int(crop_w * MARGIN_FACTOR)
    margin_h = int(crop_h * MARGIN_FACTOR)

    # Expand bounding box with margin
    x_min = max(0, x_min - margin_w)
    x_max = min(frame_w, x_max + margin_w)
    y_min = max(0, y_min - margin_h)
    y_max = min(frame_h, y_max + margin_h)

    # Check if previous crop exists and is not a full-frame crop (from no faces)
    if (prev_crops and len(prev_crops) > 0 and isinstance(prev_crops[0], tuple) and
            prev_face_count > 0 and prev_crops[0] != (0, 0, frame_w, frame_h)):
        px_min, py_min, px_max, py_max = prev_crops[0]
        if (x_min >= px_min and x_max <= px_max and
                y_min >= py_min and y_max <= py_max):
            logger.debug("Reusing previous crop for 1 face")
            return [prev_crops[0]], False

    # Compute new crop centered on face
    x_crop = x_center - crop_w // 2
    y_crop = y_center - crop_h // 2
    x_crop = max(0, min(frame_w - crop_w, x_crop))
    y_crop = max(0, min(frame_h - crop_h, y_crop))
    crop = (x_crop, y_crop, x_crop + crop_w, y_crop + crop_h)
    logger.debug(f"New crop for 1 face: {crop}")
    if prev_face_count == 0:
        logger.debug("Reset crop due to transition from no faces to 1 face")
    return [crop], True

def compute_crop_2_faces(bboxes, frame_shape, prev_crops, prev_face_count):
    """Compute two crops for two faces, resetting if previous crop was for no faces."""
    logger.debug("Computing crops for 2 faces")
    frame_w, frame_h = frame_shape
    # Sort by x to assign left face to top, right face to bottom
    bboxes = sorted(bboxes, key=lambda b: b[0])
    top_bbox, bottom_bbox = bboxes

    # Each crop targets half height (9:8), but preserve input aspect for scaling
    crop_h = min(frame_h, int(frame_w / (ASPECT_RATIO * 2)))
    crop_w = int(crop_h * ASPECT_RATIO)
    margin_w = int(crop_w * MARGIN_FACTOR)
    margin_h = int(crop_h * MARGIN_FACTOR)

    crops = []
    new_crop_flags = []
    for idx, (x_min, y_min, x_max, y_max) in enumerate([top_bbox, bottom_bbox]):
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2
        x_min = max(0, x_min - margin_w)
        x_max = min(frame_w, x_max + margin_w)
        y_min = max(0, y_min - margin_h)
        y_max = min(frame_h, y_max + margin_h)

        # Check if previous crop exists and is not from no faces
        if (prev_crops and len(prev_crops) > idx and isinstance(prev_crops[idx], tuple) and
                prev_face_count > 0 and prev_crops[0] != (0, 0, frame_w, frame_h)):
            px_min, py_min, px_max, py_max = prev_crops[idx]
            if (x_min >= px_min and x_max <= px_max and
                    y_min >= py_min and y_max <= py_max):
                logger.debug(f"Reusing previous crop for face {idx}")
                crops.append(prev_crops[idx])
                new_crop_flags.append(False)
                continue

        # Compute new crop
        x_crop = x_center - crop_w // 2
        y_crop = y_center - crop_h // 2
        x_crop = max(0, min(frame_w - crop_w, x_crop))
        y_crop = max(0, min(frame_h - crop_h, y_crop))
        crop = (x_crop, y_crop, x_crop + crop_w, y_crop + crop_h)
        crops.append(crop)
        new_crop_flags.append(True)
        logger.debug(f"New crop for face {idx}: {crop}")

    if prev_face_count == 0:
        logger.debug("Reset crops due to transition from no faces to 2 faces")
    return crops, new_crop_flags

def compute_crop_multiple_faces(bboxes, frame_shape, prev_crops, prev_face_count):
    """Compute a single crop encompassing multiple faces, resetting if previous crop was for no faces."""
    logger.debug(f"Computing crop for {len(bboxes)} faces")
    frame_w, frame_h = frame_shape
    x_mins = [b[0] for b in bboxes]
    y_mins = [b[1] for b in bboxes]
    x_maxs = [b[2] for b in bboxes]
    y_maxs = [b[3] for b in bboxes]
    x_min = min(x_mins)
    y_min = min(y_mins)
    x_max = max(x_maxs)
    y_max = max(y_maxs)

    # Calculate crop dimensions (target 9:16 but preserve input aspect for scaling)
    crop_h = min(frame_h, int(frame_w / ASPECT_RATIO))
    crop_w = int(crop_h * ASPECT_RATIO)
    margin_w = int(crop_w * MARGIN_FACTOR)
    margin_h = int(crop_h * MARGIN_FACTOR)

    # Expand with margin
    x_min = max(0, x_min - margin_w)
    x_max = min(frame_w, x_max + margin_w)
    y_min = max(0, y_min - margin_h)
    y_max = min(frame_h, y_max + margin_h)

    # Check if all faces are within previous crop and not from no faces
    if (prev_crops and len(prev_crops) > 0 and isinstance(prev_crops[0], tuple) and
            prev_face_count > 0 and prev_crops[0] != (0, 0, frame_w, frame_h)):
        px_min, py_min, px_max, py_max = prev_crops[0]
        all_within = all(px_min <= x_mins[i] and x_maxs[i] <= px_max and
                         py_min <= y_mins[i] and y_maxs[i] <= py_max
                         for i in range(len(bboxes)))
        if all_within:
            logger.debug("Reusing previous crop for multiple faces")
            return [prev_crops[0]], False

    # Compute new crop to encompass all faces
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    x_crop = x_center - crop_w // 2
    y_crop = y_center - crop_h // 2
    x_crop = max(0, min(frame_w - crop_w, x_crop))
    y_crop = max(0, min(frame_h - crop_h, y_crop))
    crop = (x_crop, y_crop, x_crop + crop_w, y_crop + crop_h)
    logger.debug(f"New crop for multiple faces: {crop}")
    if prev_face_count == 0:
        logger.debug(f"Reset crop due to transition from no faces to {len(bboxes)} faces")
    return [crop], True

def compute_center_crop(frame_shape):
    """Use the full frame for no faces case."""
    logger.debug("Computing full frame for no faces")
    frame_w, frame_h = frame_shape
    crop = (0, 0, frame_w, frame_h)
    logger.debug(f"Full frame crop: {crop}")
    return [crop], True

def smooth_crop(current_crop, prev_smooth_crop):
    """Apply exponential moving average to smooth crop coordinates."""
    if prev_smooth_crop is None:
        logger.debug("No previous smooth crop, using current")
        return current_crop
    x_min, y_min, x_max, y_max = current_crop
    px_min, py_min, px_max, py_max = prev_smooth_crop
    x_min = int(SMOOTHING_ALPHA * x_min + (1 - SMOOTHING_ALPHA) * px_min)
    y_min = int(SMOOTHING_ALPHA * y_min + (1 - SMOOTHING_ALPHA) * py_min)
    x_max = int(SMOOTHING_ALPHA * x_max + (1 - SMOOTHING_ALPHA) * px_max)
    y_max = int(SMOOTHING_ALPHA * y_max + (1 - SMOOTHING_ALPHA) * py_max)
    logger.debug(f"Smoothed crop: ({x_min}, {y_min}, {x_max}, {y_max})")
    return (x_min, y_min, x_max, y_max)

def create_blurred_background(frame, target_w, target_h):
    """Create a blurred, darkened background from the input frame, preserving aspect ratio."""
    logger.debug(f"Creating blurred background: {target_w}x{target_h}")
    frame_h, frame_w = frame.shape[:2]
    blurred = cv2.GaussianBlur(frame, BLUR_KERNEL, 0)
    
    # Darken the blurred frame
    darkened = cv2.convertScaleAbs(blurred, alpha=DARKEN_FACTOR, beta=0)
    logger.debug(f"Applied darkening with factor {DARKEN_FACTOR}")

    # Scale to cover target dimensions without stretching
    frame_aspect = frame_w / frame_h
    target_aspect = target_w / target_h
    if frame_aspect > target_aspect:
        # Frame is wider: scale by height to cover target height
        scale = target_h / frame_h
    else:
        # Frame is taller: scale by width to cover target width
        scale = target_w / frame_w

    new_w = int(frame_w * scale)
    new_h = int(frame_h * scale)
    logger.debug(f"Scaling blurred frame {frame_w}x{frame_h} to {new_w}x{new_h} (scale: {scale:.2f})")

    # Resize blurred frame
    scaled_blurred = cv2.resize(darkened, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Crop to target dimensions
    offset_x = (new_w - target_w) // 2
    offset_y = (new_h - target_h) // 2
    logger.debug(f"Cropping blurred frame at offset ({offset_x}, {offset_y})")

    x1 = max(0, offset_x)
    y1 = max(0, offset_y)
    x2 = min(new_w, offset_x + target_w)
    y2 = min(new_h, offset_y + target_h)
    background = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    background[0:y2-y1, 0:x2-x1] = scaled_blurred[y1:y2, x1:x2]
    return background

def place_crop_on_background(crop_frame, target_w, target_h, background):
    """Scale crop to fit target dimensions without stretching, place on background."""
    crop_h, crop_w = crop_frame.shape[:2]
    crop_aspect = crop_w / crop_h
    target_aspect = target_w / target_h

    # Scale to fit while preserving aspect ratio
    if crop_aspect > target_aspect:
        # Crop is wider: scale by width
        scale = target_w / crop_w
    else:
        # Crop is taller: scale by height
        scale = target_h / crop_h

    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    logger.debug(f"Scaling crop {crop_w}x{crop_h} to {new_w}x{new_h} (scale: {scale:.2f})")

    # Resize crop
    scaled_crop = cv2.resize(crop_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create output frame with background
    output_frame = background.copy()
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    logger.debug(f"Placing crop at offset ({offset_x}, {offset_y})")

    # Ensure offsets and dimensions are within bounds
    x1 = max(0, offset_x)
    y1 = max(0, offset_y)
    x2 = min(target_w, offset_x + new_w)
    y2 = min(target_h, offset_y + new_h)
    crop_x1 = max(0, -offset_x)
    crop_y1 = max(0, -offset_y)
    crop_x2 = crop_x1 + (x2 - x1)
    crop_y2 = crop_y1 + (y2 - y1)

    # Place crop on background
    output_frame[y1:y2, x1:x2] = scaled_crop[crop_y1:crop_y2, crop_x1:crop_x2]
    return output_frame

def process_frame(frame, bboxes, prev_crops, prev_smooth_crops, frame_shape, prev_face_count):
    """Process a single frame based on number of faces."""
    logger.debug(f"Processing frame with {len(bboxes)} faces (previous: {prev_face_count})")
    frame_w, frame_h = frame_shape

    # Compute crops
    if len(bboxes) == 0:
        # No faces: use full frame
        crops, new_crop = compute_center_crop(frame_shape)
    elif len(bboxes) == 1:
        # One face: center on face
        crops, new_crop = compute_crop_1_face(bboxes[0], frame_shape, prev_crops, prev_face_count)
    elif len(bboxes) == 2:
        # Two faces: split frame
        crops, new_crop_flags = compute_crop_2_faces(bboxes, frame_shape, prev_crops, prev_face_count)
        new_crop = any(new_crop_flags)
    else:
        # Multiple faces: encompass all
        crops, new_crop = compute_crop_multiple_faces(bboxes, frame_shape, prev_crops, prev_face_count)

    # Smooth crops if new and not transitioning from no faces
    if new_crop and prev_face_count > 0:
        smoothed_crops = [
            smooth_crop(c, prev_smooth_crops[i] if prev_smooth_crops and i < len(prev_smooth_crops) else None)
            for i, c in enumerate(crops)
        ]
    else:
        smoothed_crops = crops
        if prev_face_count == 0 and len(bboxes) > 0:
            logger.debug("Skipping smoothing due to transition from no faces")
        else:
            logger.debug("No new crop or no faces, using current crops")

    # Create output frame
    output_frame = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
    background = create_blurred_background(frame, OUTPUT_WIDTH, OUTPUT_HEIGHT)

    if len(bboxes) == 2:
        # Two faces: process each crop for half the frame
        for idx, (x_min, y_min, x_max, y_max) in enumerate(smoothed_crops):
            crop_frame = frame[y_min:y_max, x_min:x_max]
            target_h = OUTPUT_HEIGHT // 2
            target_w = OUTPUT_WIDTH
            # Create half-sized background for this crop
            half_background = create_blurred_background(frame, target_w, target_h)
            half_frame = place_crop_on_background(crop_frame, target_w, target_h, half_background)
            y_offset = idx * target_h
            output_frame[y_offset:y_offset + target_h, :] = half_frame
            logger.debug(f"Applied crop {idx} to half {y_offset}:{y_offset + target_h}")
    else:
        # No faces, one face, or multiple faces: single crop
        x_min, y_min, x_max, y_max = smoothed_crops[0]
        crop_frame = frame[y_min:y_max, x_min:x_max]
        output_frame = place_crop_on_background(crop_frame, OUTPUT_WIDTH, OUTPUT_HEIGHT, background)
        logger.debug(f"Applied single crop: ({x_min}, {y_min}, {x_max}, {y_max})")

    return output_frame, smoothed_crops, new_crop, len(bboxes)

def merge_audio_video(input_video, temp_video, output_video):
    """Merge audio from input video with processed video using ffmpeg."""
    logger.info(f"Merging audio from {input_video} to {output_video}")
    try:
        input_stream = ffmpeg.input(input_video)
        video_stream = ffmpeg.input(temp_video)
        audio = input_stream.audio
        video = video_stream.video
        output = ffmpeg.output(video, audio, output_video, vcodec='copy', acodec='copy', **{'b:v': BITRATE})
        ffmpeg.run(output, overwrite_output=True)
        logger.info("Audio merged successfully")
        os.remove(temp_video)
        logger.debug(f"Removed temporary file {temp_video}")
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error: {e.stderr.decode()}")
        raise

def main(input_path, output_path):
    """Main function to process video."""
    logger.info(f"Starting video processing: input={input_path}, output={output_path}")
    if not os.path.exists(input_path):
        logger.error(f"Input video {input_path} does not exist")
        sys.exit(1)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Cannot open input video {input_path}")
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_shape = (frame_w, frame_h)
    logger.info(f"Video info: width={frame_w}, height={frame_h}, fps={fps}, frames={frame_count}")

    # Initialize video writer for temporary file
    temp_output = '/data/temp_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    if not out.isOpened():
        logger.error(f"Cannot create temporary video {temp_output}")
        cap.release()
        sys.exit(1)

    prev_crops = None
    prev_smooth_crops = None
    last_bboxes = []
    prev_face_count = 0
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video reached")
            break

        logger.debug(f"Processing frame {frame_idx}")
        # Detect faces every DETECTION_INTERVAL frames
        if frame_idx % DETECTION_INTERVAL == 0:
            bboxes = get_face_bboxes(frame)
            last_bboxes = bboxes
        else:
            bboxes = last_bboxes
            logger.debug(f"Reusing face bboxes from frame {frame_idx - (frame_idx % DETECTION_INTERVAL)}")

        try:
            output_frame, crops, new_crop, curr_face_count = process_frame(
                frame, bboxes, prev_crops, prev_smooth_crops, frame_shape, prev_face_count
            )
        except Exception as e:
            logger.error(f"Error processing frame {frame_idx}: {str(e)}")
            cap.release()
            out.release()
            sys.exit(1)

        out.write(output_frame)
        prev_crops = crops
        prev_smooth_crops = crops
        prev_face_count = curr_face_count
        frame_idx += 1

    logger.info(f"Processed {frame_idx} frames")
    cap.release()
    out.release()
    face_detection.close()

    # Merge audio with video
    merge_audio_video(input_path, temp_output, output_path)
    logger.info("Video processing completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert video to 9:16 shorts with smart cropping.")
    parser.add_argument("input_video", help="Path to input video")
    parser.add_argument("output_video", help="Path to output video")
    args = parser.parse_args()
    main(args.input_video, args.output_video)