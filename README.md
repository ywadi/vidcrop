# **Auto-Cropping Video to Shorts**

This project provides a powerful Python script that uses computer vision to automatically reframe standard 16:9 videos into engaging 9:16 vertical shorts, perfect for platforms like TikTok, YouTube Shorts, and Instagram Reels.

The script intelligently tracks faces, handles different numbers of subjects, and uses sophisticated smoothing and scene detection to create a professional-looking final product with a "virtual camera operator."

## **Key Features**

* **Multi-Face Detection:** Uses Google's MediaPipe to accurately detect multiple faces in real-time.  
* **Dynamic Layouts:**  
  * **1 Face:** A stable, centered crop that follows the subject without zooming.  
  * **2 Faces:** A split-screen view, stacking the subjects vertically (left face on top).  
  * **3+ Faces:** A group shot that intelligently pans and zooms to keep everyone in the frame.  
  * **0 Faces:** A clean view of the original, uncropped frame, centered over a blurred background.  
* **Cinematic Camera Motion:**  
  * **Smoothing:** Eliminates jarring movements by creating smooth, fluid camera pans.  
  * **Inertia:** Prevents jitter from minor head movements by creating a "dead zone" where the camera stays still.  
* **Intelligent Scene Cut Detection:** Automatically detects hard cuts and instantly snaps the camera to the new subject, avoiding slow pans across scene changes.  
* **Robust Mode Switching:** Uses temporal filtering (grace periods) to prevent the layout from glitching if a face is momentarily lost.  
* **Enhanced Detection:** Can upscale and sharpen frames before analysis to improve the detection of small or distant faces.  
* **High Performance:** Pipes frames directly to FFmpeg for efficient encoding and preserves the original audio track without re-encoding.  
* **Debug Mode:** A visual mode to draw bounding boxes around detected faces for easy tuning and troubleshooting.

## **How it Works (Flow Logic)**

The script processes the video frame by frame, making a series of intelligent decisions to produce the final output.

1. **Frame Ingestion:** The script reads a single frame from the input video.  
2. **Scene Cut Detection:**  
   * It first compares the current frame's color histogram to the previous frame's.  
   * If the difference is larger than the \--scene-cut-threshold, it flags a "hard cut."  
   * A hard cut triggers a **full reset**: the camera's position and the layout mode are instantly updated to match the new scene, bypassing all smoothing and timers.  
3. **Face Detection (Pre-processing):**  
   * If the \--upscale factor is greater than 1, the script creates a temporary, larger, and sharpened version of the frame.  
   * It runs the MediaPipe face detector on this enhanced frame. This dramatically increases the chances of finding small or distant faces.  
   * The coordinates of any found faces are then scaled back down to match the original frame's dimensions.  
4. **Layout Mode Selection:**  
   * The script determines the "target" layout based on the number of faces found (0, 1, 2, or 3+).  
   * It then uses a **temporal filter** to decide whether to switch the layout:  
     * **Entering a Mode:** It will only switch *to* a new mode (e.g., from 1-face to 2-face) if the new face count is stable for \--enter-threshold consecutive frames.  
     * **Exiting a Mode:** It will only switch *away* from a mode (e.g., from 2-face to 1-face) if a face has been missing for \--exit-threshold consecutive frames. This prevents glitches from momentary detection failures.  
5. **Camera Target Calculation:**  
   * Based on the active layout mode, the script calculates the ideal "target" crop window(s) for the current frame.  
   * For the 2-face and 3+-face modes, this involves calculating a bounding box that contains the relevant subjects.  
6. **Camera Motion (Smoothing & Inertia):**  
   * The script compares the new target position with the camera's position from the previous frame.  
   * **Inertia Check:** If the target is within the "dead zone" defined by \--inertia, the camera **does not move at all**. This is the key to eliminating jitter.  
   * **Smoothing:** If the target is outside the dead zone, the camera moves towards it, but its movement is smoothed by the \--smoothing factor. A low value creates a slow, graceful pan, while a high value creates a quicker response.  
7. **Final Frame Composition:**  
   * A blurred and darkened version of the original frame is created as a 9:16 background.  
   * The final, smoothed crop window(s) are used to extract the foreground content from the original, high-quality frame.  
   * This foreground content is then placed on top of the background.  
8. **Output:** The final composed frame is piped to FFmpeg, which encodes it and combines it with the original audio stream into the output file.

## **Setup & Usage**

### **1\. Build the Docker Image**

Place the Dockerfile and video\_to\_shorts.py in the same directory and run:

docker build \-t video-to-shorts .

To ensure you are using the latest version of the script, you may need to build without the cache:

docker build \--no-cache \-t video-to-shorts .

### **2\. Run the Script**

Use the following command structure:

docker run \--rm \-v "$(pwd)":/data video-to-shorts \[INPUT\_FILE\] \[OUTPUT\_FILE\] \[OPTIONS\]

**Example:**

docker run \--rm \-v "$(pwd)":/data video-to-shorts /data/my\_video.mp4 /data/output.mp4 \--smoothing 0.2 \--scene-cut-threshold 0.6

## **Command-Line Arguments Explained**

### **input\_video & output\_video**

**Usage:** \[path\_to\_input\] \[path\_to\_output\]

* The first two arguments are the required paths for the input and output video files.

### **\--smoothing**

**Default:** 0.07

* **What it does:** Controls the speed of the virtual camera. It's a value between 0.0 and 1.0.  
* **How it affects the video:**  
  * **Low Value (e.g., 0.07):** Creates a very smooth, slow, and graceful camera pan. Ideal for interviews or slow-moving scenes.  
  * **High Value (e.g., 0.3):** Creates a fast, responsive camera that quickly catches up to new positions. Better for videos with faster action or frequent scene changes.

### **\--inertia**

**Default:** 0.7

* **What it does:** Creates an invisible "dead zone" around the tracked faces. The value represents the percentage of the crop window to use as this zone.  
* **How it affects the video:**  
  * **High** Value (e.g., 0.7): The camera is less reactive. It will ignore minor head movements and only move when a subject makes a significant position change. This is the primary tool for eliminating jitter.  
  * **Low Value (e.g., 0.2):** The camera is very sensitive and will react to even the smallest movements.

### **\--enter-threshold & \--exit-threshold**

**Defaults:** enter=5, exit=15

* **What they do:** Control the script's "patience" before changing layouts to prevent glitches.  
* **How they affect the video:**  
  * \--enter-threshold: The number of frames the script must consistently see a new layout before switching *to* it. A low value makes it quick to adapt to new people entering the scene.  
  * \--exit-threshold: The number of frames the script will wait after a face disappears before switching *away* from a layout. A **higher value is crucial** for handling cases where the detector momentarily loses a face, preventing the layout from flickering.

### **\--confidence**

**Default:** 0.5

* **What it does:** The minimum confidence score (0.0 to 1.0) the face detector needs to report a face.  
* **How it affects the video:**  
  * **Lower Value (e.g., 0.4):** Makes the detector more lenient. It will find more faces, especially those that are distant, blurry, or partially turned. This is the best way to improve detection on challenging footage.  
  * **Higher Value (e.g., 0.7):** Makes the detector stricter, reducing the chance of false positives but potentially missing difficult-to-see faces.

### **\--upscale**

**Default:** 1.0 (disabled)

* **What it does:** A factor by which to digitally zoom and sharpen the frame *before* running face detection.  
* **How it affects the video:**  
  * **Value \> 1.0 (e.g., 1.5):** This makes small, distant faces appear larger to the detector, significantly increasing the chance they will be found. This is a powerful tool for wide shots. It does not affect the final output quality, only the detection process.

### **\--scene-cut-threshold**

**Default:** 0.99 (mostly disabled)

* **What it does:** Controls the sensitivity of the automatic scene cut detection. It measures the similarity between frames (0.0 to 1.0).  
* **How it affects the video:**  
  * **Lower Value (e.g., 0.6):** Makes the detector more likely to register a scene cut. At this value, it will trigger on "hard cuts" but ignore smooth transitions like dissolves. This forces the camera and layout to snap instantly to the new scene.  
  * **Higher** Value (e.g., 0.99): Makes the detector extremely insensitive, effectively disabling it.

### **\--debug**

**Default:** False

* **What it does:** When enabled, it draws green bounding boxes around all detected faces directly onto the final output video.  
* **How it affects the video:** This is an invaluable tool for tuning.
