"""
Improved MOG2 background subtractor for efficient processing.
- For class 0 only, use the Excel 'Start Time for Leak 0 Video Clip'.
- For all classes, remove the first 15s and last 5s before saving.
- Continuous background subtraction with a 210-frame rolling buffer.
- If the filename has '_s2', we do a train/val split; if '_s1', we save everything to test.
- We also write each segment's final start/end (mm:ss) to a .txt file.
"""

import cv2
import os
import numpy as np
import pandas as pd
from collections import deque

def parse_excel_time(time_str):
    """
    Interpret strings like:
      '0:16'      => 16   seconds
      '3:05'      => 185  seconds (3*60 + 5)
      '12:16:00'  => 16   seconds (ignore hour part)
    
    This helps correct Excel's misinterpretation of '0:16' as midnight times.
    """
    parts = time_str.split(":")
    if len(parts) == 1:
        # e.g. '16' => 16 seconds
        return int(parts[0])
    elif len(parts) == 2:
        # e.g. '3:16' => 3 min, 16 sec => 196
        minutes = int(parts[0])
        seconds = int(parts[1])
        return minutes * 60 + seconds
    elif len(parts) == 3:
        # e.g. '12:16:00' => treat middle part as 16 seconds
        return int(parts[1])
    else:
        # Fallback if something unexpected
        return 0

def seconds_to_min_sec(total_sec):
    """
    Convert an integer total seconds to the format MM:SS (e.g. 185 -> '3:05').
    """
    mm = total_sec // 60
    ss = total_sec % 60
    return f"{mm}:{ss:02d}"


def MOG2_process_video(
    video_path,
    xlss_path,
    output_dir,
    frame_sizes=[15],
    segment_length_sec=180,    # 3 minutes = 180 seconds
    total_segments=8,         # 8 segments for a 24-minute video
    remove_start_sec=15,
    remove_end_sec=5,
    gaussian_kernel_size=(7,7),
    history=210,
    var_threshold=16,
    detect_shadows=True,
    # Farneback parameters
    pyr_scale=0.5,
    levels=3,
    winsize=21,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    # Motion thresholds
    mmt_threshold=1.2,
    pat_threshold=100,
    min_movement_threshold=12,
    K = 3,
    alpha = 0.08,
    initial_variance = 15.0,
    min_variance = 10.0,
    weight_threshold = 0.9
):
    """
    Process a ~24-minute video in eight 3-minute segments (classes 0..7).
    Using MOG2 background subtractor for efficient processing.
    - For class 0 only, use the Excel 'Start Time for Leak 0 Video Clip'.
    - For all classes, remove the first 15s and last 5s before saving.
    - Continuous background subtraction with a 210-frame rolling buffer.
    - If the filename has '_s2', we do a train/val split; if '_s1', we save everything to test.
    - We also write each segment's final start/end (mm:ss) to a .txt file.
    """

    sharpen_kernel = np.array([
        [0, -0.2, 0],
    [-0.2,  2, -0.2],
    [0, -0.2, 0]
    ], dtype=np.float32)

    def detect_flag_motion(flow, magnitude, prev_flag=None):
        """Detect flag motion in optical flow"""
        flow_x, flow_y = flow[..., 0], flow[..., 1]
        angle = np.arctan2(flow_y, flow_x)
        
        local_mag = cv2.blur(magnitude, (15,15))
        local_angle = cv2.blur(angle, (15,15))
        
        flag_motion = (
            (magnitude > mmt_threshold * 2.5) &
            (local_mag > mmt_threshold * 2.2) &
            (np.abs(local_angle - angle) < 0.2)
        )
        
        if prev_flag is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            flag_motion = flag_motion & (cv2.dilate(prev_flag.astype(np.uint8), kernel) > 0)
        
        return flag_motion

    # Initialize MOG2 background subtractor
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows
    )

    # 1. Read the Excel for "Start Time for Leak 0"
    metadata = pd.read_excel(xlss_path)

    global_index = 0  # naming .npz segments
    
    # 2. Create output directories
    for split in ['train', 'val', 'test']:
        for class_label in range(8):
            os.makedirs(os.path.join(output_dir, split, str(class_label)), exist_ok=True)

    # 3. Basic checks on the video filename (e.g., 'MOV_1237_69_s1.mp4')
    if not os.path.isfile(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    video_name = os.path.basename(video_path)
    if not video_name.endswith(".mp4"):
        print(f"[WARNING] Not an MP4 file: {video_name}")
        return

    # Identify whether it's s1 or s2
    if "_s2" in video_name.lower():
        s_split = 's2'
    elif "_s1" in video_name.lower():
        s_split = 's1'
    else:
        print(f"[SKIP] {video_name} does not contain '_s1' or '_s2'")
        return

    # Parse the video number from the filename: e.g. "MOV_1237_69_s1"
    parts = video_name.split("_")
    distance_code = parts[2]
    # Typically: parts[0] = "MOV", parts[1] = "1237", parts[2] = "69", parts[3] = "s1.mp4"
    try:
        video_no = int(parts[1])  # e.g. 1237
    except ValueError:
        print(f"[ERROR] Cannot parse integer from {parts[1]} in {video_name}")
        return

    # 4. Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    fps = int(np.ceil(cap.get(cv2.CAP_PROP_FPS)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_sec = total_frames / fps
    print(f"[INFO] Video {video_name}: {total_frames} frames at ~{fps} FPS ({video_duration_sec:.2f} sec)")

    # 5. Get the Excel-based start time (for class 0 only)
    start_time_row = metadata[metadata['Video No.'] == video_no]
    if start_time_row.empty:
        print(f"[WARNING] No matching row in Excel for Video No. = {video_no}. Using 0 as start.")
        excel_start_sec = 0
    else:
        raw_excel_time = str(start_time_row['Start Time for Leak 0 Video Clip'].values[0])  
        excel_start_sec = parse_excel_time(raw_excel_time)
        print(f"[INFO] Excel-based start time for class 0 = '{raw_excel_time}' => {excel_start_sec} seconds")

    # 6. We will read the entire 24-minute range (or up to the actual end),
    #    starting from excel_start_sec. That is: [excel_start_sec, excel_start_sec + 24*60].
    #    But we'll break it into 8 consecutive 3-min chunks (class 0..7).
    #    If the video is shorter or longer, we clamp as needed.
    full_start_sec = excel_start_sec
    full_end_sec   = excel_start_sec + (total_segments * segment_length_sec)  # +24 minutes
    if full_end_sec * fps > total_frames:
        full_end_sec = video_duration_sec  # clamp to actual video end

    full_start_frame = int(full_start_sec * fps)
    full_end_frame   = int(full_end_sec * fps)

    # Move to the earliest frame:
    cap.set(cv2.CAP_PROP_POS_FRAMES, full_start_frame)
    current_frame_index = full_start_frame

    # 7. Loop over the 8 segments (class labels 0..7)
    for class_label in range(total_segments):

        # Each class => 3 min chunk
        chunk_start_sec = excel_start_sec + class_label * segment_length_sec
        chunk_end_sec   = chunk_start_sec + segment_length_sec

        if chunk_start_sec >= full_end_sec:
            break  # no more chunks to process
        if chunk_end_sec > full_end_sec:
            chunk_end_sec = full_end_sec

        chunk_start_frame = int(chunk_start_sec * fps)
        chunk_end_frame   = int(chunk_end_sec * fps)

        # If needed, skip frames if current_frame_index < chunk_start_frame
        if current_frame_index < chunk_start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_start_frame)
            current_frame_index = chunk_start_frame

        segment_frames = []
        prev_gray = None
        prev_flag = None
        first_frame = True

        # Read frames from chunk_start_frame up to chunk_end_frame
        while current_frame_index < chunk_end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            kernel = np.ones((3, 3), np.uint8)

            # Convert the current frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            gray = cv2.GaussianBlur(gray, gaussian_kernel_size, 0)
            
            #MOG2 algorithm (Test)
            if first_frame:
                # Initialize Gaussian mixtures for the first frame
                height, width = gray.shape
                means = np.zeros((height, width, K), dtype=np.float32)
                variances = np.ones((height, width, K), dtype=np.float32) * initial_variance
                weights = np.zeros((height, width, K), dtype=np.float32)

                means[:, :, 0] = gray
                weights[:, :, 0] = 1.0

                fg_mask = np.zeros_like(gray, dtype=np.uint8)
                first_frame = False
            else:
                # MOG2: Compute matches
                diffs = np.abs(gray[:, :, None] - means)
                sigmas = np.sqrt(variances)
                matches = diffs < 2.5 * sigmas
                match_any = np.any(matches, axis=2)

                # Identify best-matching Gaussian (highest weight among matches)
                masked_weights = weights.copy()
                masked_weights[~matches] = -1
                best_k = np.argmax(masked_weights, axis=2)

                # Identify least-weighted Gaussian for replacement
                min_k = np.argmin(weights, axis=2)

                # Update weights
                weights = (1 - alpha) * weights
                rows, cols = np.where(match_any)
                if len(rows) > 0:
                    k_to_update = best_k[rows, cols]
                    weights[rows, cols, k_to_update] += alpha

                    # Update means and variances
                    diff = gray[rows, cols] - means[rows, cols, k_to_update]
                    means[rows, cols, k_to_update] += alpha * diff
                    variances[rows, cols, k_to_update] = (
                        (1 - alpha) * variances[rows, cols, k_to_update] + alpha * diff**2
                    )
                    variances[rows, cols, k_to_update] = np.maximum(
                        variances[rows, cols, k_to_update], min_variance
                    )

                # Replace least-weighted Gaussian for non-matching pixels
                rows_no_match, cols_no_match = np.where(~match_any)
                if len(rows_no_match) > 0:
                    k_to_replace = min_k[rows_no_match, cols_no_match]
                    means[rows_no_match, cols_no_match, k_to_replace] = gray[rows_no_match, cols_no_match]
                    variances[rows_no_match, cols_no_match, k_to_replace] = initial_variance
                    weights[rows_no_match, cols_no_match, k_to_replace] = alpha

                # Normalize weights
                sum_weights = np.sum(weights, axis=2, keepdims=True)
                weights = weights / sum_weights

                # Compute foreground mask
                bg_gaussians = weights > weight_threshold
                matches_bg = matches & bg_gaussians
                max_weight_k = np.argmax(weights, axis=2)
                bg_mean = means[np.arange(height)[:, None], np.arange(width), max_weight_k]
                fg_mask = cv2.absdiff(gray, bg_mean.astype(np.uint8))
            
            
            flag_mask = None
            # 1. Motion Analysis (keep the same as it works well)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale, levels, winsize,
                    iterations, poly_n, poly_sigma, 0
                )
                
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                flag_mask = detect_flag_motion(flow, magnitude, prev_flag)
                prev_flag = flag_mask
  
            if flag_mask is not None:
                # Mask out flag regions
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                flag_mask_dilated = cv2.dilate(flag_mask.astype(np.uint8), kernel)
                fg_mask[flag_mask_dilated > 0] = 0
            
            # Post-processing on foreground mask
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # 3. Component Analysis
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
            min_area = 80
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < min_area:
                    fg_mask[labels == i] = 0

            if np.count_nonzero(fg_mask) < 10:
                fg_mask = np.zeros_like(fg_mask)
            
            # Basic movement check
            if np.count_nonzero(fg_mask) < min_movement_threshold:
                fg_mask = np.zeros_like(fg_mask)
            
            fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
            fg_mask = cv2.filter2D(fg_mask, -1, sharpen_kernel)
            fg_mask = np.clip(fg_mask / 255.0, 0, 1)
            
            # Normalize to [0,1] range
            
            prev_gray = gray.copy()
            segment_frames.append(fg_mask)
            current_frame_index += 1

            if current_frame_index >= chunk_end_frame:
                break

        # Keep the same segment-level checks and saving code
        # if np.mean([np.count_nonzero(f) for f in segment_frames]) < pat_threshold * 0.2:
        #     segment_frames = [np.zeros_like(segment_frames[0]) for _ in segment_frames]

        # Remove first 15 sec and last 5 sec from this chunk
        start_remove = remove_start_sec * fps
        end_remove   = remove_end_sec * fps
        if len(segment_frames) <= (start_remove + end_remove):
            print(f"[WARNING] Class {class_label} has too few frames after removing 15s+5s.")
            continue
        segment_frames = segment_frames[start_remove : -end_remove]

        # final shape: (N, 240, 320, 1) if your frames are 240×320
        segment_frames = np.array(segment_frames).reshape(-1, 240, 320, 1)

        # Calculate the final kept start/end time for logging
        # e.g. chunk_abs_start + 15s, chunk_abs_end - 5s
        final_start_sec = chunk_start_sec + remove_start_sec
        final_end_sec   = chunk_end_sec   - remove_end_sec
        start_str = seconds_to_min_sec(int(final_start_sec))
        end_str   = seconds_to_min_sec(int(final_end_sec))

        # Before writing to time_log_path, add:
        # txt_dir = os.path.join(output_dir, 'txt_file')
        # os.makedirs(txt_dir, exist_ok=True)  # Create directory if it doesn't exist

        # time_log_path = os.path.join(txt_dir, f'{video_name}_class{class_label}_times.txt')
        # with open(time_log_path, 'w') as f:
        #     f.write(f"Class {class_label} Start Time: {start_str}\n")
        #     f.write(f"Class {class_label} End Time:   {end_str}\n")

        # 8. Save sliding-window patches
        step = 5  # a step of 5 frames
        for num_frames_to_sample in frame_sizes:
            # e.g. if num_frames_to_sample=15, we collect consecutive 15-frame clips
            # overlapping with stride 5
            slices = []
            for start_idx in range(0, len(segment_frames)-num_frames_to_sample+1, step):
                clip = segment_frames[start_idx : start_idx + num_frames_to_sample]
                slices.append(clip)
            slices = np.array(slices)

            # Save them. If '_s2', random 80% to train, 20% to val; if '_s1', all to test
            if s_split == 's2':
                for clip_arr in slices:
                    if np.random.rand() < 0.8:
                        sub_dir = os.path.join(output_dir, 'train', str(class_label))
                    else:
                        sub_dir = os.path.join(output_dir, 'val', str(class_label))
                    os.makedirs(sub_dir, exist_ok=True)
                    seg_path = os.path.join(
                        sub_dir, f"segment{class_label}_{video_no}_{distance_code}_{global_index}.npz"
                    )
                    np.savez_compressed(seg_path, segment=clip_arr)
                    global_index += 1
            else:
                # s_split == 's1' -> test
                sub_dir = os.path.join(output_dir, 'test', str(class_label))
                os.makedirs(sub_dir, exist_ok=True)
                for clip_arr in slices:
                    seg_path = os.path.join(
                        sub_dir, f"segment{class_label}_{video_no}_{distance_code}_{global_index}.npz"
                    )
                    np.savez_compressed(seg_path, segment=clip_arr)
                    global_index += 1

        print(f"[INFO] Class {class_label} saved. Final start={start_str}, end={end_str}")

    cap.release()
    print("[DONE] Finished processing video:", video_name)