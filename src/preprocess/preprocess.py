import cv2
import os
import numpy as np
import pandas as pd
from collections import deque
from scipy.fft import fft

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

def process_video(
    video_path,
    xlss_path,
    output_dir,
    frame_sizes=[15],
    segment_length_sec=180,
    total_segments=8,
    remove_start_sec=15,
    remove_end_sec=5,
    gaussian_kernel_size=(7,7),
    background_window=210,
    # Farneback parameters - adjusted for better plume detection
    pyr_scale=0.5,
    levels=3,
    winsize=21,  # Increased for better motion pattern detection
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    # Refined thresholds
    mmt_threshold=1.2,
    pat_threshold=100,  # Further lowered for very subtle plumes
    min_movement_threshold=10
):
    """
    Process a ~24-minute video in eight 3-minute segments (classes 0..7).
    - For class 0 only, use the Excel 'Start Time for Leak 0 Video Clip'.
    - For all classes, remove the first 15s and last 5s before saving.
    - Continuous background subtraction with a 210-frame rolling buffer.
    - If the filename has '_s2', we do a train/val split; if '_s1', we save everything to test.
    - We also write each segment's final start/end (mm:ss) to a .txt file.
    """
    
    def detect_flag_motion(flow, magnitude, prev_flag=None):
        """Simplified function that only detects flag motion"""
        flow_x, flow_y = flow[..., 0], flow[..., 1]
        angle = np.arctan2(flow_y, flow_x)
        
        # Local motion statistics for flag detection
        local_mag = cv2.blur(magnitude, (15,15))
        local_angle = cv2.blur(angle, (15,15))
        
        # Flag characteristics: high velocity + consistent direction
        flag_motion = (
            (magnitude > mmt_threshold * 2.5) &          # High velocity
            (local_mag > mmt_threshold * 2.2) &          # Sustained high motion
            (np.abs(local_angle - angle) < 0.2)          # Consistent direction
        )
        
        # Temporal consistency for flag detection
        if prev_flag is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            flag_motion = flag_motion & (cv2.dilate(prev_flag.astype(np.uint8), kernel) > 0)
        
        return flag_motion

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

    # Rolling buffer for background frames
    background_frames = deque(maxlen=background_window)

    prev_gray = None
    prev_flag = None
    
    
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
        # Motion history for temporal analysis
        # Read frames from chunk_start_frame up to chunk_end_frame
        while current_frame_index < chunk_end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, gaussian_kernel_size, 0)
            
            # 1. Detect flag motion using optical flow
            flag_mask = None
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale, levels, winsize,
                    iterations, poly_n, poly_sigma, 0
                )
                
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                flag_mask = detect_flag_motion(flow, magnitude, prev_flag)
                prev_flag = flag_mask
            
            # 2. Moving Average Background Subtraction
            gray_filtered = gray.copy()
            if flag_mask is not None:
                # Mask out flag regions
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                flag_mask_dilated = cv2.dilate(flag_mask.astype(np.uint8), kernel)
                gray_filtered[flag_mask_dilated > 0] = 0
            
            background_frames.append(gray_filtered)
            if len(background_frames) < background_window:
                bg = np.mean(background_frames, axis=0).astype(np.uint8)
            else:
                bg = np.median(np.array(background_frames), axis=0).astype(np.uint8)
            
            # Get foreground
            fg = cv2.absdiff(gray_filtered, bg)
            _, fg = cv2.threshold(fg, 8, 255, cv2.THRESH_BINARY)
            
            # Basic component filtering
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
                fg, connectivity=8
            )
            
            clean_fg = np.zeros_like(fg)
            for i in range(1, nb_components):
                size = stats[i, -1]
                if 10 <= size <= 1000:  # Basic size filtering
                    mask = output == i
                    clean_fg[mask] = 255
            
            # Basic movement check
            if np.count_nonzero(clean_fg) < min_movement_threshold:
                clean_fg = np.zeros_like(clean_fg)
            
            # Minimal cleanup
            clean_fg = cv2.medianBlur(clean_fg, 3)
            
            # Normalize
            fg = np.clip(clean_fg / 255.0, 0, 1)
            
            prev_gray = gray.copy()
            segment_frames.append(fg)
            current_frame_index += 1

            if current_frame_index >= chunk_end_frame:
                break
            
        # More permissive segment-level consistency
        if np.mean([np.count_nonzero(f) for f in segment_frames]) < pat_threshold * 0.2:
            segment_frames = [np.zeros_like(segment_frames[0]) for _ in segment_frames]

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


