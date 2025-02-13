import os
from src.preprocess.mog2_preprocess import MOG2_process_video
from src.train.train import train_binary
from src.test.test import test_binary
import tensorflow as tf
import time
from datetime import datetime

def save_step_time(method_name, step_name, duration, timestamp, exe_time_dir):
    """Save execution time for a single step immediately after completion"""
    filename = f'step_time_{method_name}_{step_name}_{timestamp}.txt'
    filepath = os.path.join(exe_time_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(f"{method_name} - {step_name} Execution Time\n")
        f.write("================================\n\n")
        f.write(f"Total time: {duration:.2f} seconds\n")
        
        # Add human-readable format
        hours, rem = divmod(duration, 3600)
        minutes, seconds = divmod(rem, 60)
        f.write(f"Human Readable: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n")
    
    print(f"\n[INFO] {method_name} {step_name} time saved to {filepath}")

def save_preprocessing_benchmark(method_name, benchmark_results, benchmark_dir):
    """Save preprocessing benchmark results for a specific method"""
    benchmark_path = os.path.join(benchmark_dir, f'{method_name}_preprocessing_benchmark.txt')
    with open(benchmark_path, 'w') as f:
        f.write(f"{method_name} Preprocessing Benchmark Results\n")
        f.write("=" * (len(method_name) + 30) + "\n\n")
        
        total_time = 0
        for res in benchmark_results:
            f.write(f"Video: {res['video_name']}\n")
            f.write(f"Size: {res['size_mb']:.2f} MB\n")
            f.write(f"Processing Time: {res['processing_time_sec']:.2f} seconds\n")
            f.write(f"Processing Speed: {res['processing_speed_mbs']:.2f} MB/s\n")
            f.write("----------------------------\n")
            total_time += res['processing_time_sec']
        
        # Add summary statistics
        avg_time = total_time / len(benchmark_results)
        avg_speed = sum(r['processing_speed_mbs'] for r in benchmark_results) / len(benchmark_results)
        f.write("\nSummary Statistics\n")
        f.write("=================\n")
        f.write(f"Average Processing Time: {avg_time:.2f} seconds\n")
        f.write(f"Average Processing Speed: {avg_speed:.2f} MB/s\n")
        f.write(f"Total Processing Time: {total_time:.2f} seconds\n")
        f.write(f"Total Videos Processed: {len(benchmark_results)}\n")
    
    print(f"\n[INFO] {method_name} benchmark results saved to {benchmark_path}")
    return total_time

def main():
    # Create all necessary directories
    required_dirs = [
        "./result/benchmark/data_aug",
        "./result/exe_time/data_aug",
        "./result/binary/data_aug",
        "./result/binary_mog2/data_aug",
        "./result/best_model/data_aug",
        "./result/best_model_mog2/data_aug",
        "./data/processed/train",
        "./data/processed/val",
        "./data/processed/test",
        "./data/processed_mog2/train",
        "./data/processed_mog2/val",
        "./data/processed_mog2/test"
    ]
    
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"[INFO] Ensuring directory exists: {dir_path}")

    # Gather all .mp4 files from ./data/raw/
    video_dir = './data/raw/'
    video_paths = [
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.endswith('.mp4')
    ]

    # Define directory variables
    exe_time_dir = "./result/exe_time/data_aug"
    benchmark_dir = "./result/benchmark/data_aug"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Common parameters
    batch_size = 32
    epochs = 100
    
    xlss_path = "./resources/GasVid_Logging_File.xlsx"
    output_dir = "./data/processed_mog2"

    # # Process videos with Moving Average
    # print("\n====== Moving Average Background Subtraction ======")
    # mvag_benchmark_results = []
    # mvag_start_time = time.time()  # Start timing entire Moving Average process
    
    # for single_video_path in video_paths:
    #     video_name = os.path.basename(single_video_path)
    #     print(f"\nProcessing video with Moving Average: {video_name}")
        
    #     start_time = time.time()
    #     MOG2_process_video(single_video_path, xlss_path, output_dir)
    #     end_time = time.time()
        
    #     processing_time = end_time - start_time
    #     fps = os.path.getsize(single_video_path) / (1024*1024) / processing_time
        
    #     mvag_benchmark_results.append({
    #         'video_name': video_name,
    #         'size_mb': os.path.getsize(single_video_path) / (1024*1024),
    #         'processing_time_sec': processing_time,
    #         'processing_speed_mbs': fps
    #     })
        
    #     # Save interim results after each video
    #     save_preprocessing_benchmark('MovingAverage', mvag_benchmark_results, benchmark_dir)
    
    # mvag_total_time = time.time() - mvag_start_time  # Calculate total time including overhead

    # # Process videos with MOG2
    # print("\n====== MOG2 Background Subtraction ======")
    # mog2_benchmark_results = []
    # mog2_start_time = time.time()  # Start timing entire MOG2 process
    
    # for single_video_path in video_paths:
    #     video_name = os.path.basename(single_video_path)
    #     print(f"\nProcessing video with MOG2: {video_name}")
        
    #     start_time = time.time()
    #     MOG2_process_video(single_video_path, xlss_path, output_dir)
    #     end_time = time.time()
        
    #     processing_time = end_time - start_time
    #     fps = os.path.getsize(single_video_path) / (1024*1024) / processing_time
        
    #     mog2_benchmark_results.append({
    #         'video_name': video_name,
    #         'size_mb': os.path.getsize(single_video_path) / (1024*1024),
    #         'processing_time_sec': processing_time,
    #         'processing_speed_mbs': fps
    #     })
        
    #     # Save interim results after each video
    #     save_preprocessing_benchmark('MOG2', mog2_benchmark_results, benchmark_dir)
    
    # mog2_total_time = time.time() - mog2_start_time  # Calculate total time including overhead

    # # Save overall comparison with actual total times
    # comparison_path = os.path.join(benchmark_dir, 'preprocessing_comparison.txt')
    # with open(comparison_path, 'w') as f:
    #     f.write("Background Subtraction Methods Comparison\n")
    #     f.write("=======================================\n\n")
    #     f.write("Moving Average Method:\n")
    #     f.write(f"Total Processing Time (including overhead): {mvag_total_time:.2f} seconds\n")
    #     f.write(f"Pure Processing Time (sum of videos): {sum(r['processing_time_sec'] for r in mvag_benchmark_results):.2f} seconds\n")
    #     f.write(f"Average Time per Video: {mvag_total_time/len(video_paths):.2f} seconds\n\n")
        
    #     f.write("MOG2 Method:\n")
    #     f.write(f"Total Processing Time (including overhead): {mog2_total_time:.2f} seconds\n")
    #     f.write(f"Pure Processing Time (sum of videos): {sum(r['processing_time_sec'] for r in mog2_benchmark_results):.2f} seconds\n")
    #     f.write(f"Average Time per Video: {mog2_total_time/len(video_paths):.2f} seconds\n\n")
        
    #     f.write("\nTime Difference:\n")
    #     f.write(f"Absolute: {abs(mvag_total_time - mog2_total_time):.2f} seconds\n")
    #     f.write(f"Relative: {abs(mvag_total_time - mog2_total_time)/min(mvag_total_time, mog2_total_time)*100:.2f}%\n")
    
    # print(f"\n[INFO] Method comparison saved to {comparison_path}")

    # Moving Average method
    print("\n====== Moving Average Background Subtraction Method ======")
    # Moving Average method
    print("\n====== Moving Average Background Subtraction Method ======")
    mvag_train_dir = "./data/processed/train"
    mvag_val_dir = "./data/processed/val"
    mvag_test_dir = "./data/processed/test"

    mvag_model_path = "./result/binary/data_aug/cnn_3d_binaryAllLeak.keras"
    mvag_best_model_path = "./result/best_model/data_aug/cnn_3d_binaryAllLeak.keras"

    try:
        # Moving Average Training
        print("\n[INFO] Starting Moving Average method training...")
        mvag_train_start = time.time()
        train_binary(mvag_train_dir, mvag_val_dir, mvag_model_path, mvag_best_model_path, batch_size, epochs)
        mvag_train_duration = time.time() - mvag_train_start
        # Save training time immediately
        save_step_time("MovingAverage", "Training", mvag_train_duration, timestamp, exe_time_dir)
    except Exception as e:
        print(f"[ERROR] Moving Average training failed: {str(e)}")

    try:
        # Moving Average Testing
        print("\n[INFO] Starting Moving Average method testing...")
        mvag_test_start = time.time()
        test_binary(
            test_dir=mvag_test_dir, 
            model_path=mvag_best_model_path, 
            batch_size=batch_size,
            output_base_dir="./result/moving_average/data_aug",
            method_name="moving_average"
        )
        mvag_test_duration = time.time() - mvag_test_start
        save_step_time("MovingAverage", "Testing", mvag_test_duration, timestamp, exe_time_dir)
    except Exception as e:
        print(f"[ERROR] Moving Average testing failed: {str(e)}")

    # MOG2 method
    print("\n====== MOG2 Background Subtraction Method ======")
    mog2_train_dir = "./data/processed_mog2/train"
    mog2_val_dir = "./data/processed_mog2/val"
    mog2_test_dir = "./data/processed_mog2/test"
    
    mog2_model_path = "./result/binary_mog2/data_aug/cnn_3d_binaryAllLeak.keras"
    mog2_best_model_path = "./result/best_model_mog2/data_aug/cnn_3d_binaryAllLeak.keras"

    try:
        # MOG2 Training
        print("\n[INFO] Starting MOG2 method training...")
        mog2_train_start = time.time()
        train_binary(mog2_train_dir, mog2_val_dir, mog2_model_path, mog2_best_model_path, batch_size, epochs)
        mog2_train_duration = time.time() - mog2_train_start
        # Save training time immediately
        save_step_time("MOG2", "Training", mog2_train_duration, timestamp, exe_time_dir)
    except Exception as e:
        print(f"[ERROR] MOG2 training failed: {str(e)}")

    try:
        # MOG2 Testing
        print("\n[INFO] Starting MOG2 method testing...")
        mog2_test_start = time.time()
        test_binary(
            test_dir=mog2_test_dir, 
            model_path=mog2_best_model_path, 
            batch_size=batch_size,
            output_base_dir="./result/mog2/data_aug",
            method_name="mog2"
        )
        mog2_test_duration = time.time() - mog2_test_start
        # Save testing time immediately
        save_step_time("MOG2", "Testing", mog2_test_duration, timestamp, exe_time_dir)
    except Exception as e:
        print(f"[ERROR] MOG2 testing failed: {str(e)}")

    # Optionally create a summary of all completed steps
    try:
        create_summary(exe_time_dir, timestamp)
    except Exception as e:
        print(f"[ERROR] Failed to create summary: {str(e)}")

def create_summary(exe_time_dir, timestamp):
    """Create a summary of all completed steps if possible"""
    summary_path = os.path.join(exe_time_dir, f'summary_{timestamp}.txt')
    
    # Collect all step times for this run
    step_files = [f for f in os.listdir(exe_time_dir) if f.startswith(f'step_time_') and timestamp in f]
    
    with open(summary_path, 'w') as f:
        f.write("Execution Summary\n")
        f.write("================\n\n")
        
        for step_file in sorted(step_files):
            with open(os.path.join(exe_time_dir, step_file), 'r') as step_f:
                f.write(f"--- {step_file} ---\n")
                f.write(step_f.read())
                f.write("\n")

    print(f"\n[INFO] Summary created at {summary_path}")

if __name__ == "__main__":
    # Disable XLA and mixed precision
    tf.config.optimizer.set_jit(False)
    tf.keras.mixed_precision.set_global_policy('float32')

    # Force GPU memory growth to prevent OOM errors
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    main()
