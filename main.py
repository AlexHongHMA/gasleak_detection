import os
from src.preprocess.mog2_preprocess import MOG2_process_video
from src.train.train import train_binary
from src.test.test import test_binary
import tensorflow as tf
import time
from datetime import datetime

def main():
    # Create all necessary directories
    required_dirs = [
        "./result/benchmark",
        "./result/exe_time",
        "./result/binary_mog2",
        "./result/best_model_mog2",
        "./data/processed_mog2/train",
        "./data/processed_mog2/val",
        "./data/processed_mog2/test"
    ]
    
    # Define directory variables
    exe_time_dir = "./result/exe_time"
    benchmark_dir = "./result/benchmark"
    
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
    
    xlss_path = "./resources/GasVid_Logging_File.xlsx"
    output_dir = "./data/processed_mog2"

    # Benchmark results
    benchmark_results = []
    
    # Loop over each video path with timing
    for single_video_path in video_paths:
        video_name = os.path.basename(single_video_path)
        print(f"\nProcessing video: {video_name}")
        
        # Time the processing
        start_time = time.time()
        MOG2_process_video(single_video_path, xlss_path, output_dir)
        end_time = time.time()
        
        # Calculate processing time
        processing_time = end_time - start_time
        fps = os.path.getsize(single_video_path) / (1024*1024) / processing_time  # MB/s
        
        # Store results
        result = {
            'video_name': video_name,
            'size_mb': os.path.getsize(single_video_path) / (1024*1024),
            'processing_time_sec': processing_time,
            'processing_speed_mbs': fps
        }
        benchmark_results.append(result)
        
        # Save interim results after each video
        benchmark_path = os.path.join(benchmark_dir, 'preprocessing_benchmark.txt')
        with open(benchmark_path, 'w') as f:
            f.write("Preprocessing Benchmark Results\n")
            f.write("============================\n\n")
            for res in benchmark_results:
                f.write(f"Video: {res['video_name']}\n")
                f.write(f"Size: {res['size_mb']:.2f} MB\n")
                f.write(f"Processing Time: {res['processing_time_sec']:.2f} seconds\n")
                f.write(f"Processing Speed: {res['processing_speed_mbs']:.2f} MB/s\n")
                f.write("----------------------------\n")
            
            # Add summary statistics
            avg_time = sum(r['processing_time_sec'] for r in benchmark_results) / len(benchmark_results)
            avg_speed = sum(r['processing_speed_mbs'] for r in benchmark_results) / len(benchmark_results)
            f.write("\nSummary Statistics\n")
            f.write("=================\n")
            f.write(f"Average Processing Time: {avg_time:.2f} seconds\n")
            f.write(f"Average Processing Speed: {avg_speed:.2f} MB/s\n")
            f.write(f"Total Videos Processed: {len(benchmark_results)}\n")

    # Paths for training/testing
    train_dir = "./data/processed_mog2/train"
    val_dir = "./data/processed_mog2/val"
    test_dir = "./data/processed_mog2/test"
    model_save_path = "./result/binary_mog2/cnn_3d_binaryAllLeak.keras"
    batch_size = 32
    epochs = 100
    best_model_path = "./result/best_model_mog2/cnn_3d_binaryAllLeak.keras"

    # Measure training time
    print("\n[INFO] Starting training...")
    train_start_time = time.time()
    train_binary(train_dir, val_dir, model_save_path, best_model_path, batch_size, epochs)
    train_end_time = time.time()
    train_duration = train_end_time - train_start_time

    # Measure testing time
    print("\n[INFO] Starting testing...")
    test_start_time = time.time()
    test_binary(test_dir, best_model_path, batch_size)
    test_end_time = time.time()
    test_duration = test_end_time - test_start_time

    # Save execution times
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exe_time_path = os.path.join(exe_time_dir, f'execution_times_{timestamp}.txt')
    
    with open(exe_time_path, 'w') as f:
        f.write("Execution Time Report\n")
        f.write("===================\n\n")
        f.write(f"Training:\n")
        f.write(f"  - Total time: {train_duration:.2f} seconds\n")
        f.write(f"  - Time per epoch: {train_duration/epochs:.2f} seconds\n\n")
        f.write(f"Testing:\n")
        f.write(f"  - Total time: {test_duration:.2f} seconds\n")
        
        # Add human-readable format
        hours_train, rem = divmod(train_duration, 3600)
        minutes_train, seconds_train = divmod(rem, 60)
        hours_test, rem = divmod(test_duration, 3600)
        minutes_test, seconds_test = divmod(rem, 60)
        
        f.write("\nHuman Readable Format:\n")
        f.write(f"Training: {int(hours_train)}h {int(minutes_train)}m {seconds_train:.2f}s\n")
        f.write(f"Testing:  {int(hours_test)}h {int(minutes_test)}m {seconds_test:.2f}s\n")

    print(f"\n[INFO] Execution times saved to {exe_time_path}")

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
