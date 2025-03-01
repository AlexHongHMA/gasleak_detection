import os
from src.preprocess.mog2_preprocess import MOG2_process_video
from src.preprocess.preprocess import process_video
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
        "./data/processed_optical_flow_gray",
        "./result/benchmark/optical_flow_gray",
        "./result/exe_time/optical_flow_gray",
        "./result/binary_optical_flow_gray/data_aug",
        "./result/best_model_optical_flow_gray/data_aug",
        "./result/optical_flow_gray/moving_average",

        "./data/processed_optical_flow_mog2_gray",
        "./result/benchmark/optical_flow_mog2_gray",
        "./result/exe_time/optical_flow_mog2_gray",
        "./result/binary_optical_flow_mog2_gray/data_aug",
        "./result/best_model_optical_flow_mog2_gray/data_aug",
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
    exe_time_dir = "./result/exe_time/optical_flow_gray"
    benchmark_dir = "./result/benchmark/optical_flow_gray"
    benchmark_dir_mog2 = "./result/benchmark/optical_flow_mog2_gray"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    # Common parameters
    batch_size = 32
    epochs = 50
    
    xlss_path = "./resources/GasVid_Logging_File.xlsx"

    ma_dir = "./data/processed_optical_flow_gray"
    mog2_dir = "./data/processed_optical_flow_mog2_gray"
    # Define base parameters
    base_params = {
        'gaussian_kernel_size':(3,3),
        'frame_sizes': [15],
        'segment_length_sec': 180,
        'total_segments': 8,
        'remove_start_sec': 15,
        'remove_end_sec': 5,
        # Farneback parameters
        'pyr_scale': 0.5,
        'levels': 3,
        'winsize': 21,
        'iterations': 3,
        'poly_n': 5,
        'poly_sigma': 1.2,
        # Enhanced thresholds
        'mmt_threshold': 1.2,
        'pat_threshold': 30,
        'min_movement_threshold': 10
    }

    # Moving Average specific parameters
    ma_params = {
        **base_params,
        'background_window': 210
    }

    # MOG2 specific parameters
    mog2_params = {
        **base_params,
        'history': 210,
        'var_threshold': 16,
        'detect_shadows': True,
        'K': 3,
        'alpha': 0.08,
        'initial_variance': 15.0,
        'min_variance': 10.0,
        'weight_threshold': 0.9
    }

    # Process videos with different methods
    print("\n====== Processing Videos ======")
    
    # Initialize benchmark results for each method
    optical_flow_ma_results = []  # Optical Flow + Moving Average
    optical_flow_mog2_results = [] # Optical Flow + MOG2
    
    # for single_video_path in video_paths:
    #     video_name = os.path.basename(single_video_path)
        
    #     # # 1. Process with Optical Flow + Moving Average
    #     # print(f"\nProcessing with Optical Flow + Moving Average: {video_name}")
    #     # start_time = time.time()
    #     # process_video(
    #     #     single_video_path, 
    #     #     xlss_path, 
    #     #     ma_dir,
    #     #     **ma_params
    #     # )
    #     # processing_time = time.time() - start_time
        
    #     # optical_flow_ma_results.append({
    #     #     'video_name': video_name,
    #     #     'size_mb': os.path.getsize(single_video_path) / (1024*1024),
    #     #     'processing_time_sec': processing_time,
    #     #     'processing_speed_mbs': os.path.getsize(single_video_path) / (1024*1024) / processing_time
    #     # })
        
    #     # 2. Process with Optical Flow + MOG2
    #     print(f"\nProcessing with Optical Flow + our own MOG2: {video_name}")
    #     start_time = time.time()
    #     MOG2_process_video(
    #         single_video_path, 
    #         xlss_path, 
    #         mog2_dir,
    #         **mog2_params
    #     )
    #     processing_time = time.time() - start_time
        
    #     optical_flow_mog2_results.append({
    #         'video_name': video_name,
    #         'size_mb': os.path.getsize(single_video_path) / (1024*1024),
    #         'processing_time_sec': processing_time,
    #         'processing_speed_mbs': os.path.getsize(single_video_path) / (1024*1024) / processing_time
    #     })
        
    #     # Save benchmark results after each video
    #     # save_preprocessing_benchmark('OpticalFlow_MovingAverage', optical_flow_ma_results, benchmark_dir)
    #     save_preprocessing_benchmark('OpticalFlow_MOG2', optical_flow_mog2_results, benchmark_dir_mog2)

    # Train and test both methods
    methods = [
        # {
        #     'name': 'OpticalFlow_MovingAverage with gray scale',
        #     'train_dir': "./data/processed_optical_flow_gray/train",
        #     'val_dir': "./data/processed_optical_flow_gray/val",
        #     'test_dir': "./data/processed_optical_flow_gray/test",
        #     'model_path': "./result/binary_optical_flow_gray/data_aug/cnn_3d_binaryAllLeak.keras",
        #     'best_model_path': "./result/best_model_optical_flow_gray/data_aug/cnn_3d_binaryAllLeak.keras",
        #     'output_dir': "./result/optical_flow_gray/moving_average"
        # }
        # ,
        {
            'name': 'OpticalFlow_MOG2 with gray scale',
            'train_dir': "./data/processed_optical_flow_mog2_gray/train",
            'val_dir': "./data/processed_optical_flow_mog2_gray/val",
            'test_dir': "./data/processed_optical_flow_mog2_gray/test",
            'model_path': "./result/binary_optical_flow_mog2_gray/data_aug/cnn_3d_binaryAllLeak_gray.keras",
            'best_model_path': "./result/best_model_optical_flow_mog2_gray/data_aug/cnn_3d_binaryAllLeak_gray.keras",
            'output_dir': "./result/optical_flow_mog2_gray"
        }
    ]

    for method in methods:
        try:
            # Training
            print(f"\n[INFO] Starting {method['name']} method training...")
            train_start = time.time()
            train_binary(
                method['train_dir'],
                method['val_dir'],
                method['model_path'],
                method['best_model_path'],
                method['output_dir'],
                batch_size,
                epochs
            )
            train_duration = time.time() - train_start
            save_step_time(method['name'], "Training", train_duration, timestamp, exe_time_dir)
        except Exception as e:
            print(f"[ERROR] {method['name']} training failed: {str(e)}")

        try:
            # Testing
            print(f"\n[INFO] Starting {method['name']} method testing...")
            test_start = time.time()
            test_binary(
                test_dir=method['test_dir'],
                model_path=method['best_model_path'],
                batch_size=batch_size,
                output_base_dir=method['output_dir'],
                method_name=method['name'].lower()
            )
            test_duration = time.time() - test_start
            save_step_time(method['name'], "Testing", test_duration, timestamp, exe_time_dir)
        except Exception as e:
            print(f"[ERROR] {method['name']} testing failed: {str(e)}")

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
