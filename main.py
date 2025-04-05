import os
import argparse
from src.preprocess.mog2_preprocess import MOG2_process_video
from src.preprocess.runavg_preprocess import runavg_process_video
from src.preprocess.preprocess import process_video
from src.train.train import train_model
from src.test.test import test_model
import tensorflow as tf
import time
from datetime import datetime

# Add at the beginning of your script
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Gas Leak Detection System')
    parser.add_argument('--methods', nargs='+', choices=['ma', 'mog2', 'runavg'], default=['runavg'],
                        help='Background subtraction methods to use: ma (Moving Average), mog2 (MOG2), runavg (Running Average)')
    parser.add_argument('--skip-training', action='store_true', help='Skip the training phase')
    parser.add_argument('--skip-testing', action='store_true', help='Skip the testing phase')
    parser.add_argument('--skip-preprocessing', action='store_true', help='Skip the preprocessing phase')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training and testing')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--data-dir', type=str, default='./data', help='Base directory for data')
    parser.add_argument('--result-dir', type=str, default='./result', help='Base directory for results')
    parser.add_argument('--resolutions', nargs='+', choices=['240x320', '120x160', '60x80'], default=['240x320'],
                        help='Image resolutions to use for training and testing')
    parser.add_argument('--train-modes', nargs='+', choices=['binary', 'three_class', 'eight_class'], 
                        default=['binary'], help='Classification modes to train')
    parser.add_argument('--test-modes', nargs='+', choices=['binary', 'three_class', 'eight_class'], 
                        default=['binary'], help='Classification modes to test')
    parser.add_argument('--distance-filter', choices=['46', '69', 'all'], default='all',
                        help='Filter by imaging distance: 46 for 4.6m, 69 for 6.9m, all for no filtering')
    args = parser.parse_args()

    # Set distance filter (None if 'all' is selected)
    distance_filter = None if args.distance_filter == 'all' else args.distance_filter
    
    # Map method keys to their configurations
    method_configs = {
        'ma': {
            'name': 'OpticalFlow_MovingAverage_with_grayscale',
            'data_dir': f"/mnt/d/processsed_data/processed_optical_flow_gray/",
            'benchmark_dir': f"{args.result_dir}/benchmark/optical_flow_gray",
            'exe_time_dir': f"{args.result_dir}/exe_time/optical_flow_gray",
            'model_dirs': {
                'binary': f"{args.result_dir}/binary_optical_flow_gray/data_aug",
                'three_class': f"{args.result_dir}/three_class_optical_flow_gray/data_aug",
                'eight_class': f"{args.result_dir}/eight_class_optical_flow_gray/data_aug"
            },
            'best_model_dirs': {
                'binary': f"{args.result_dir}/best_model_optical_flow_gray/binary/data_aug",
                'three_class': f"{args.result_dir}/best_model_optical_flow_gray/three_class/data_aug",
                'eight_class': f"{args.result_dir}/best_model_optical_flow_gray/eight_class/data_aug"
            },
            'output_dir': f"{args.result_dir}/optical_flow_gray",
            'process_func': process_video,
            'params_key': 'ma_params'
        },
        'mog2': {
            'name': 'OpticalFlow_MOG2_with_grayscale',
            'data_dir': f"/mnt/d/processsed_data/processed_optical_flow_mog2_gray/",
            'benchmark_dir': f"{args.result_dir}/benchmark/optical_flow_mog2_gray",
            'exe_time_dir': f"{args.result_dir}/exe_time/optical_flow_mog2_gray",
            'model_dirs': {
                'binary': f"{args.result_dir}/binary_optical_flow_mog2_gray/data_aug",
                'three_class': f"{args.result_dir}/three_class_optical_flow_mog2_gray/data_aug",
                'eight_class': f"{args.result_dir}/eight_class_optical_flow_mog2_gray/data_aug"
            },
            'best_model_dirs': {
                'binary': f"{args.result_dir}/best_model_optical_flow_mog2_gray/binary/data_aug",
                'three_class': f"{args.result_dir}/best_model_optical_flow_mog2_gray/three_class/data_aug",
                'eight_class': f"{args.result_dir}/best_model_optical_flow_mog2_gray/eight_class/data_aug"
            },
            'output_dir': f"{args.result_dir}/optical_flow_mog2_gray",
            'process_func': MOG2_process_video,
            'params_key': 'mog2_params'
        },
        'runavg': {
            'name': 'OpticalFlow_RunningAverage_with_grayscale',
            'data_dir': f"{args.data_dir}/processed_optical_flow_runavg_gray",
            'benchmark_dir': f"{args.result_dir}/benchmark_new/optical_flow_runavg_gray_new",
            'exe_time_dir': f"{args.result_dir}/exe_time_new/optical_flow_runavg_gray_new",
            'model_dirs': {
                'binary': f"{args.result_dir}/binary_optical_flow_runavg_gray_new/data_aug",
                'three_class': f"{args.result_dir}/three_class_optical_flow_runavg_gray_new/data_aug",
                'eight_class': f"{args.result_dir}/eight_class_optical_flow_runavg_gray_new/data_aug"
            },
            'best_model_dirs': {
                'binary': f"{args.result_dir}/best_model_optical_flow_runavg_gray_new/binary/data_aug",
                'three_class': f"{args.result_dir}/best_model_optical_flow_runavg_gray_new/three_class/data_aug",
                'eight_class': f"{args.result_dir}/best_model_optical_flow_runavg_gray_new/eight_class/data_aug"
            },
            'output_dir': f"{args.result_dir}/optical_flow_runavg_gray_new",
            'process_func': runavg_process_video,
            'params_key': 'runavg_params'
        }
    }

    selected_methods = [method_configs[method] for method in args.methods]
    
    # Create all necessary directories for selected methods and modes
    required_dirs = []
    for method in selected_methods:
        # Add data directory
        required_dirs.append(method['data_dir'])
        
        # Add result directories
        required_dirs.append(method['benchmark_dir'])
        required_dirs.append(method['exe_time_dir'])
        
        # Add model directories for selected training modes
        for mode in args.train_modes:
            required_dirs.append(method['model_dirs'][mode])
            required_dirs.append(method['best_model_dirs'][mode])
        
        # Add output directory
        required_dirs.append(method['output_dir'])
    
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"[INFO] Ensuring directory exists: {dir_path}")

    # # Gather all .mp4 files from ./data/raw/
    video_dir = f'{args.data_dir}/raw/'
    video_paths = [
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.endswith('.mp4')
    ]

    # Set common timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Define base parameters
    base_params = {
        'gaussian_kernel_size': (3, 3),
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

    # Running Average specific parameters
    runavg_params = {
        **base_params,
        'alpha': 0.01
    }

    # Parameter dictionary
    params_dict = {
        'ma_params': ma_params,
        'mog2_params': mog2_params,
        'runavg_params': runavg_params
    }

    # Path to Excel file with video metadata
    xlss_path = "./resources/GasVid_Logging_File.xlsx"

    # Process videos with selected methods (if not skipped)
    if not args.skip_preprocessing:
        print("\n====== Processing Videos ======")
        
        # Initialize benchmark results for each selected method
        benchmark_results = {method['name']: [] for method in selected_methods}
        
        for single_video_path in video_paths:
            video_name = os.path.basename(single_video_path)
            
            for method in selected_methods:
                print(f"\nProcessing with {method['name']}: {video_name}")
                
                start_time = time.time()
                method['process_func'](
                    single_video_path, 
                    xlss_path, 
                    method['data_dir'],
                    **params_dict[method['params_key']]
                )
                processing_time = time.time() - start_time
                
                benchmark_results[method['name']].append({
                    'video_name': video_name,
                    'size_mb': os.path.getsize(single_video_path) / (1024*1024),
                    'processing_time_sec': processing_time,
                    'processing_speed_mbs': os.path.getsize(single_video_path) / (1024*1024) / processing_time
                })
                
                # Save benchmark results after each video
                save_preprocessing_benchmark(
                    method['name'], 
                    benchmark_results[method['name']], 
                    method['benchmark_dir']
                )

    # Loop through all resolutions
    for resolution in args.resolutions:
        # Parse resolution dimensions
        target_height, target_width = map(int, resolution.split('x'))
        resolution_str = f"{target_height}x{target_width}"
        print(f"\n====== Processing with resolution {resolution_str} ======")
        
        # Training phase (if not skipped)
        if not args.skip_training:
            for train_mode in args.train_modes:
                print(f"\n=== Training {train_mode} classification models ===")
                
                # Configure training based on mode
                is_three_class = train_mode == 'three_class'
                is_eight_class = train_mode == 'eight_class'
                
                # Train selected methods for current mode
                methods_for_training = []
                for method in selected_methods:
                    # Create model type prefix and paths
                    model_type_prefix = f"cnn_3d_{train_mode}"
                    model_path = f"{method['model_dirs'][train_mode]}/{model_type_prefix}_{method['name']}_{resolution_str}_{timestamp}.keras"
                    best_model_path = f"{method['best_model_dirs'][train_mode]}/{model_type_prefix}_{method['name']}_{resolution_str}_{timestamp}.keras"
                    
                    methods_for_training.append({
                        'name': method['name'],
                        'train_dir': f"{method['data_dir']}/train",
                        'val_dir': f"{method['data_dir']}/val",
                        'test_dir': f"{method['data_dir']}/test",
                        'model_path': model_path,
                        'best_model_path': best_model_path,
                        'output_dir': method['output_dir'],
                        'exe_time_dir': method['exe_time_dir']
                    })

                for method in methods_for_training:
                    try:
                        # Training
                        print(f"\n[INFO] Starting {method['name']} method {train_mode} training with resolution {resolution_str}...")
                        train_start = time.time()
                        train_model(
                            method['train_dir'],
                            method['val_dir'],
                            method['model_path'],
                            method['best_model_path'],
                            method['output_dir'],
                            args.batch_size,
                            args.epochs,
                            target_height=target_height,
                            target_width=target_width,
                            three_class_mode=is_three_class,
                            eight_class_mode=is_eight_class,
                            distance_filter=distance_filter
                        )
                        train_duration = time.time() - train_start
                        save_step_time(method['name'], f"{train_mode}_Training_{resolution_str}", train_duration, timestamp, method['exe_time_dir'])
                    except Exception as e:
                        print(f"[ERROR] {method['name']} {train_mode} training failed: {str(e)}")
        
        # Testing phase (if not skipped)
        if not args.skip_testing:
            for test_mode in args.test_modes:
                print(f"\n=== Testing {test_mode} classification models ===")
                
                # Configure testing based on mode
                is_three_class = test_mode == 'three_class'
                is_eight_class = test_mode == 'eight_class'
                
                # Set up methods for testing
                methods_for_testing = []
                for method in selected_methods:
                    # Create model type prefix and path
                    model_type_prefix = f"cnn_3d_{test_mode}"
                    best_model_path = f"{method['best_model_dirs'][test_mode]}/{model_type_prefix}_{method['name']}_{resolution_str}_{timestamp}.keras"
                    
                    methods_for_testing.append({
                        'name': method['name'],
                        'test_dir': f"{method['data_dir']}/test",
                        'best_model_path': best_model_path,
                        'output_dir': method['output_dir'],
                        'exe_time_dir': method['exe_time_dir']
                    })
                    
                for method in methods_for_testing:
                    try:
                        # Testing
                        print(f"\n[INFO] Starting {method['name']} method {test_mode} testing with resolution {resolution_str}...")
                        test_start = time.time()
                        test_model(
                            test_dir=method['test_dir'],
                            model_path=method['best_model_path'],
                            batch_size=args.batch_size,
                            output_base_dir=method['output_dir'],
                            method_name=method['name'].lower(),
                            target_height=target_height,
                            target_width=target_width,
                            three_class_mode=is_three_class,
                            eight_class_mode=is_eight_class,
                            distance_filter=distance_filter
                        )
                        test_duration = time.time() - test_start
                        save_step_time(method['name'], f"{test_mode}_Testing_{resolution_str}", test_duration, timestamp, method['exe_time_dir'])
                    except Exception as e:
                        print(f"[ERROR] {method['name']} {test_mode} testing failed: {str(e)}")

   # Clear GPU memory between runs
def clear_gpu_memory():
       import tensorflow as tf
       tf.keras.backend.clear_session()
       import gc
       gc.collect()
   
   # Call this between major operations

if __name__ == "__main__":
    # Disable XLA and mixed precision
    clear_gpu_memory()
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
