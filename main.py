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

def print_system_summary(args):
    """Print a summary of the current system settings"""
    print("\n" + "="*50)
    print("GAS LEAK DETECTION SYSTEM SETTINGS")
    print("="*50)
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Resolutions: {', '.join(args.resolutions)}")
    print(f"Training Modes: {', '.join(args.train_modes)}")
    print(f"Testing Modes: {', '.join(args.test_modes)}")
    print(f"Distance Filter: {args.distance_filter}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Result Directory: {args.result_dir}")
    
    # Skip flags
    skipped = []
    if args.skip_preprocessing: skipped.append("Preprocessing")
    if args.skip_training: skipped.append("Training")
    if args.skip_testing: skipped.append("Testing")
    if skipped:
        print(f"Skipped Steps: {', '.join(skipped)}")
    print("="*50 + "\n")

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
    
    # Initialize timing variables at the beginning
    train_start_time = time.time()  # Default initialization
    test_start_time = time.time()   # Default initialization
    perform_training = not args.skip_training
    perform_testing = not args.skip_testing
    
    # Create timestamp at the beginning - just the date (YYYYMMDD)
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Print system settings summary
    print_system_summary(args)
    
    # Set mode flags based on argument
    three_class_mode = 'three_class' in args.train_modes or 'three_class' in args.test_modes
    eight_class_mode = 'eight_class' in args.train_modes or 'eight_class' in args.test_modes
    
    # Determine mode-specific directory name
    mode_dir_suffix = "three_class" if three_class_mode else "binary"
    
    # Map selected methods to their full names and directory suffixes
    method_configs = {
        'ma': {
            'name': 'OpticalFlow_MovingAverage_with_grayscale',
            'data_dir': f"/mnt/d/processsed_data/processed_optical_flow_gray/",
            'benchmark_dir': f"{args.result_dir}/benchmark/optical_flow_gray",
            'exe_time_dir': f"{args.result_dir}/exe_time/optical_flow_gray",
            'model_dir': f"{args.result_dir}/{mode_dir_suffix}_optical_flow_gray/data_aug",
            'best_model_dir': f"{args.result_dir}/best_model_optical_flow_gray/{mode_dir_suffix}/data_aug",
            'output_dir': f"{args.result_dir}/optical_flow_gray/{mode_dir_suffix}",
            'process_func': process_video,
            'params_key': 'ma_params'
        },
        'mog2': {
            'name': 'OpticalFlow_MOG2_with_grayscale',
            'data_dir': f"/mnt/d/processsed_data/processed_optical_flow_mog2_gray/",
            'benchmark_dir': f"{args.result_dir}/benchmark/optical_flow_mog2_gray",
            'exe_time_dir': f"{args.result_dir}/exe_time/optical_flow_mog2_gray",
            'model_dir': f"{args.result_dir}/{mode_dir_suffix}_optical_flow_mog2_gray/data_aug",
            'best_model_dir': f"{args.result_dir}/best_model_optical_flow_mog2_gray/data_aug",
            'output_dir': f"{args.result_dir}/optical_flow_mog2_gray/{mode_dir_suffix}",
            'process_func': MOG2_process_video,
            'params_key': 'mog2_params'
        },
        'runavg': {
            'name': 'OpticalFlow_RunningAverage_with_grayscale',
            'data_dir': f"{args.data_dir}/processed_optical_flow_runavg_gray",
            'benchmark_dir': f"{args.result_dir}/benchmark_new/optical_flow_runavg_gray_new",
            'exe_time_dir': f"{args.result_dir}/exe_time_new/optical_flow_runavg_gray_new",
            'model_dir': f"{args.result_dir}/{mode_dir_suffix}_optical_flow_runavg_gray_new/data_aug",
            'best_model_dir':f"{args.result_dir}/best_model_optical_flow_runavg_gray_new/{mode_dir_suffix}/data_aug",
            'output_dir': f"{args.result_dir}/optical_flow_runavg_gray_new/{mode_dir_suffix}",
            'process_func': runavg_process_video,
            'params_key': 'runavg_params'
        }
    }

    selected_methods = [method_configs[method] for method in args.methods]
    
    # Create all necessary directories for selected methods
    required_dirs = []
    for method in selected_methods:
        # Add data directory
        required_dirs.append(method['data_dir'])
        
        # Add result directories
        required_dirs.append(method['benchmark_dir'])
        required_dirs.append(method['exe_time_dir'])
        # Add model directories for selected training modes
        for mode in args.train_modes:
            mode_model_dir = method['model_dir'].replace(mode_dir_suffix, mode)
            mode_best_model_dir = method['best_model_dir'].replace(mode_dir_suffix, mode)
            required_dirs.append(mode_model_dir)
            required_dirs.append(mode_best_model_dir)
        
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

    # Training phase
    if perform_training:
        print("\n\n" + "="*50)
        print("TRAINING PHASE")
        print("="*50 + "\n")
        
        # Track execution time
        train_start_time = time.time()  # Reset this to the actual training start time
        
        for resolution_str in args.resolutions:
            # Parse resolution
            target_height, target_width = map(int, resolution_str.split('x'))
            resolution_str = f"{target_height}x{target_width}"
            print(f"\n====== Processing with resolution {resolution_str} ======")
            
            # Train each selected classification mode
            for train_mode in args.train_modes:
                # Train using each method 
                # Set mode flags based on current mode
                three_class_mode = train_mode == 'three_class'
                eight_class_mode = train_mode == 'eight_class'
                
                methods_for_training = []
                for method in selected_methods:
                    # Set model type prefix based on classification mode
                    if eight_class_mode:
                        model_type_prefix = "cnn_3d_eight_class"
                    elif three_class_mode:
                        model_type_prefix = "cnn_3d_three_class"
                    else:
                        model_type_prefix = "cnn_3d_binary"

                    # Construct unique paths for each model using the timestamp from the beginning
                    model_path = f"{method['model_dir']}/{model_type_prefix}_{method['name']}_{resolution_str}_{timestamp}.keras"
                    best_model_path = f"{method['best_model_dir']}/{model_type_prefix}_{method['name']}_{resolution_str}_{timestamp}.keras"
                    
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
                            print(f"\n[INFO] Starting {train_mode} training for {method['name']} method with resolution {resolution_str}...")
                            # Track execution time for this specific training
                            method_train_start = time.time()
                            
                            # Train the model with appropriate mode flags
                            train_model(
                                method['train_dir'],
                                method['val_dir'],
                                model_path,
                                best_model_path,
                                method['output_dir'],
                                args.batch_size,
                                args.epochs,
                                target_height=target_height,
                                target_width=target_width,
                                three_class_mode=three_class_mode,
                                eight_class_mode=eight_class_mode, 
                                distance_filter=distance_filter
                            )
                            
                            # Calculate training duration for this method and save timing info
                            method_train_duration = time.time() - method_train_start
                            save_step_time(
                                method['name'], 
                                f"{train_mode}_Training_{resolution_str}", 
                                method_train_duration, 
                                timestamp, 
                                method['exe_time_dir']
                            )
                            
                            # Store model path for later testing
                            method[f'{train_mode}_best_model_path'] = best_model_path
                        
                        except Exception as e:
                            print(f"[ERROR] {method['name']} {train_mode} training failed: {str(e)}")
                            import traceback
                            traceback.print_exc()
                    
                # Clear GPU memory between training runs
                clear_gpu_memory()

    # Testing phase
    if perform_testing:
        print("\n\n" + "="*50)
        print("TESTING PHASE")
        print("="*50 + "\n")
        
        # Track execution time
        test_start_time = time.time()  # Reset this to the actual testing start time
        
        for resolution_str in args.resolutions:
            # Parse resolution
            target_height, target_width = map(int, resolution_str.split('x'))
            
            # Test each selected classification mode
            for test_mode in args.test_modes:
                # Set mode flags based on current mode
                three_class_mode = test_mode == 'three_class'
                eight_class_mode = test_mode == 'eight_class'
                
                
                # Set up methods for testing
                methods_for_testing = []
                # Test using each method 
                for method in selected_methods:
                    
                    # Set model type prefix based on classification mode
                    if eight_class_mode:
                        model_type_prefix = "cnn_3d_eight_class"
                    elif three_class_mode:
                        model_type_prefix = "cnn_3d_three_class"
                    else:
                        model_type_prefix = "cnn_3d_binary"
                        
                    # Construct unique paths for each model using the timestamp from the beginning
                    best_model_path = f"{method['best_model_dir']}/{model_type_prefix}_{method['name']}_{resolution_str}_{timestamp}.keras"
                    
                    methods_for_testing.append({
                        'name': method['name'],
                        'test_dir': f"{method['data_dir']}/test",
                        'best_model_path': best_model_path,
                        'output_dir': method['output_dir'],
                        'exe_time_dir': method['exe_time_dir']
                    })

                for method in methods_for_testing:
                    try:
                        print(f"\n[INFO] Starting {test_mode} testing for {method['name']} method with resolution {resolution_str}...")
                        # Track execution time for this specific testing
                        method_test_start = time.time()
                        
                        # Test the model with appropriate mode flags
                        test_model(
                            test_dir=method['test_dir'],
                            model_path=method['best_model_path'],
                            batch_size=args.batch_size,
                            output_base_dir=method['output_dir'],
                            method_name=method['name'].lower(),
                            target_height=target_height,
                            target_width=target_width,
                            three_class_mode=three_class_mode,
                            eight_class_mode=eight_class_mode,
                            distance_filter=distance_filter
                        )
                        
                        # Calculate testing duration for this method and save timing info
                        method_test_duration = time.time() - method_test_start
                        save_step_time(
                            method['name'], 
                            f"{test_mode}_Testing_{resolution_str}", 
                            method_test_duration, 
                            timestamp, 
                            method['exe_time_dir']
                        )
                        
                    except Exception as e:
                        print(f"[ERROR] {method['name']} {test_mode} testing failed: {str(e)}")
                        import traceback
                        traceback.print_exc()
                    
                    # Clear GPU memory between testing runs
                    clear_gpu_memory()

    # Calculate total training and testing duration only if those phases were performed
    if perform_training:
        total_train_duration = time.time() - train_start_time
        save_step_time(
            "Total", 
            "Training", 
            total_train_duration, 
            timestamp, 
            selected_methods[0]['exe_time_dir']
        )
    
    if perform_testing:
        total_test_duration = time.time() - test_start_time
        save_step_time(
            "Total", 
            "Testing", 
            total_test_duration, 
            timestamp, 
            selected_methods[0]['exe_time_dir']
        )

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
