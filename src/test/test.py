import os
import numpy as np
import tensorflow as tf
from src.model.model import cnn_3d_model
from src.loader.loader import DataGenerator, DataGeneratorThreeClass
import math
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time 
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import traceback
import platform
import psutil

def save_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Save confusion matrix as an image file."""
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f'confusion_matrix_{timestamp}.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Confusion matrix saved to {output_path}")
    except Exception as e:
        print(f"Error saving confusion matrix: {e}")
        traceback.print_exc()

def save_classification_report(y_true, y_pred, output_dir):
    try:
        report = classification_report(y_true, y_pred, digits=4)
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f'classification_report_{timestamp}.txt')
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Classification report saved to {output_path}")
    except Exception as e:
        print(f"Error saving classification report: {e}")
        traceback.print_exc()

def evaluate_three_class(model_path, test_dir, batch_size, output_dir, target_height=240, target_width=320):
    """
    Evaluate three-class classification: small (0-2), medium (3-5), large (6-7) leaks.
    """
    print("[INFO] Evaluating three-class leak size classification...")
    
    try:
        # Directly use the model path provided from main.py
        print(f"[INFO] Using model from: {model_path}")
        
        # Check if the model exists
        if not os.path.exists(model_path):
            print(f"[ERROR] Model not found at: {model_path}")
            raise FileNotFoundError(f"Model not found at: {model_path}")
            
        # Define class names for reporting
        class_names = ['Small Leak (0-2)', 'Medium Leak (3-5)', 'Large Leak (6-7)']
        
        # Load model
        model = cnn_3d_model(input_shape=(15, target_height, target_width, 1), num_classes=3)
        model.load_weights(model_path)

        # Warmup a dummy input
        print("[INFO] Warming up the model with a dummy input")
        dummy_input = np.random.rand(1, 15, target_height, target_width, 1).astype(np.float32)
        _ = model.predict(dummy_input, verbose=1)
        
        # Create data generator for three-class mode using the new DataGeneratorThreeClass
        test_gen = DataGeneratorThreeClass(
            data_dir=test_dir,
            batch_size=batch_size,
            shuffle=False,
            balance_classes=False,
            training=False,
            resize=True,
            target_height=target_height,
            target_width=target_width,
            distance_filter=None  # No distance filtering for general evaluation
        )
        
        if len(test_gen.filepaths) == 0:
            print("[ERROR] No test data found for three-class evaluation")
            return {
                'accuracy': 0.0,
                'class_report': None,
                'confusion_matrix': None,
                'y_true': [],
                'y_pred': []
            }
        
        # Calculate number of batches
        num_batches = len(test_gen)
        print(f"[INFO] Running predictions on {num_batches} batches...")
        
        # Store predictions and true labels
        y_true = []
        y_pred = []
        
        # Track time for performance monitoring
        start_time = time.time()
        
        # Process batches
        for batch_idx in range(num_batches):
            if batch_idx % 10 == 0:
                print(f"[INFO] Processing batch {batch_idx+1}/{num_batches}")
            
            X_batch, y_batch = test_gen[batch_idx]
            if len(X_batch) == 0:
                continue
                
            # Make predictions
            batch_pred = model.predict(X_batch, verbose=0)
            batch_pred_classes = np.argmax(batch_pred, axis=1)
            
            # Store results
            y_true.extend(y_batch)
            y_pred.extend(batch_pred_classes)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        elapsed_time = time.time() - start_time
        
        print(f"[INFO] Three-class evaluation completed in {elapsed_time:.2f} seconds")
        print(f"[INFO] Accuracy: {accuracy:.4f}")
        
        # Save confusion matrix and classification report
        save_confusion_matrix(y_true, y_pred, class_names, output_dir)
        save_classification_report(y_true, y_pred, output_dir)
        
        # Save detailed results
        results_path = os.path.join(output_dir, f'three_class_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        with open(results_path, 'w') as f:
            f.write(f"Three-Class Leak Size Classification Results\n")
            f.write(f"==========================================\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Test Data: {test_dir}\n")
            f.write(f"Resolution: {target_height}x{target_width}\n\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
            f.write(f"Class Distribution:\n")
            f.write(f"  Small Leak (0-2): {y_true.count(0)} samples\n")
            f.write(f"  Medium Leak (3-5): {y_true.count(1)} samples\n")
            f.write(f"  Large Leak (6-7): {y_true.count(2)} samples\n\n")
            f.write(f"Evaluation Time: {elapsed_time:.2f} seconds\n")
        
        print(f"[INFO] Detailed results saved to {results_path}")
        
        return {
            'accuracy': accuracy,
            'y_true': y_true,
            'y_pred': y_pred,
            'class_report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
    except Exception as e:
        print(f"[ERROR] Three-class evaluation failed: {e}")
        traceback.print_exc()
        return {
            'accuracy': 0.0,
            'class_report': None,
            'confusion_matrix': None,
            'y_true': [],
            'y_pred': []
        }

def evaluate_all_leak_vs_no_leak(model_path, test_dir, batch_size, output_dir, target_height=240, target_width=320):
    """
    Evaluate binary classification performance for VideoGasNet (all leak vs. no leak).
    """
    print("[INFO] Evaluating all leak vs no leak binary classification...")
    start_time = time.time()
    
    try:
        # Directly use the model path provided from main.py
        print(f"[INFO] Using model from: {model_path}")
        
        # Check if the model exists
        if not os.path.exists(model_path):
            print(f"[ERROR] Model not found at: {model_path}")
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        # Load model
        model = cnn_3d_model(input_shape=(15, target_height, target_width, 1), num_classes=2)
        model.load_weights(model_path)

        # Warmup a dummy input
        print("[INFO] Warming up a dummy input")
        dummy_input = np.random.rand(1, 15, target_height, target_width, 1).astype(np.float32)
        _ = model.predict(dummy_input, verbose=1)
        
        # Perform another warmup for more stable timing
        _ = model.predict(dummy_input, verbose=0)

        test_gen = DataGenerator(
            data_dir=test_dir,
            batch_size=batch_size,
            shuffle=False,
            binary_all_leak=True,
            balance_classes=False,
            training=False,
            resize=True,
            target_height=target_height,
            target_width=target_width
        )   

        # Count class distribution
        class_counts = {}
        for _, label in test_gen.filepaths:
            if test_gen.binary_all_leak:
                binary_label = 1 if label > 0 else 0
            else:
                binary_label = label
            
            class_counts[binary_label] = class_counts.get(binary_label, 0) + 1
        
        print("[INFO] Class distribution:")
        print(f"  No leak (0): {class_counts.get(0, 0)} samples")
        print(f"  Leak (1-7): {class_counts.get(1, 0)} samples")
        print(f"[INFO] Found {len(test_gen.filepaths)} files to process")
        
        total_batches = len(test_gen)
        print(f"[INFO] Total batches to process: {total_batches}")

        # Verify generator output shape before prediction loop
        try:
            test_batch, test_labels = test_gen[0]
            print(f"\n[SHAPE TEST] First batch shape: {test_batch.shape}")
            print(f"Sample data range: {test_batch.min()} to {test_batch.max()}")
        except IndexError:
            print("[ERROR] Test generator is empty!")
            return

        y_true, y_pred = [], []
        print("[INFO] Testing Start")
        processed_samples = 0
        start_time = time.time()
        
        # Inference time tracking (in milliseconds to match test_quan.py)
        total_inference_time = 0
        inference_samples = 0
        all_inference_times_ms = []

        for batch_idx, (X_batch, y_batch) in enumerate(test_gen):
            if len(X_batch) == 0:
                print(f"[WARNING] Empty batch detected at batch {batch_idx}. Skipping this batch.")
                break

            processed_samples += len(X_batch)
            elapsed = time.time() - start_time
            samples_per_sec = processed_samples / elapsed if elapsed > 0 else 0
            
            print(f"\nBatch {batch_idx+1}/{total_batches} | "
                  f"Samples: {processed_samples}/{len(test_gen.filepaths)} | "
                  f"Speed: {samples_per_sec:.1f} samples/sec | "
                  f"Elapsed: {elapsed:.1f}s")

            # Measure inference time specifically (in milliseconds)
            inference_start = time.time()
            preds = model.predict(X_batch, verbose=0)
            inference_time = time.time() - inference_start
            inference_time_ms = inference_time * 1000  # Convert to milliseconds
            
            # Track inference statistics
            total_inference_time += inference_time_ms
            inference_samples += len(X_batch)
            all_inference_times_ms.extend([inference_time_ms / len(X_batch)] * len(X_batch))
            
            # Per sample in milliseconds (matching test_quan.py)
            per_sample_ms = inference_time_ms / len(X_batch)
            print(f"Batch inference time: {inference_time_ms:.2f} ms | "
                  f"Per sample: {per_sample_ms:.2f} ms")

            y_pred.extend(preds.argmax(axis=-1))
            y_true.extend(y_batch)
            
            # Explicitly collect garbage to reduce memory pressure
            gc.collect()
            
            start_time = time.time()

        # Calculate and save inference statistics
        avg_inference_time_per_sample_ms = total_inference_time / inference_samples if inference_samples > 0 else 0
        avg_inference_time_per_batch_ms = total_inference_time / (batch_idx + 1) if batch_idx >= 0 else 0
        throughput = 1000 / avg_inference_time_per_sample_ms if avg_inference_time_per_sample_ms > 0 else 0
        
        print(f"\n[INFERENCE STATS] Average inference time: {avg_inference_time_per_sample_ms:.2f} ms per sample")
        print(f"[INFERENCE STATS] Throughput: {throughput:.2f} samples/sec")
        
        # Save inference stats and metrics to file
        inference_stats_path = os.path.join(output_dir, f'inference_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        with open(inference_stats_path, 'w') as f:
            f.write(f"Total samples processed: {inference_samples}\n")
            f.write(f"Total inference time: {total_inference_time:.2f} ms\n")
            f.write(f"Average inference time per sample: {avg_inference_time_per_sample_ms:.2f} ms\n")
            f.write(f"Average inference time per batch (batch size={batch_size}): {avg_inference_time_per_batch_ms:.2f} ms\n")
            f.write(f"Throughput: {throughput:.2f} samples/sec\n")
            f.write("Performance Metrics for All Leaks vs No-Leak\n")
            f.write("==========================================\n\n")
            f.write(f"Total samples processed: {processed_samples}\n")
            f.write(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(classification_report(y_true, y_pred, digits=4))
            
            # Add hardware info
            try:
                f.write(f"\nHardware Information:\n")
                f.write(f"System: {platform.system()} {platform.release()}\n")
                f.write(f"CPU: {platform.processor()}\n")
                f.write(f"RAM: {psutil.virtual_memory().total // (1024**3)} GB\n")
                
                # GPU info for NVIDIA
                try:
                    import subprocess
                    gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True)
                    f.write(f"GPU: {gpu_info.decode('utf-8').strip()}\n")
                except:
                    f.write("GPU: Information not available\n")
            except:
                f.write("Hardware information not available\n")
        
        print(f"Inference statistics saved to {inference_stats_path}")

        # Save results
        class_names = ['No Leak', 'Leak']
        save_confusion_matrix(y_true, y_pred, class_names, output_dir)
        save_classification_report(y_true, y_pred, output_dir)          
        
        # Include acceleration method in terminal output
        print(f"[INFERENCE STATS] Processed {processed_samples} samples")
        print(f"[INFERENCE STATS] Average inference time: {avg_inference_time_per_sample_ms:.2f} ms/sample")
        print(f"[INFERENCE STATS] Throughput: {throughput:.2f} samples/sec")
        print(f"[INFERENCE STATS] Total evaluation time: {total_inference_time:.2f} seconds")
        print(f"[INFERENCE STATS] Effective batch size: {batch_size}")

        return y_true, y_pred
    
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        traceback.print_exc()
        return [], []

def evaluate_per_leak_class(model_path, test_dir, batch_size, output_dir, target_height=240, target_width=320):
    """
    Perform 10-fold testing for each leak class vs. no leak (0..7) using VideoGasNet.
    Only saves the final accuracy table for VideoGasNet.
    """
    print("[INFO] Evaluating leak classes using VideoGasNet (10-fold)...")
    
    try:
        # Directly use the model path provided from main.py
        print(f"[INFO] Using model from: {model_path}")
        
        # Check if the model exists
        if not os.path.exists(model_path):
            print(f"[ERROR] Model not found at: {model_path}")
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        # Load the trained VideoGasNet model (3D CNN)
        base_model = cnn_3d_model(input_shape=(15, target_height, target_width, 1), num_classes=2)
        base_model.load_weights(model_path)

        final_results = {i: [] for i in range(1, 8)}  # Store accuracies for each leak class
        inference_stats = {i: {'times': [], 'samples': 0} for i in range(1, 8)}  # Store inference times
        all_inference_times_ms = []  # Store all per-sample times in ms
        total_samples = 0
        overall_start = time.time()

        for leak_class in range(1, 8):
            print(f"\n--- 0 vs {leak_class} (10-fold) ---")
            
            # Create a balanced data generator for this leak class
            temp_gen = DataGenerator(
                data_dir=test_dir,
                batch_size=batch_size,
                shuffle=False,
                binary_all_leak=False,
                binary_pair=(0, leak_class),
                balance_classes=False,
                training=False,
                resize=True,
                target_height=target_height,
                target_width=target_width
            )

            if len(temp_gen.filepaths) == 0:
                print(f"Skipping 0 vs {leak_class} - no data found.")
                continue

            class_start_time = time.time()
            
            # 10-fold cross-validation
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            fold_accuracies = []
            
            for fold_i, (_, test_idx) in enumerate(kf.split(temp_gen.filepaths)):
                # Get fold-specific test data
                fold_files = [temp_gen.filepaths[i] for i in test_idx]
                
                fold_gen = DataGenerator(
                    data_dir=test_dir,
                    batch_size=batch_size,
                    shuffle=False,
                    binary_all_leak=False,
                    binary_pair=(0, leak_class),
                    explicit_files=fold_files,
                    balance_classes=False,
                    training=False,
                    resize=True,
                    target_height=target_height,
                    target_width=target_width
                )
                
                y_true, y_pred = [], []
                fold_inference_time_ms = 0
                fold_samples = 0
                
                for X_batch, y_batch in fold_gen:
                    if len(X_batch) == 0:
                        break
                    
                    # Measure inference time in milliseconds
                    inference_start = time.time()
                    preds = base_model.predict(X_batch, verbose=0)
                    inference_time = time.time() - inference_start
                    inference_time_ms = inference_time * 1000
                    
                    fold_inference_time_ms += inference_time_ms
                    fold_samples += len(X_batch)
                    
                    # Store per-sample times
                    per_sample_ms = inference_time_ms / len(X_batch)
                    all_inference_times_ms.extend([per_sample_ms] * len(X_batch))
                    
                    y_pred.extend(preds.argmax(axis=-1))
                    y_true.extend(y_batch)
                
                if len(y_true) > 0:
                    acc = accuracy_score(y_true, y_pred)
                    fold_accuracies.append(acc)
                    
                    # Store inference data
                    inference_stats[leak_class]['times'].append(fold_inference_time_ms)
                    inference_stats[leak_class]['samples'] += fold_samples
                    total_samples += fold_samples
                    
                    avg_time_ms = fold_inference_time_ms / fold_samples if fold_samples > 0 else 0
                    throughput = 1000 / avg_time_ms if avg_time_ms > 0 else 0
                    
                    print(f"Fold {fold_i+1}/10 | Accuracy: {acc:.4f} | "
                          f"Avg inference time: {avg_time_ms:.2f} ms/sample | "
                          f"Throughput: {throughput:.2f} samples/sec")
                
                # Explicit garbage collection
                gc.collect()
            
            # Store and print class summary
            final_results[leak_class] = fold_accuracies
            mean_acc = np.mean(fold_accuracies) if fold_accuracies else 0
            std_acc = np.std(fold_accuracies) if fold_accuracies else 0
            
            # Round the results appropriately
            mean_acc = round(mean_acc, 4)
            std_acc = round(std_acc, 3)
            
            class_time = time.time() - class_start_time
            print(f"\n0 vs {leak_class} | Mean Accuracy: {mean_acc:.4f} ± {std_acc:.3f}")
            print(f"Class evaluation time: {class_time:.2f} seconds")

        # Save inference statistics and overall metrics
        inference_stats_path = os.path.join(output_dir, f'inference_stats_per_class_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        # Calculate overall metrics
        total_time = time.time() - overall_start
        overall_avg_time = np.mean(all_inference_times_ms) if all_inference_times_ms else 0
        overall_throughput = 1000 / overall_avg_time if overall_avg_time > 0 else 0

        with open(inference_stats_path, 'w') as f:
            f.write("Inference Statistics Per Leak Class (10-fold testing)\n")
            f.write("===================================================\n\n")

            f.write("Overall 10-fold Cross-Validation Performance Metrics\n")
            f.write("=================================================\n\n")
            f.write(f"Total samples processed: {total_samples}\n")
            f.write(f"Average inference time: {overall_avg_time:.2f} ms/sample\n")
            f.write(f"Throughput: {overall_throughput:.2f} samples/sec\n")
            f.write(f"Total execution time: {total_time:.2f} seconds\n")
            f.write(f"Effective batch size: {batch_size}\n\n")
            
            f.write("Per-Class Mean Accuracies:\n")
            for leak_class in range(1, 8):
                accs = final_results[leak_class]
                if accs:
                    mean_acc = np.mean(accs)
                    std_acc = np.std(accs)
                    f.write(f"  Class 0 vs {leak_class}: {mean_acc:.4f} ± {std_acc:.4f}\n")
                else:
                    f.write(f"  Class 0 vs {leak_class}: No valid results\n")

            print(f"\n########### Inference statistics ###########")
            
            for leak_class in range(1, 8):
                stats = inference_stats[leak_class]
                if stats['samples'] > 0 and len(stats['times']) > 0:
                    total_time_ms = sum(stats['times'])
                    avg_time_per_sample_ms = total_time_ms / stats['samples']
                    throughput = 1000 / avg_time_per_sample_ms if avg_time_per_sample_ms > 0 else 0
                    
                    f.write(f"Class 0 vs {leak_class}:\n")
                    f.write(f"  Total samples: {stats['samples']}\n")
                    f.write(f"  Total inference time: {total_time_ms:.2f} ms\n")
                    f.write(f"  Average inference time per sample: {avg_time_per_sample_ms:.2f} ms\n")
                    f.write(f"  Throughput: {throughput:.2f} samples/sec\n\n")
            
            # Add hardware info
            try:
                f.write(f"\nHardware Information:\n")
                f.write(f"System: {platform.system()} {platform.release()}\n")
                f.write(f"CPU: {platform.processor()}\n")
                f.write(f"RAM: {psutil.virtual_memory().total // (1024**3)} GB\n")
                
                # GPU info for NVIDIA
                try:
                    import subprocess
                    gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True)
                    f.write(f"GPU: {gpu_info.decode('utf-8').strip()}\n")
                except:
                    f.write("GPU: Information not available\n")
            except:
                f.write("Hardware information not available\n")
        
        
        print(f"\n[INFO] Overall performance metrics saved to {inference_stats_path}")
        print(f"[INFERENCE STATS] Processed {total_samples} samples")
        print(f"[INFERENCE STATS] Average inference time: {overall_avg_time:.2f} ms/sample")
        print(f"[INFERENCE STATS] Throughput: {overall_throughput:.2f} samples/sec")
        print(f"[INFERENCE STATS] Total evaluation time: {total_time:.2f} seconds")
        print(f"[INFERENCE STATS] Effective batch size: {batch_size}")

        # Generate final table
        generate_final_table(final_results, output_dir)
        return final_results
    
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        traceback.print_exc()
        return {i: [] for i in range(1, 8)}

def generate_final_table(final_results, output_dir):
    """Generate a formatted table of model accuracies."""
    try:
        # Calculate mean accuracies
        table_data = {}
        
        for leak_class in range(1, 8):
            accuracies = final_results[leak_class]
            
            if len(accuracies) == 0:
                table_data[leak_class] = 0.0
            else:
                # Convert to percentage with 1 decimal place
                table_data[leak_class] = round(np.mean(accuracies) * 100, 1)
        
        # Create formatted table
        table_filename = os.path.join(output_dir, f"performance_table_60x80.txt")
        with open(table_filename, "w") as f:
            f.write("Table: 3DCNN model with 60x80 resolution performance in leak vs. non-leak binary classification\n\n")
            
            # Accuracy table
            f.write("ACCURACY\n")
            f.write("Architecture     | 0-1   0-2   0-3   0-4   0-5   0-6   0-7\n")
            f.write("----------------------------------------------------------\n")
            
            row_str = "VideoGasNet   |"
            for leak_class in range(1, 8):
                acc = table_data[leak_class]
                row_str += f" {acc:6.1f}% "
            f.write(row_str + "\n\n")
               
        print(f"\nPerformance table saved to: {table_filename}")
    except Exception as e:
        print(f"[ERROR] Failed to generate performance table: {e}")

def test_model(test_dir, model_path, batch_size, output_base_dir="./result", method_name="unknown", 
               target_height=120, target_width=160, three_class_mode=False, distance_filter=None):
    """
    Main testing function that evaluates the model on test data.
    Now supports both binary and three-class classification modes.
    
    Args:
        test_dir: Directory containing test data
        model_path: Path to the trained model
        batch_size: Batch size for testing
        output_base_dir: Base directory to save test results
        method_name: Name of the method (for result organization)
        target_height: Target image height
        target_width: Target image width
        three_class_mode: Set to True for three-class classification (small/medium/large leaks)
        distance_filter: Filter data by imaging distance: '46' for 4.6m, '69' for 6.9m, or 'all' for all data
    """
    try:
        # Check if model_path exists, if not, try to find the model with alternative naming
        if not os.path.exists(model_path) and not three_class_mode:
            # Extract the base directory from the path
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(model_path)))
            method_dir = "optical_flow_gray" if "gray" in method_name.lower() else "optical_flow_mog2"
            
            # Alternative model path formats based on the method
            if "mog2" in method_name.lower():
                alt_model_path = os.path.join(base_dir, f"best_model_{method_dir}", "data_aug", 
                                             "cnn_3d_binaryAllLeak_best_model.keras")
            else:  # Moving Average
                alt_model_path = os.path.join(base_dir, f"best_model_{method_dir}", "data_aug", 
                                             "cnn_3d_binaryAllLeak_best.keras")
            
            # Use alternative path if it exists
            if os.path.exists(alt_model_path):
                print(f"[INFO] Original model path not found: {model_path}")
                print(f"[INFO] Using alternative model path: {alt_model_path}")
                model_path = alt_model_path
            else:
                print(f"[WARNING] Could not find model at either path:")
                print(f"  - Original: {model_path}")
                print(f"  - Alternative: {alt_model_path}")
        
        # Create method-specific output directory
        output_dir = os.path.join(output_base_dir, method_name)
        os.makedirs(output_dir, exist_ok=True)

        # Create subdirectories for different tests
        if three_class_mode:
            # For three-class mode
            three_class_dir = os.path.join(output_dir, "three_class_test")
            os.makedirs(three_class_dir, exist_ok=True)
            
            # Add dimension info to directory names
            dimension_str = f"{target_height}x{target_width}"
            three_class_dir_with_info = os.path.join(three_class_dir, dimension_str)
            os.makedirs(three_class_dir_with_info, exist_ok=True)
            
            # Save system information and configuration
            config_file = os.path.join(output_base_dir, f"test_config_{datetime.now().strftime('%Y-%m-%d')}.txt")
            with open(config_file, 'w') as f:
                f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model Path: {model_path}\n")
                f.write(f"Test Data Directory: {test_dir}\n")
                f.write(f"Target Size: {target_height}x{target_width}\n")
                f.write(f"Batch Size: {batch_size}\n")
                f.write(f"Method: {method_name}\n")
                f.write(f"Mode: Three-Class Classification\n")
                f.write(f"Python Version: {os.sys.version}\n")
                f.write(f"TensorFlow Version: {tf.__version__}\n")
                
                # Hardware info if available
                try:
                    f.write(f"System: {platform.system()} {platform.release()}\n")
                    f.write(f"CPU: {platform.processor()}\n")
                    f.write(f"RAM: {psutil.virtual_memory().total // (1024**3)} GB\n")
                    
                    # GPU info 
                    try:
                        import subprocess
                        gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True)
                        f.write(f"GPU: {gpu_info.decode('utf-8').strip()}\n")
                    except:
                        f.write("GPU: Information not available\n")
                except:
                    f.write("Hardware information not available\n")
            
            print(f"\n{'='*70}")
            print(f"THREE-CLASS TESTING FOR {method_name} [{dimension_str}]")
            print(f"{'='*70}")
            
            # Run three-class evaluation
            results = None
            results_46m = None
            results_69m = None
            
            # Run evaluations on specific distances or all distances
            if distance_filter == '46':
                print("\n[INFO] Evaluating only 4.6m distance data as specified...")
                results_46m = evaluate_three_class_by_distance(model_path, test_dir, batch_size, 
                                                             three_class_dir_with_info, '46', target_height, target_width)
            elif distance_filter == '69':
                print("\n[INFO] Evaluating only 6.9m distance data as specified...")
                results_69m = evaluate_three_class_by_distance(model_path, test_dir, batch_size, 
                                                             three_class_dir_with_info, '69', target_height, target_width)
            elif distance_filter == 'all' or distance_filter is None:
                # First evaluate on all data
                print("\n[INFO] Evaluating on all imaging distances...")
                results = evaluate_three_class(model_path, test_dir, batch_size, three_class_dir_with_info, 
                                      target_height, target_width)
                
                # Then evaluate each distance separately
                print("\n[INFO] Additionally evaluating 4.6m distance specifically...")
                results_46m = evaluate_three_class_by_distance(model_path, test_dir, batch_size, 
                                                             three_class_dir_with_info, '46', target_height, target_width)
                print("\n[INFO] Additionally evaluating 6.9m distance specifically...")
                results_69m = evaluate_three_class_by_distance(model_path, test_dir, batch_size, 
                                                             three_class_dir_with_info, '69', target_height, target_width)
            
            # Combine results for summary
            print("\n[SUMMARY] Three-Class Classification Results:")
            
            if results is not None:
                print(f"All data accuracy: {results['accuracy']:.4f}")
            
            if results_46m is not None:
                print(f"4.6m distance accuracy: {results_46m['accuracy']:.4f}")
                
            if results_69m is not None:
                print(f"6.9m distance accuracy: {results_69m['accuracy']:.4f}")
            
            # Save summary to file
            summary_path = os.path.join(three_class_dir_with_info, f'three_class_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
            with open(summary_path, 'w') as f:
                f.write("Three-Class Classification Summary\n")
                f.write("================================\n\n")
                f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {model_path}\n")
                f.write(f"Resolution: {target_height}x{target_width}\n\n")
                
                if results is not None:
                    f.write(f"All data accuracy: {results['accuracy']:.4f}\n")
                
                if results_46m is not None:
                    f.write(f"4.6m distance accuracy: {results_46m['accuracy']:.4f}\n")
                
                if results_69m is not None:
                    f.write(f"6.9m distance accuracy: {results_69m['accuracy']:.4f}\n")
            
            print(f"[INFO] Summary saved to {summary_path}")
            
            # Return the main results (prioritize 'all' if available)
            return results if results is not None else (results_46m if results_46m is not None else results_69m)
            
        else:
            # Original binary classification testing
            all_leak_dir = os.path.join(output_dir, "all_leak_test")
            fold_test_dir = os.path.join(output_dir, "10fold_test")
            os.makedirs(all_leak_dir, exist_ok=True)
            os.makedirs(fold_test_dir, exist_ok=True)

            # Add dimension and device info to directory names
            dimension_str = f"{target_height}x{target_width}"
            
            output_subdir = f"{dimension_str}"
            all_leak_dir_with_info = os.path.join(all_leak_dir, output_subdir)
            fold_test_dir_with_info = os.path.join(fold_test_dir, output_subdir)
            os.makedirs(all_leak_dir_with_info, exist_ok=True)
            os.makedirs(fold_test_dir_with_info, exist_ok=True)

            # Save system information and configuration
            config_file = os.path.join(output_base_dir, f"test_config_{datetime.now().strftime('%Y-%m-%d')}.txt")
            with open(config_file, 'w') as f:
                f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model Path: {model_path}\n")
                f.write(f"Test Data Directory: {test_dir}\n")
                f.write(f"Target Size: {target_height}x{target_width}\n")
                f.write(f"Batch Size: {batch_size}\n")
                f.write(f"Method: {method_name}\n")
                f.write(f"Mode: Binary Classification\n")
                f.write(f"Python Version: {os.sys.version}\n")
                f.write(f"TensorFlow Version: {tf.__version__}\n")
                
                # Hardware info if available
                try:
                    f.write(f"System: {platform.system()} {platform.release()}\n")
                    f.write(f"CPU: {platform.processor()}\n")
                    f.write(f"RAM: {psutil.virtual_memory().total // (1024**3)} GB\n")
                    
                    # GPU info 
                    try:
                        import subprocess
                        gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True)
                        f.write(f"GPU: {gpu_info.decode('utf-8').strip()}\n")
                    except:
                        f.write("GPU: Information not available\n")
                except:
                    f.write("Hardware information not available\n")

            print(f"\n{'='*70}")
            print(f"TESTING ALL LEAKS VS NO-LEAK FOR {method_name} [{output_subdir}]")
            print(f"{'='*70}")
            evaluate_all_leak_vs_no_leak(model_path, test_dir, batch_size, all_leak_dir_with_info, 
                                      target_height, target_width)

            print(f"\n{'='*70}")
            print(f"10-FOLD TESTING FOR EACH LEAK CLASS VS NO-LEAK FOR {method_name} [{output_subdir}]")
            print(f"{'='*70}")
            final_results = evaluate_per_leak_class(model_path, test_dir, batch_size, fold_test_dir_with_info, 
                                                target_height, target_width)

            # Generate final table with complete info
            print(f"\n{'='*70}")
            print(f"GENERATING FINAL TABLE FOR {method_name} [{output_subdir}]")
            print(f"{'='*70}")
            output_dir_with_info = os.path.join(output_base_dir, output_subdir)
            os.makedirs(output_dir_with_info, exist_ok=True)
            generate_final_table(final_results, output_dir_with_info)

            # Create a summary file with complete info
            summary_path = os.path.join(output_dir_with_info, f"{method_name}_{output_subdir}_test_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(f"Test Summary for {method_name} [{output_subdir}]\n")
                f.write("================================\n\n")
                f.write(f"Resolution: {target_height}x{target_width}\n")
                f.write(f"All leak vs no-leak results: {all_leak_dir_with_info}\n")
                f.write(f"10-fold test results: {fold_test_dir_with_info}\n")
                f.write(f"Final table location: {output_dir_with_info}\n")
            
            return final_results
    
    except Exception as e:
        print(f"[ERROR] Test failed with error: {e}")
        traceback.print_exc()
        if three_class_mode:
            return {
                'accuracy': 0.0,
                'class_report': None,
                'confusion_matrix': None,
                'y_true': [],
                'y_pred': []
            }
        else:
            return {i: [] for i in range(1, 8)}

def evaluate_three_class_by_distance(model_path, test_dir, batch_size, output_dir, distance_code, 
                                     target_height=240, target_width=320):
    """
    Evaluate three-class classification separately for a specific imaging distance.
    
    Args:
        model_path: Path to the trained model
        test_dir: Directory containing test data
        batch_size: Batch size for evaluation
        output_dir: Directory to save output files
        distance_code: Distance code to filter (e.g., '46' for 4.6m or '69' for 6.9m)
        target_height: Target image height
        target_width: Target image width
    """
    print(f"[INFO] Evaluating three-class classification for distance {distance_code}m...")
    
    try:
        # Directly use the model path provided from main.py
        print(f"[INFO] Using model from: {model_path}")
        
        # Check if the model exists
        if not os.path.exists(model_path):
            print(f"[ERROR] Model not found at: {model_path}")
            raise FileNotFoundError(f"Model not found at: {model_path}")
            
        # Define class names for reporting
        class_names = ['Small Leak (0-2)', 'Medium Leak (3-5)', 'Large Leak (6-7)']
        
        # Load model
        model = cnn_3d_model(input_shape=(15, target_height, target_width, 1), num_classes=3)
        model.load_weights(model_path)

        # Warmup a dummy input
        print("[INFO] Warming up the model with a dummy input")
        dummy_input = np.random.rand(1, 15, target_height, target_width, 1).astype(np.float32)
        _ = model.predict(dummy_input, verbose=1)
        
        # Create data generator for three-class mode with distance filtering
        test_gen = DataGeneratorThreeClass(
            data_dir=test_dir,
            batch_size=batch_size,
            shuffle=False,
            balance_classes=False,
            training=False,
            resize=True,
            target_height=target_height,
            target_width=target_width,
            distance_filter=distance_code
        )
        
        if len(test_gen.filepaths) == 0:
            print(f"[ERROR] No test data found for distance {distance_code}m")
            return {
                'accuracy': 0.0,
                'class_report': None,
                'confusion_matrix': None,
                'y_true': [],
                'y_pred': []
            }
        
        print(f"[INFO] Found {len(test_gen.filepaths)} files with distance code {distance_code}")
        
        # Calculate number of batches
        num_batches = len(test_gen)
        print(f"[INFO] Running predictions on {num_batches} batches...")
        
        # Store predictions and true labels
        y_true = []
        y_pred = []
        
        # Track time for performance monitoring
        start_time = time.time()
        
        # Process batches
        for batch_idx in range(num_batches):
            if batch_idx % 10 == 0:
                print(f"[INFO] Processing batch {batch_idx+1}/{num_batches}")
            
            X_batch, y_batch = test_gen[batch_idx]
            if len(X_batch) == 0:
                continue
                
            # Make predictions
            batch_pred = model.predict(X_batch, verbose=0)
            batch_pred_classes = np.argmax(batch_pred, axis=1)
            
            # Store results
            y_true.extend(y_batch)
            y_pred.extend(batch_pred_classes)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        elapsed_time = time.time() - start_time
        
        print(f"[INFO] Three-class evaluation for distance {distance_code}m completed in {elapsed_time:.2f} seconds")
        print(f"[INFO] Accuracy: {accuracy:.4f}")
        
        # Create a subdirectory for this distance evaluation
        distance_output_dir = os.path.join(output_dir, f"distance_{distance_code}m")
        os.makedirs(distance_output_dir, exist_ok=True)
        
        # Save confusion matrix and classification report
        save_confusion_matrix(y_true, y_pred, class_names, distance_output_dir)
        save_classification_report(y_true, y_pred, distance_output_dir)
        
        # Save detailed results
        results_path = os.path.join(
            distance_output_dir, 
            f'three_class_results_distance_{distance_code}m_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        )
        
        with open(results_path, 'w') as f:
            f.write(f"Three-Class Leak Size Classification Results for Distance {distance_code}m\n")
            f.write(f"=================================================================\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Test Data: {test_dir} (filtered for distance {distance_code}m)\n")
            f.write(f"Resolution: {target_height}x{target_width}\n\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
            f.write(f"Class Distribution:\n")
            f.write(f"  Small Leak (0-2): {y_true.count(0)} samples\n")
            f.write(f"  Medium Leak (3-5): {y_true.count(1)} samples\n")
            f.write(f"  Large Leak (6-7): {y_true.count(2)} samples\n\n")
            f.write(f"Evaluation Time: {elapsed_time:.2f} seconds\n")
        
        print(f"[INFO] Detailed results saved to {results_path}")
        
        return {
            'accuracy': accuracy,
            'y_true': y_true,
            'y_pred': y_pred,
            'class_report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
    except Exception as e:
        print(f"[ERROR] Three-class evaluation for distance {distance_code}m failed: {e}")
        traceback.print_exc()
        return {
            'accuracy': 0.0,
            'class_report': None,
            'confusion_matrix': None,
            'y_true': [],
            'y_pred': []
        }

def evaluate_three_class_distance_46(model_path, test_dir, batch_size, output_dir, target_height=240, target_width=320):
    """Evaluate three-class classification for 4.6m imaging distance."""
    return evaluate_three_class_by_distance(
        model_path, test_dir, batch_size, output_dir, '46', target_height, target_width
    )

def evaluate_three_class_distance_69(model_path, test_dir, batch_size, output_dir, target_height=240, target_width=320):
    """Evaluate three-class classification for 6.9m imaging distance."""
    return evaluate_three_class_by_distance(
        model_path, test_dir, batch_size, output_dir, '69', target_height, target_width
    )
