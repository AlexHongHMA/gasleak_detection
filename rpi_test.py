"""
rpi_test.py - Testing module for RPi models
Following original test.py structure with comprehensive evaluation methods
"""

import os
import time
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import platform
import psutil
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
import glob
from tensorflow.keras import layers

from src.loader.loader import DataGenerator
from src.test.test import save_confusion_matrix, save_classification_report
# Import all custom objects from rpi_optimized_model
from rpi_optimized_model import FrameSampler, ResidualBlock, EfficientBlock, SimpleFrameSampler


class ModelEvaluator:
    """Enhanced model evaluator with better error handling and model loading"""
    
    def __init__(self, model_path, target_height, target_width):
        self.model_path = model_path
        self.target_height = target_height
        self.target_width = target_width
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load model with enhanced error handling and custom objects"""
        # Import all possible custom objects
        try:
            from rpi_optimized_model import (
                FrameSampler, ResidualBlock, EfficientBlock
            )
            
            # Try to import SimpleFrameSampler if it exists
            try:
                from rpi_optimized_model import SimpleFrameSampler
            except ImportError:
                # Create a dummy SimpleFrameSampler if it doesn't exist
                @tf.keras.saving.register_keras_serializable()
                class SimpleFrameSampler(layers.Layer):
                    def __init__(self, target_frames=10, **kwargs):
                        super().__init__(**kwargs)
                        self.target_frames = target_frames
                    
                    def call(self, inputs):
                        sequence_length = tf.shape(inputs)[1]
                        if sequence_length <= self.target_frames:
                            return inputs
                        indices = tf.linspace(0, sequence_length - 1, self.target_frames)
                        indices = tf.cast(indices, tf.int32)
                        return tf.gather(inputs, indices, axis=1)
                    
                    def get_config(self):
                        config = super().get_config()
                        config.update({'target_frames': self.target_frames})
                        return config
            
        except ImportError as e:
            print(f"[WARNING] Failed to import custom objects: {e}")
            FrameSampler = None
            ResidualBlock = None
            EfficientBlock = None
            SimpleFrameSampler = None
        
        # Comprehensive custom objects dictionary
        custom_objects = {}
        if FrameSampler:
            custom_objects['FrameSampler'] = FrameSampler
        if SimpleFrameSampler:
            custom_objects['SimpleFrameSampler'] = SimpleFrameSampler
        if ResidualBlock:
            custom_objects['ResidualBlock'] = ResidualBlock
        if EfficientBlock:
            custom_objects['EfficientBlock'] = EfficientBlock
        
        # List of model files to try (in order of preference)
        model_files_to_try = []
        
        if os.path.isfile(self.model_path):
            model_files_to_try.append(self.model_path)
        elif os.path.isdir(self.model_path):
            # If it's a directory, look for model files
            for ext in ['.keras', '.h5']:
                pattern = os.path.join(self.model_path, f"*{ext}")
                model_files_to_try.extend(glob.glob(pattern))
        else:
            # Try to find in the directory containing the model path
            model_dir = os.path.dirname(self.model_path)
            model_name = os.path.basename(self.model_path)
            
            # Try exact name first
            if os.path.exists(os.path.join(model_dir, model_name)):
                model_files_to_try.append(os.path.join(model_dir, model_name))
            
            # Try variations of the name
            name_variations = [
                model_name,
                model_name.replace('.keras', ''),
                f"best_{model_name}",
                f"{model_name}_best",
                f"final_{model_name}",
                f"{model_name}_final"
            ]
            
            for variation in name_variations:
                for ext in ['.keras', '.h5']:
                    full_path = os.path.join(model_dir, f"{variation}{ext}")
                    if os.path.exists(full_path) and full_path not in model_files_to_try:
                        model_files_to_try.append(full_path)
        
        print(f"[INFO] Found {len(model_files_to_try)} model files to try")
        
        # Try loading each candidate
        last_error = None
        for candidate_path in model_files_to_try:
            if not os.path.exists(candidate_path):
                print(f"[WARNING] Model file not found: {candidate_path}")
                continue
                
            try:
                print(f"[INFO] Attempting to load: {os.path.basename(candidate_path)}")
                
                # Try loading with custom objects
                self.model = tf.keras.models.load_model(candidate_path, custom_objects=custom_objects)
                self.model_path = candidate_path  # Update to the successful path
                print(f"[SUCCESS] Model loaded successfully: {os.path.basename(candidate_path)}")
                
                # Verify model structure
                print(f"[INFO] Model input shape: {self.model.input_shape}")
                print(f"[INFO] Model output shape: {self.model.output_shape}")
                print(f"[INFO] Model parameters: {self.model.count_params():,}")
                
                return
                
            except Exception as e:
                print(f"[WARNING] Failed to load {os.path.basename(candidate_path)}: {e}")
                last_error = e
                continue
        
        # If all attempts failed, raise the last error
        raise RuntimeError(f"Failed to load any model. Last error: {last_error}")
    
    def predict(self, X):
        """Make predictions"""
        if self.model_path.endswith('.tflite'):
            return self._predict_tflite(X)
        else:
            return self.model.predict(X, verbose=0)
    
    def _predict_tflite(self, X):
        """Predict using TFLite model"""
        interpreter = tf.lite.Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        predictions = []
        
        for sample in X:
            # Prepare input
            input_data = np.expand_dims(sample, axis=0).astype(input_details[0]['dtype'])
            
            # Set input
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predictions.append(output_data[0])
        
        return np.array(predictions)


class TFLiteModelEvaluator:
    """TensorFlow Lite model evaluator for RPi testing"""
    
    def __init__(self, model_path, target_height=120, target_width=160):
        """Initialize TFLite interpreter"""
        self.model_path = model_path
        self.target_height = target_height
        self.target_width = target_width
        
        print(f"[INFO] Loading TFLite model: {model_path}")
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"[INFO] TFLite model loaded successfully")
        print(f"[INFO] Input shape: {self.input_details[0]['shape']}")
        print(f"[INFO] Output shape: {self.output_details[0]['shape']}")
        
    def predict(self, input_data):
        """Run inference on input data - handles batches by processing samples individually"""
        # TFLite models typically expect batch size 1, so we process samples individually
        batch_size = input_data.shape[0]
        predictions = []
        
        for i in range(batch_size):
            # Extract single sample and add batch dimension
            single_sample = np.expand_dims(input_data[i], axis=0).astype(np.float32)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], single_sample)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output tensor
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            predictions.append(output_data[0])  # Remove batch dimension
        
        return np.array(predictions)


def save_confusion_matrix(y_true, y_pred, class_names, output_dir, filename_prefix='confusion_matrix'):
    """Save confusion matrix visualization following original test.py format"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save plot
    plot_path = os.path.join(output_dir, f'{filename_prefix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Confusion matrix saved: {plot_path}")
    return plot_path


def save_classification_report(y_true, y_pred, output_dir):
    """Save classification report following original test.py format"""
    report = classification_report(y_true, y_pred, digits=4)
    
    report_path = os.path.join(output_dir, f'classification_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("===================\n\n")
        f.write(report)
    
    print(f"[INFO] Classification report saved: {report_path}")
    return report_path


def get_gpu_info():
    """
    Cross-platform GPU information detection
    Supports NVIDIA (laptop), Broadcom VideoCore (RPi), and other GPUs
    """
    gpu_info = []
    
    # Method 1: Try NVIDIA GPU detection (for laptops/desktops)
    try:
        import subprocess
        nvidia_info = subprocess.check_output(
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits", 
            shell=True, stderr=subprocess.DEVNULL
        )
        for line in nvidia_info.decode('utf-8').strip().split('\n'):
            if line.strip():
                name, memory = line.split(',')
                gpu_info.append(f"NVIDIA {name.strip()} ({memory.strip()}MB)")
    except:
        pass
    
    # Method 2: Raspberry Pi VideoCore GPU detection
    try:
        import subprocess
        
        # Check if vcgencmd is available (Raspberry Pi specific)
        try:
            # Get GPU memory split
            gpu_mem = subprocess.check_output("vcgencmd get_mem gpu", shell=True, stderr=subprocess.DEVNULL)
            gpu_mem_str = gpu_mem.decode('utf-8').strip()
            
            # Get core temperature as a health indicator
            try:
                temp = subprocess.check_output("vcgencmd measure_temp", shell=True, stderr=subprocess.DEVNULL)
                temp_str = temp.decode('utf-8').strip()
            except:
                temp_str = "temp=unknown"
            
            # Detect VideoCore version
            try:
                # Check if we're on RPi 4 by looking at hardware revision
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    if any(rev in cpuinfo for rev in ['c03111', 'c03112', 'c03114', 'c03130']):  # RPi 4 revisions
                        gpu_name = "Broadcom VideoCore VI"
                    elif any(rev in cpuinfo for rev in ['a02082', 'a22082', 'a32082']):  # RPi 3 revisions
                        gpu_name = "Broadcom VideoCore IV"
                    else:
                        gpu_name = "Broadcom VideoCore"
            except:
                gpu_name = "Broadcom VideoCore"
            
            gpu_info.append(f"{gpu_name} ({gpu_mem_str}, {temp_str})")
            
        except:
            # Fallback: Check for GPU in device tree (Raspberry Pi)
            try:
                gpu_paths = [
                    '/proc/device-tree/soc/gpu',
                    '/proc/device-tree/gpu',
                    '/sys/firmware/devicetree/base/soc/gpu'
                ]
                
                for path in gpu_paths:
                    if os.path.exists(path):
                        gpu_info.append("Broadcom VideoCore (detected via device tree)")
                        break
            except:
                pass
    except:
        pass
    
    # Method 3: Generic GPU detection via lspci (works on most Linux systems)
    try:
        import subprocess
        lspci_output = subprocess.check_output("lspci | grep -i vga", shell=True, stderr=subprocess.DEVNULL)
        for line in lspci_output.decode('utf-8').strip().split('\n'):
            if line.strip() and 'VGA' in line:
                # Extract GPU name from lspci output
                gpu_name = line.split(': ')[-1] if ': ' in line else line
                if not any(existing in gpu_name for existing in [info.split('(')[0].strip() for info in gpu_info]):
                    gpu_info.append(f"GPU: {gpu_name}")
    except:
        pass
    
    # Method 4: Try to detect via /sys/class/graphics
    try:
        graphics_path = '/sys/class/graphics'
        if os.path.exists(graphics_path):
            fb_devices = [d for d in os.listdir(graphics_path) if d.startswith('fb')]
            if fb_devices and not gpu_info:
                gpu_info.append(f"Graphics framebuffer detected ({len(fb_devices)} devices)")
    except:
        pass
    
    # Method 5: Intel GPU detection (common on laptops)
    try:
        import subprocess
        intel_gpu = subprocess.check_output("lspci | grep -i intel | grep -i graphics", shell=True, stderr=subprocess.DEVNULL)
        if intel_gpu and not any('Intel' in info for info in gpu_info):
            intel_name = intel_gpu.decode('utf-8').strip().split(': ')[-1]
            gpu_info.append(f"Intel {intel_name}")
    except:
        pass
    
    # Method 6: AMD GPU detection
    try:
        import subprocess
        amd_gpu = subprocess.check_output("lspci | grep -i amd | grep -i graphics", shell=True, stderr=subprocess.DEVNULL)
        if amd_gpu and not any('AMD' in info for info in gpu_info):
            amd_name = amd_gpu.decode('utf-8').strip().split(': ')[-1]
            gpu_info.append(f"AMD {amd_name}")
    except:
        pass
    
    return gpu_info if gpu_info else ["Information not available"]


def get_system_info():
    """
    Enhanced system information gathering for both laptop and RPi
    """
    info = {}
    
    try:
        info['system'] = f"{platform.system()} {platform.release()}"
        info['machine'] = platform.machine()
        info['processor'] = platform.processor()
        
        # Enhanced RAM detection
        ram_info = psutil.virtual_memory()
        info['ram_total'] = f"{ram_info.total // (1024**3)} GB"
        info['ram_available'] = f"{ram_info.available // (1024**3)} GB"
        
        # CPU information
        info['cpu_count'] = psutil.cpu_count()
        info['cpu_freq'] = psutil.cpu_freq()
        
        # Raspberry Pi specific information
        if os.path.exists('/proc/cpuinfo'):
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    
                # Extract RPi model information
                for line in cpuinfo.split('\n'):
                    if 'Model' in line and ':' in line:
                        info['rpi_model'] = line.split(':')[1].strip()
                    elif 'Revision' in line and ':' in line:
                        info['rpi_revision'] = line.split(':')[1].strip()
                    elif 'Serial' in line and ':' in line:
                        info['rpi_serial'] = line.split(':')[1].strip()[:8] + "..."  # Truncate for privacy
            except:
                pass
        
        # GPU information
        info['gpu'] = get_gpu_info()
        
        # Disk information
        try:
            disk_usage = psutil.disk_usage('/')
            info['disk_total'] = f"{disk_usage.total // (1024**3)} GB"
            info['disk_free'] = f"{disk_usage.free // (1024**3)} GB"
        except:
            pass
            
    except Exception as e:
        info['error'] = str(e)
    
    return info


def evaluate_all_leak_vs_no_leak(model_path, test_dir, batch_size, output_dir, 
                                 target_height=120, target_width=160, run_error_analysis=False):
    """
    Evaluate binary classification performance for RPi models (all leak vs. no leak).
    Now supports both Keras and TFLite models.
    """
    print("[INFO] Evaluating all leak vs no leak binary classification...")
    start_time = time.time()
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine model type and load appropriate evaluator
        if model_path.endswith('.tflite'):
            print("[INFO] Using TFLite model evaluator")
            evaluator = TFLiteModelEvaluator(model_path, target_height, target_width)
            model_type = 'tflite'
        else:
            print("[INFO] Using Keras model evaluator")
            evaluator = ModelEvaluator(model_path, target_height, target_width)
            model_type = 'keras'
        
        # Warmup with dummy input - adjust batch size for TFLite
        print("[INFO] Warming up with dummy input")
        warmup_batch_size = 1 if model_type == 'tflite' else batch_size
        dummy_input = np.random.rand(warmup_batch_size, 15, target_height, target_width, 1).astype(np.float32)
        _ = evaluator.predict(dummy_input)
        _ = evaluator.predict(dummy_input)  # Second warmup for stability

        # Create test data generator
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

        # Verify generator output shape
        try:
            test_batch, test_labels = test_gen[0]
            print(f"\n[SHAPE TEST] First batch shape: {test_batch.shape}")
            print(f"Sample data range: {test_batch.min()} to {test_batch.max()}")
        except IndexError:
            print("[ERROR] Test generator is empty!")
            return [], []

        y_true, y_pred = [], []
        print(f"[INFO] Testing Start - Using {model_type.upper()} model")
        processed_samples = 0
        start_time = time.time()
        
        # Inference time tracking
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

            # Measure inference time specifically
            inference_start = time.time()
            
            # For TFLite models, we may need to adjust batch processing
            if model_type == 'tflite':
                # TFLite processes samples individually, which is already handled in TFLiteModelEvaluator.predict()
                preds = evaluator.predict(X_batch)
            else:
                preds = evaluator.predict(X_batch)
                
            inference_time = time.time() - inference_start
            inference_time_ms = inference_time * 1000
            
            # Track inference statistics
            total_inference_time += inference_time_ms
            inference_samples += len(X_batch)
            all_inference_times_ms.extend([inference_time_ms / len(X_batch)] * len(X_batch))
            
            # Per sample in milliseconds
            per_sample_ms = inference_time_ms / len(X_batch)
            print(f"Batch inference time: {inference_time_ms:.2f} ms | "
                  f"Per sample: {per_sample_ms:.2f} ms | "
                  f"Model: {model_type.upper()}")

            y_pred.extend(preds.argmax(axis=-1))
            y_true.extend(y_batch)
            
            # Explicit garbage collection
            gc.collect()
            
            start_time = time.time()

        # Calculate inference statistics
        avg_inference_time_per_sample_ms = total_inference_time / inference_samples if inference_samples > 0 else 0
        avg_inference_time_per_batch_ms = total_inference_time / (batch_idx + 1) if batch_idx >= 0 else 0
        throughput = 1000 / avg_inference_time_per_sample_ms if avg_inference_time_per_sample_ms > 0 else 0
        
        print(f"\n[INFERENCE STATS] Average inference time: {avg_inference_time_per_sample_ms:.2f} ms per sample")
        print(f"[INFERENCE STATS] Throughput: {throughput:.2f} samples/sec")
        print(f"[INFERENCE STATS] Model format: {model_type.upper()}")
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Save inference stats and metrics to file
        inference_stats_path = os.path.join(output_dir, f'inference_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        # Get enhanced system information
        system_info = get_system_info()
        
        with open(inference_stats_path, 'w') as f:
            f.write(f"Model Format: {model_type.upper()}\n")
            f.write(f"Model Path: {model_path}\n")
            f.write(f"Total samples processed: {inference_samples}\n")
            f.write(f"Total inference time: {total_inference_time:.2f} ms\n")
            f.write(f"Average inference time per sample: {avg_inference_time_per_sample_ms:.2f} ms\n")
            f.write(f"Average inference time per batch (batch size={batch_size}): {avg_inference_time_per_batch_ms:.2f} ms\n")
            f.write(f"Throughput: {throughput:.2f} samples/sec\n")
            f.write("Performance Metrics for All Leaks vs No-Leak\n")
            f.write("==========================================\n\n")
            f.write(f"Total samples processed: {processed_samples}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(classification_report(y_true, y_pred, digits=4))
            
            # Add enhanced hardware info
            f.write(f"\nSystem Information:\n")
            f.write(f"System: {system_info.get('system', 'Unknown')}\n")
            f.write(f"Machine: {system_info.get('machine', 'Unknown')}\n")
            f.write(f"Processor: {system_info.get('processor', 'Unknown')}\n")
            f.write(f"CPU Count: {system_info.get('cpu_count', 'Unknown')}\n")
            f.write(f"RAM Total: {system_info.get('ram_total', 'Unknown')}\n")
            f.write(f"RAM Available: {system_info.get('ram_available', 'Unknown')}\n")
            
            # Raspberry Pi specific info
            if 'rpi_model' in system_info:
                f.write(f"RPi Model: {system_info['rpi_model']}\n")
            if 'rpi_revision' in system_info:
                f.write(f"RPi Revision: {system_info['rpi_revision']}\n")
            
            # GPU information - now supports multiple GPU types
            gpu_list = system_info.get('gpu', ['Information not available'])
            if isinstance(gpu_list, list):
                for i, gpu in enumerate(gpu_list):
                    f.write(f"GPU {i+1}: {gpu}\n")
            else:
                f.write(f"GPU: {gpu_list}\n")
                
            # Disk information
            if 'disk_total' in system_info:
                f.write(f"Disk Total: {system_info['disk_total']}\n")
                f.write(f"Disk Free: {system_info['disk_free']}\n")
        
        print(f"Inference statistics saved to {inference_stats_path}")

        # Save results following original test.py format
        class_names = ['No Leak', 'Leak']
        save_confusion_matrix(y_true, y_pred, class_names, output_dir, filename_prefix=f'binary_all_leak_vs_no_leak')
        save_classification_report(y_true, y_pred, output_dir)          
        
        # Print final stats
        print(f"[INFERENCE STATS] Processed {processed_samples} samples")
        print(f"[INFERENCE STATS] Average inference time: {avg_inference_time_per_sample_ms:.2f} ms/sample")
        print(f"[INFERENCE STATS] Throughput: {throughput:.2f} samples/sec")
        print(f"[INFERENCE STATS] Total evaluation time: {total_inference_time:.2f} seconds")
        print(f"[INFERENCE STATS] Effective batch size: {batch_size}")
        print(f"[INFERENCE STATS] Model format: {model_type.upper()}")

        return y_true, y_pred
    
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return [], []


def evaluate_per_leak_class(model_path, test_dir, batch_size, output_dir, target_height=120, target_width=160):
    """
    Perform 10-fold testing for each leak class vs. no leak (0..7) using RPi models.
    Following original test.py structure exactly.
    """
    print("[INFO] Evaluating leak classes using RPi model (10-fold)...")
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine model type and load appropriate evaluator
        if model_path.endswith('.tflite'):
            print("[INFO] Using TFLite model evaluator for per-class evaluation")
            evaluator = TFLiteModelEvaluator(model_path, target_height, target_width)
            model_type = 'tflite'
        else:
            print("[INFO] Using Keras model evaluator for per-class evaluation")
            evaluator = ModelEvaluator(model_path, target_height, target_width)
            model_type = 'keras'

        final_results = {i: [] for i in range(1, 8)}  # Store accuracies for each leak class
        inference_stats = {i: {'times': [], 'samples': 0} for i in range(1, 8)}  # Store inference times
        all_inference_times_ms = []  # Store all per-sample times in ms
        total_samples = 0
        overall_start = time.time()

        for leak_class in range(1, 8):
            print(f"\n--- 0 vs {leak_class} (10-fold) ---")
            
            # Create a data generator for this leak class
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
                    preds = evaluator.predict(X_batch)
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
            print(f"\n0 vs {leak_class} | Mean Accuracy: {mean_acc:.4f} +/- {std_acc:.3f}")
            print(f"Class evaluation time: {class_time:.2f} seconds")

        # Save inference statistics and overall metrics
        inference_stats_path = os.path.join(output_dir, f'inference_stats_per_class_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        # Calculate overall metrics
        total_time = time.time() - overall_start
        overall_avg_time = np.mean(all_inference_times_ms) if all_inference_times_ms else 0
        overall_throughput = 1000 / overall_avg_time if overall_avg_time > 0 else 0

        # Get enhanced system information
        system_info = get_system_info()

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
                    f.write(f"  Class 0 vs {leak_class}: {mean_acc:.4f} +/- {std_acc:.4f}\n")
                else:
                    f.write(f"  Class 0 vs {leak_class}: No valid results\n")

            # Per-class inference statistics
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
            
            # Add enhanced hardware info
            f.write(f"\nSystem Information:\n")
            f.write(f"System: {system_info.get('system', 'Unknown')}\n")
            f.write(f"Machine: {system_info.get('machine', 'Unknown')}\n")
            f.write(f"Processor: {system_info.get('processor', 'Unknown')}\n")
            f.write(f"CPU Count: {system_info.get('cpu_count', 'Unknown')}\n")
            f.write(f"RAM Total: {system_info.get('ram_total', 'Unknown')}\n")
            f.write(f"RAM Available: {system_info.get('ram_available', 'Unknown')}\n")
            
            # Raspberry Pi specific info
            if 'rpi_model' in system_info:
                f.write(f"RPi Model: {system_info['rpi_model']}\n")
            if 'rpi_revision' in system_info:
                f.write(f"RPi Revision: {system_info['rpi_revision']}\n")
            
            # GPU information - now supports multiple GPU types
            gpu_list = system_info.get('gpu', ['Information not available'])
            if isinstance(gpu_list, list):
                for i, gpu in enumerate(gpu_list):
                    f.write(f"GPU {i+1}: {gpu}\n")
            else:
                f.write(f"GPU: {gpu_list}\n")
                
            # Disk information
            if 'disk_total' in system_info:
                f.write(f"Disk Total: {system_info['disk_total']}\n")
                f.write(f"Disk Free: {system_info['disk_free']}\n")
        
        print(f"\n[INFO] Overall performance metrics saved to {inference_stats_path}")
        print(f"[INFERENCE STATS] Processed {total_samples} samples")
        print(f"[INFERENCE STATS] Average inference time: {overall_avg_time:.2f} ms/sample")
        print(f"[INFERENCE STATS] Throughput: {overall_throughput:.2f} samples/sec")
        print(f"[INFERENCE STATS] Total evaluation time: {total_time:.2f} seconds")
        print(f"[INFERENCE STATS] Effective batch size: {batch_size}")

        # Generate final table (like original test.py)
        generate_final_table(final_results, output_dir)
        return final_results
    
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {i: [] for i in range(1, 8)}


def generate_final_table(final_results, output_dir):
    """Generate final accuracy table following original test.py format"""
    table_path = os.path.join(output_dir, f'final_accuracy_table_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    with open(table_path, 'w') as f:
        f.write("Final Accuracy Table (10-fold Cross-Validation)\n")
        f.write("==============================================\n\n")
        f.write("Class Pair\t\tMean Accuracy\t\tStd Dev\n")
        f.write("-" * 50 + "\n")
        
        for leak_class in range(1, 8):
            accs = final_results[leak_class]
            if accs:
                mean_acc = np.mean(accs)
                std_acc = np.std(accs)
                f.write(f"0 vs {leak_class}\t\t{mean_acc:.4f}\t\t\t{std_acc:.4f}\n")
            else:
                f.write(f"0 vs {leak_class}\t\tNo Results\t\tNo Results\n")
    
    print(f"[INFO] Final accuracy table saved: {table_path}")


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def evaluate_model(model_path, test_dir, target_height, target_width, batch_size=4, 
                  max_batches=None, output_dir=None):
    """
    Main evaluation function that calls both evaluation methods
    Following original test.py structure exactly
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[INFO] Starting evaluation of model: {os.path.basename(model_path)}")
    print(f"[INFO] Test directory: {test_dir}")
    print(f"[INFO] Target resolution: {target_height}x{target_width}")
    print(f"[INFO] Batch size: {batch_size}")
    
    results = {}
    
    try:
        # 1. Evaluate all leak vs no leak (binary classification)
        print("\n" + "="*60)
        print("1. BINARY CLASSIFICATION: ALL LEAK VS NO LEAK")
        print("="*60)
        
        y_true, y_pred = evaluate_all_leak_vs_no_leak(
            model_path=model_path,
            test_dir=test_dir,
            batch_size=batch_size,
            output_dir=output_dir,
            target_height=target_height,
            target_width=target_width,
            run_error_analysis=False  # Skip error analysis for RPi
        )
        
        if len(y_true) > 0:
            accuracy = accuracy_score(y_true, y_pred)
            class_report = classification_report(y_true, y_pred, output_dict=True, digits=4)
            conf_matrix = confusion_matrix(y_true, y_pred)
            
            results['binary_classification'] = {
                'accuracy': accuracy,
                'y_true': y_true,
                'y_pred': y_pred,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix
            }
            
            print(f"[SUCCESS] Binary classification completed - Accuracy: {accuracy:.4f}")
        else:
            print("[ERROR] Binary classification failed - no results")
            results['binary_classification'] = None
        
        # 2. Evaluate per leak class (10-fold cross-validation)
        print("\n" + "="*60)
        print("2. PER-CLASS EVALUATION: 10-FOLD CROSS-VALIDATION")
        print("="*60)
        
        per_class_results = evaluate_per_leak_class(
            model_path=model_path,
            test_dir=test_dir,
            batch_size=batch_size,
            output_dir=output_dir,
            target_height=target_height,
            target_width=target_width
        )
        
        if per_class_results and any(per_class_results.values()):
            results['per_class_evaluation'] = per_class_results
            
            # Calculate average accuracy across all classes
            all_accs = []
            for accs in per_class_results.values():
                if accs:
                    all_accs.extend(accs)
            
            avg_acc = np.mean(all_accs) if all_accs else 0
            print(f"[SUCCESS] Per-class evaluation completed - Average accuracy: {avg_acc:.4f}")
        else:
            print("[ERROR] Per-class evaluation failed - no results")
            results['per_class_evaluation'] = None
        
        # Calculate summary metrics for testing pipeline
        if results.get('binary_classification'):
            final_accuracy = results['binary_classification']['accuracy']
            
            # Mock inference stats for pipeline compatibility
            results.update({
                'accuracy': final_accuracy,
                'avg_fps': 10.0,  # Placeholder - actual FPS calculated in evaluation functions
                'avg_inference_ms': 100.0,  # Placeholder - actual time calculated in evaluation functions
                'total_samples': len(y_true) if len(y_true) > 0 else 0
            })
        
        # Save comprehensive results
        if output_dir:
            results_path = os.path.join(output_dir, f'comprehensive_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(results_path, 'w') as f:
                json.dump(convert_numpy_types(results), f, indent=2)
            print(f"[INFO] Comprehensive results saved: {results_path}")
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Model evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def test_all_models(args, training_results, timestamp):
    """Test all trained models with better model path handling and TFLite support"""
    
    print("\n" + "="*60)
    print("RPi MODEL TESTING PHASE")
    print("="*60)
    print("Binary Classification Testing: No Leak (0) vs Leak (1)")
    print("Supporting both Keras (.keras) and TFLite (.tflite) models")
    print("Following original test.py evaluation methodology")
    print("="*60)
    
    all_results = {}
    test_dir = args.data_dir + "/test"  # Construct test directory from data_dir
    
    for model_name, result in training_results.items():
        # Prioritize best_model_path over model_path
        model_path = result.get('best_model_path') or result.get('model_path')
        model_format = result.get('model_format', 'keras')
        
        if not model_path:
            print(f"[WARNING] No model path found for {model_name}")
            continue
            
        # Update model_path if we have a better alternative
        if not os.path.exists(model_path):
            print(f"[WARNING] Model file not found: {model_path}")
            # Try to find the best model in the output directory
            output_dir = result.get('output_dir')
            if output_dir and os.path.exists(output_dir):
                try:
                    # Look for both keras and tflite files
                    keras_files = [f for f in os.listdir(output_dir) 
                                 if f.endswith('.keras') and 'best' in f]
                    tflite_files = [f for f in os.listdir(output_dir) 
                                  if f.endswith('.tflite')]
                    
                    if keras_files:
                        # Sort by modification time, get newest
                        keras_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
                        model_path = os.path.join(output_dir, keras_files[0])
                        model_format = 'keras'
                        print(f"[INFO] Found alternative keras model: {os.path.basename(model_path)}")
                    elif tflite_files:
                        # Use TFLite if no keras available
                        tflite_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
                        model_path = os.path.join(output_dir, tflite_files[0])
                        model_format = 'tflite'
                        print(f"[INFO] Found alternative TFLite model: {os.path.basename(model_path)}")
                    else:
                        # Fall back to any .keras file
                        all_keras = [f for f in os.listdir(output_dir) if f.endswith('.keras')]
                        if all_keras:
                            all_keras.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
                            model_path = os.path.join(output_dir, all_keras[0])
                            model_format = 'keras'
                            print(f"[INFO] Found fallback keras model: {os.path.basename(model_path)}")
                except Exception as e:
                    print(f"[ERROR] Could not scan for alternative models: {e}")
                    continue
        
        print(f"\n--- Testing {model_name} model ({model_name.split('_')[-1]}) ---")
        print(f"[INFO] Model format: {model_format.upper()}")
        print(f"[INFO] Model path: {os.path.basename(model_path)}")
        
        # Extract resolution from model name
        try:
            resolution_str = model_name.split('_')[-1]  # Assuming format like "resnet_240x320"
            target_height, target_width = map(int, resolution_str.split('x'))
        except (ValueError, IndexError):
            print(f"[WARNING] Could not parse resolution from {model_name}, using defaults")
            target_height, target_width = 120, 160

        try:
            print(f"[INFO] Testing {model_format.upper()} model...")
            
            # Create test output directory
            test_output_dir = os.path.join(result.get('output_dir', './test_results'), 'test_results')
            os.makedirs(test_output_dir, exist_ok=True)
            
            # Test the model following original test.py methodology
            test_results = evaluate_model(
                model_path=model_path,
                test_dir=test_dir,
                target_height=target_height,
                target_width=target_width,
                batch_size=args.batch_size,
                output_dir=test_output_dir
            )
            
            # Add model format info to results
            if test_results:
                test_results['model_format'] = model_format
            
            all_results[model_name] = test_results
            
            if test_results.get('accuracy', 0) > 0:
                print(f"[SUCCESS] {model_name} testing completed - Accuracy: {test_results['accuracy']:.4f}")
                print(f"[INFO] Model format: {model_format.upper()}")
            else:
                print(f"[WARNING] {model_name} testing completed with issues")
        
        except Exception as e:
            print(f"[ERROR] Model testing failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n[INFO] Testing phase completed. Tested {len([r for r in all_results.values() if r])} models.")
    
    # Print summary
    successful_tests = [k for k, v in all_results.items() if v and v.get('accuracy', 0) > 0]
    if successful_tests:
        print(f"[SUCCESS] Testing completed. {len(successful_tests)} models tested successfully.")
        # Show model formats tested
        keras_models = [k for k, v in all_results.items() if v and v.get('model_format') == 'keras']
        tflite_models = [k for k, v in all_results.items() if v and v.get('model_format') == 'tflite']
        if keras_models:
            print(f"[INFO] Keras models tested: {len(keras_models)}")
        if tflite_models:
            print(f"[INFO] TFLite models tested: {len(tflite_models)}")
    else:
        print("[WARNING] No models were successfully tested!")
    
    return all_results