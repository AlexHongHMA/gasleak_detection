"""
rpi_main.py - Main orchestrator for RPi pipeline with enhanced model support
Enhanced support for ResNet, EfficientNet, and knowledge distillation
"""

import os
import argparse
import json
import numpy as np
from datetime import datetime

from rpi_train import train_all_models
from rpi_test import test_all_models


def parse_arguments():
    """Parse command line arguments with enhanced model support"""
    parser = argparse.ArgumentParser(description='RPi-Optimized VideoGasNet Pipeline with Knowledge Distillation')
    
    # Data configuration
    parser.add_argument('--data_dir', type=str, 
                       default='./data/processed_optical_flow_runavg_gray/',
                       help='Data directory')
    parser.add_argument('--result_dir', type=str, 
                       default='./result/rpi_results',
                       help='Results directory')
    
    # Model configuration  
    parser.add_argument('--model_types', nargs='+', 
                       choices=['resnet', 'efficientnet', 'enhanced_lightweight'],
                       default=['enhanced_lightweight'],
                       help='Model architectures to train/test')
    parser.add_argument('--resolutions', nargs='+',
                       default=['120x160'],
                       help='Resolutions to use (format: HxW)')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (increased for knowledge distillation)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--target_frames', type=int, default=10,
                       help='Number of frames to process')
    
    # Knowledge distillation configuration
    parser.add_argument('--teacher_model', type=str,
                       help='Path to teacher model for knowledge distillation')
    parser.add_argument('--enable_distillation', action='store_true',
                       help='Enable knowledge distillation training')
    
    # Pipeline control
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training phase')
    parser.add_argument('--skip_testing', action='store_true',
                       help='Skip testing phase')
    parser.add_argument('--max_test_batches', type=int,
                       help='Maximum test batches to process (for quick testing)')
    
    parser.add_argument('--interactive', 
                       action='store_true',
                       help='Enable interactive prompts for user choices')
    
    parser.add_argument('--skip-tflite', 
                       action='store_true',
                       help='Skip TFLite model creation')
    
    return parser.parse_args()


def validate_teacher_model(teacher_model_path):
    """Validate teacher model exists and is compatible"""
    if not teacher_model_path:
        return False, "No teacher model path provided"
    
    if not os.path.exists(teacher_model_path):
        return False, f"Teacher model not found: {teacher_model_path}"
    
    try:
        import tensorflow as tf
        # Try to load the model to verify it's valid
        model = tf.keras.models.load_model(teacher_model_path)
        model_info = {
            'params': model.count_params(),
            'input_shape': model.input_shape,
            'output_shape': model.output_shape
        }
        return True, f"Teacher model validated: {model_info['params']:,} parameters"
    except Exception as e:
        return False, f"Teacher model validation failed: {e}"


def find_existing_models(args):
    """Find existing models when skipping training with better model path handling"""
    
    print("[INFO] Looking for existing models...")
    models = {}
    
    for model_type in args.model_types:
        for resolution_str in args.resolutions:
            model_dir = os.path.join(args.result_dir, f"rpi_{model_type}_{resolution_str}")
            
            if not os.path.exists(model_dir):
                print(f"[WARNING] Model directory not found: {model_dir}")
                continue
            
            # Find latest models with better prioritization
            try:
                files = os.listdir(model_dir)
                # Prioritize "best" models first
                best_keras_files = [f for f in files if f.endswith('.keras') and 'best' in f]
                all_keras_files = [f for f in files if f.endswith('.keras')]
                tflite_files = [f for f in files if f.endswith('.tflite')]
                
                model_path = None
                model_type_found = None
                
                if best_keras_files:
                    # Sort by modification time, get newest
                    best_keras_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                    model_path = os.path.join(model_dir, best_keras_files[0])
                    model_type_found = 'keras'
                    print(f"[INFO] Found best keras model: {best_keras_files[0]}")
                elif all_keras_files:
                    # Fall back to any keras file
                    all_keras_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                    model_path = os.path.join(model_dir, all_keras_files[0])
                    model_type_found = 'keras'
                    print(f"[INFO] Found keras model: {all_keras_files[0]}")
                elif tflite_files:
                    # Use TFLite model if no Keras models available
                    tflite_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                    model_path = os.path.join(model_dir, tflite_files[0])
                    model_type_found = 'tflite'
                    print(f"[INFO] Found TFLite model: {tflite_files[0]}")
                
                if model_path:
                    quantized_path = None
                    if model_type_found == 'keras' and tflite_files:
                        tflite_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                        quantized_path = os.path.join(model_dir, tflite_files[0])
                    elif model_type_found == 'tflite':
                        quantized_path = model_path  # TFLite model is the quantized version
                    
                    key = f"{model_type}_{resolution_str}"
                    models[key] = {
                        'model_type': model_type,
                        'model_path': model_path,
                        'best_model_path': model_path,  # Set both to the same path
                        'quantized_path': quantized_path,
                        'output_dir': model_dir,
                        'classification_type': 'binary',
                        'final_accuracy': 0.0,  # Will be updated during testing
                        'knowledge_distillation': 'unknown',  # Can't determine from filename
                        'model_format': model_type_found  # Track if it's keras or tflite
                    }
                    
                    print(f"[INFO] Found {key}:")
                    if model_type_found == 'keras':
                        print(f"  Keras: {os.path.basename(model_path)}")
                    else:
                        print(f"  TFLite: {os.path.basename(model_path)}")
                    if quantized_path and quantized_path != model_path:
                        print(f"  TFLite: {os.path.basename(quantized_path)}")
                else:
                    print(f"[WARNING] No models found in {model_dir}")
            
            except Exception as e:
                print(f"[ERROR] Error scanning {model_dir}: {e}")
                continue
    
    if not models:
        print("[WARNING] No existing models found!")
    else:
        print(f"[INFO] Found {len(models)} model configurations")
    
    return models


def save_summary(training_results, testing_results, args, timestamp):
    """Save enhanced pipeline summary with proper type checking"""
    
    summary_dir = os.path.join(args.result_dir, "summaries")
    os.makedirs(summary_dir, exist_ok=True)
    
    summary_path = os.path.join(summary_dir, f"rpi_summary_{timestamp}.txt")
    json_path = os.path.join(summary_dir, f"rpi_results_{timestamp}.json")
    
    # Create summary
    with open(summary_path, 'w') as f:
        f.write("Enhanced RPi VideoGasNet Pipeline Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Pipeline ID: {timestamp}\n")
        f.write(f"Classification: Binary (No Leak vs Leak)\n")
        f.write(f"Architectures: ResNet, EfficientNet, Enhanced Lightweight\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Model Types: {', '.join(args.model_types)}\n")
        f.write(f"  Resolutions: {', '.join(args.resolutions)}\n")
        f.write(f"  Target Frames: {args.target_frames}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        
        # Knowledge distillation info - safe check
        teacher_model = getattr(args, 'teacher_model', None)
        if teacher_model:
            f.write(f"  Knowledge Distillation: Enabled\n")
            f.write(f"  Teacher Model: {os.path.basename(teacher_model)}\n")
        else:
            f.write(f"  Knowledge Distillation: Disabled\n")
        f.write("\n")
        
        if training_results:
            f.write("Training Results:\n")
            for key, result in training_results.items():
                f.write(f"  {key}:\n")
                
                # Safe formatting for accuracy
                accuracy = result.get('final_accuracy', 'N/A')
                if isinstance(accuracy, (int, float)):
                    f.write(f"    Accuracy: {accuracy:.4f}\n")
                else:
                    f.write(f"    Accuracy: {accuracy}\n")
                
                # Safe formatting for parameters
                params = result.get('total_params', 'N/A')
                if isinstance(params, (int, float)):
                    f.write(f"    Parameters: {params:,}\n")
                else:
                    f.write(f"    Parameters: {params}\n")
                
                # Safe formatting for training time
                training_time = result.get('training_time', 'N/A')
                if isinstance(training_time, (int, float)):
                    f.write(f"    Training Time: {training_time:.1f}s\n")
                else:
                    f.write(f"    Training Time: {training_time}\n")
                
                # Knowledge distillation status
                kd_status = result.get('knowledge_distillation', 'N/A')
                f.write(f"    Knowledge Distillation: {kd_status}\n")
            f.write("\n")
        
        if testing_results:
            f.write("Testing Results:\n")
            for key, result in testing_results.items():
                f.write(f"  {key}:\n")
                
                # Safe formatting for all metrics
                accuracy = result.get('accuracy', 0)
                fps = result.get('avg_fps', 0)
                inference_ms = result.get('avg_inference_ms', 0)
                samples = result.get('total_samples', 0)
                
                # Format with type checking
                if isinstance(accuracy, (int, float)):
                    f.write(f"    Accuracy: {accuracy:.4f}\n")
                else:
                    f.write(f"    Accuracy: {accuracy}\n")
                
                if isinstance(fps, (int, float)):
                    f.write(f"    FPS: {fps:.1f}\n")
                else:
                    f.write(f"    FPS: {fps}\n")
                
                if isinstance(inference_ms, (int, float)):
                    f.write(f"    Inference: {inference_ms:.2f}ms\n")
                else:
                    f.write(f"    Inference: {inference_ms}\n")
                
                if isinstance(samples, (int, float)):
                    f.write(f"    Samples: {samples}\n")
                else:
                    f.write(f"    Samples: {samples}\n")
            f.write("\n")
    
    # Save JSON with error handling
    all_results = {
        'timestamp': timestamp,
        'config': {
            'model_types': args.model_types,
            'resolutions': args.resolutions,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'target_frames': args.target_frames,
            'data_dir': args.data_dir,
            'result_dir': args.result_dir,
            'teacher_model': getattr(args, 'teacher_model', None),
            'knowledge_distillation_enabled': bool(getattr(args, 'teacher_model', None))
        },
        'training_results': training_results,
        'testing_results': testing_results
    }
    
    try:
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
    except Exception as e:
        print(f"[WARNING] Failed to save JSON results: {e}")
        json_path = None
    
    return summary_path, json_path
    
    # Save JSON with error handling
    all_results = {
        'timestamp': timestamp,
        'config': {
            'model_types': args.model_types,
            'resolutions': args.resolutions,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'target_frames': args.target_frames,
            'data_dir': args.data_dir,
            'result_dir': args.result_dir,
            'teacher_model': getattr(args, 'teacher_model', None),
            'knowledge_distillation_enabled': bool(getattr(args, 'teacher_model', None))
        },
        'training_results': training_results,
        'testing_results': testing_results
    }
    
    try:
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
    except Exception as e:
        print(f"[WARNING] Failed to save JSON results: {e}")
        json_path = None
    
    return summary_path, json_path


def main():
    """Enhanced main pipeline function"""
    
    # Parse arguments
    args = parse_arguments()
    timestamp = datetime.now().strftime("%Y%m%d")
    
    print("Enhanced RPi VideoGasNet Pipeline")
    print("=" * 60)
    print("Binary Classification: No Leak (0) vs Leak (1-7)")
    print("Enhanced with ResNet, EfficientNet & Knowledge Distillation")
    print("=" * 60)
    print(f"Models: {', '.join(args.model_types)}")
    print(f"Resolutions: {', '.join(args.resolutions)}")
    print(f"Data: {args.data_dir}")
    print(f"Results: {args.result_dir}")
    
    # Validate teacher model if provided
    if args.teacher_model:
        valid, message = validate_teacher_model(args.teacher_model)
        if valid:
            print(f"[INFO] {message}")
        else:
            print(f"[WARNING] {message}")
            response = input("Continue without knowledge distillation? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                return
            args.teacher_model = None
    
    print("=" * 60)
    
    # Create results directory
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Set create_tflite based on arguments
    if args.skip_tflite:
        create_tflite = False
        print("[INFO] TFLite model creation disabled by user")
    else:
        create_tflite = True
    
    # Store in args for consistency
    args.create_tflite = create_tflite
    
    # Training phase
    training_results = {}
    if not args.skip_training:
        print("\n[INFO] Starting enhanced training phase...")
        if args.teacher_model:
            print(f"[INFO] Knowledge distillation enabled with teacher: {os.path.basename(args.teacher_model)}")
        else:
            print("[INFO] Standard training (no knowledge distillation)")
            
        training_results = train_all_models(args, timestamp)
        
        if training_results:
            print(f"[SUCCESS] Training completed. {len(training_results)} models trained.")
            
            # Print training summary
            print(f"\n[SUMMARY] Training Results:")
            for model_name, result in training_results.items():
                kd_status = "with KD" if result.get('knowledge_distillation', False) else "standard"
                print(f"  {model_name}: {result.get('final_accuracy', 0):.4f} ({kd_status})")
        else:
            print("[WARNING] No models were successfully trained!")
    else:
        print("\n[INFO] Skipping training phase, looking for existing models...")
        training_results = find_existing_models(args)
    
    # Testing phase
    testing_results = {}
    if not args.skip_testing and training_results:
        print("\n[INFO] Starting testing phase...")
        args.test_dir = os.path.join(args.data_dir, "test")
        
        testing_results = test_all_models(args, training_results, timestamp)
        
        if testing_results:
            print(f"[SUCCESS] Testing completed. {len(testing_results)} models tested.")
            
            # Print testing summary
            print(f"\n[SUMMARY] Testing Results:")
            for model_name, result in testing_results.items():
                print(f"  {model_name}: {result.get('accuracy', 0):.4f} accuracy, {result.get('avg_fps', 0):.1f} FPS")
        else:
            print("[WARNING] No models were successfully tested!")
    elif args.skip_testing:
        print("\n[INFO] Skipping testing phase.")
    else:
        print("\n[WARNING] No models available for testing.")
    
    # Generate summary
    if training_results or testing_results:
        try:
            summary_path, json_path = save_summary(training_results, testing_results, args, timestamp)
            
            print(f"\n" + "="*60)
            print("ENHANCED PIPELINE COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"Pipeline ID: {timestamp}")
            print(f"Summary: {summary_path}")
            if json_path:
                print(f"Data: {json_path}")
            
            if testing_results:
                # Show best results
                best_accuracy = max(testing_results.items(), 
                                  key=lambda x: x[1].get('accuracy', 0))
                fastest_model = max(testing_results.items(), 
                                  key=lambda x: x[1].get('avg_fps', 0))
                
                print(f"\nBest Accuracy:")
                print(f"  Model: {best_accuracy[0]}")
                print(f"  Accuracy: {best_accuracy[1].get('accuracy', 0):.4f}")
                print(f"  FPS: {best_accuracy[1].get('avg_fps', 0):.2f}")
                
                print(f"\nFastest Model:")
                print(f"  Model: {fastest_model[0]}")
                print(f"  FPS: {fastest_model[1].get('avg_fps', 0):.2f}")
                print(f"  Accuracy: {fastest_model[1].get('accuracy', 0):.4f}")
            
            # Knowledge distillation summary
            if args.teacher_model and training_results:
                kd_models = [k for k, v in training_results.items() 
                           if v.get('knowledge_distillation', False)]
                if kd_models:
                    print(f"\nKnowledge Distillation Summary:")
                    print(f"  Models trained with KD: {len(kd_models)}")
                    avg_kd_accuracy = np.mean([training_results[k]['final_accuracy'] 
                                             for k in kd_models])
                    print(f"  Average KD accuracy: {avg_kd_accuracy:.4f}")
            
            print("="*60)
            
        except Exception as e:
            print(f"[ERROR] Failed to generate summary: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[WARNING] No results to summarize")
    
    print(f"\n[INFO] Enhanced RPi pipeline completed: {timestamp}")


if __name__ == "__main__":
    main()