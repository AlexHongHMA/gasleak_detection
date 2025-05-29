"""
rpi_main.py - Main orchestrator for RPi pipeline
Clean and simple following original main.py structure
"""

import os
import argparse
import json
from datetime import datetime

from rpi_train import train_all_models
from rpi_test import test_all_models


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='RPi-Optimized VideoGasNet Pipeline')
    
    # Data configuration
    parser.add_argument('--data_dir', type=str, 
                       default='./data/processed_optical_flow_runavg_gray/',
                       help='Data directory')
    parser.add_argument('--result_dir', type=str, 
                       default='./result/rpi_results',
                       help='Results directory')
    
    # Model configuration  
    parser.add_argument('--model_types', nargs='+', 
                       choices=['ultra', 'lightweight', 'mobilenet'],
                       default=['lightweight'],
                       help='Model types to train/test')
    parser.add_argument('--resolutions', nargs='+',
                       default=['240x320'],
                       help='Resolutions to use (format: HxW)')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--target_frames', type=int, default=8,
                       help='Number of frames to process')
    parser.add_argument('--teacher_model', type=str,
                       help='Path to teacher model for knowledge distillation')
    
    # Pipeline control
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training phase')
    parser.add_argument('--skip_testing', action='store_true',
                       help='Skip testing phase')
    parser.add_argument('--max_test_batches', type=int,
                       help='Maximum test batches to process (for quick testing)')
    
    return parser.parse_args()


def find_existing_models(args):
    """Find existing models when skipping training"""
    
    print("[INFO] Looking for existing models...")
    models = {}
    
    for model_type in args.model_types:
        for resolution_str in args.resolutions:
            model_dir = os.path.join(args.result_dir, f"rpi_{model_type}_{resolution_str}")
            
            if not os.path.exists(model_dir):
                print(f"[WARNING] Model directory not found: {model_dir}")
                continue
            
            # Find latest models
            try:
                files = os.listdir(model_dir)
                keras_files = [f for f in files if f.endswith('.keras') and 'best' not in f]
                tflite_files = [f for f in files if f.endswith('.tflite')]
                
                if keras_files:
                    # Sort by modification time, get newest
                    keras_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                    model_path = os.path.join(model_dir, keras_files[0])
                    
                    quantized_path = None
                    if tflite_files:
                        tflite_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                        quantized_path = os.path.join(model_dir, tflite_files[0])
                    
                    key = f"{model_type}_{resolution_str}"
                    models[key] = {
                        'model_type': model_type,
                        'model_path': model_path,
                        'quantized_path': quantized_path,
                        'output_dir': model_dir
                    }
                    
                    print(f"[INFO] Found {key}:")
                    print(f"  Keras: {os.path.basename(model_path)}")
                    if quantized_path:
                        print(f"  TFLite: {os.path.basename(quantized_path)}")
                else:
                    print(f"[WARNING] No keras models found in {model_dir}")
            
            except Exception as e:
                print(f"[ERROR] Error scanning {model_dir}: {e}")
                continue
    
    if not models:
        print("[WARNING] No existing models found!")
    else:
        print(f"[INFO] Found {len(models)} model configurations")
    
    return models


def save_summary(training_results, testing_results, args, timestamp):
    """Save pipeline summary"""
    
    summary_dir = os.path.join(args.result_dir, "summaries")
    os.makedirs(summary_dir, exist_ok=True)
    
    summary_path = os.path.join(summary_dir, f"rpi_summary_{timestamp}.txt")
    json_path = os.path.join(summary_dir, f"rpi_results_{timestamp}.json")
    
    # Create summary
    with open(summary_path, 'w') as f:
        f.write("RPi VideoGasNet Pipeline Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Pipeline ID: {timestamp}\n")
        f.write(f"Classification: Binary (No Leak vs Leak)\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Model Types: {', '.join(args.model_types)}\n")
        f.write(f"  Resolutions: {', '.join(args.resolutions)}\n")
        f.write(f"  Target Frames: {args.target_frames}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Batch Size: {args.batch_size}\n\n")
        
        if training_results:
            f.write("Training Results:\n")
            for key, result in training_results.items():
                f.write(f"  {key}:\n")
                f.write(f"    Accuracy: {result.get('final_accuracy', 'N/A'):.4f}\n")
                f.write(f"    Parameters: {result.get('total_params', 'N/A')}\n")
                f.write(f"    Training Time: {result.get('training_time', 'N/A'):.1f}s\n")
            f.write("\n")
        
        if testing_results:
            f.write("Testing Results:\n")
            for key, result in testing_results.items():
                f.write(f"  {key}:\n")
                f.write(f"    Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"    FPS: {result['avg_fps']:.1f}\n")
                f.write(f"    Inference: {result['avg_inference_ms']:.2f}ms\n")
                f.write(f"    Samples: {result['total_samples']}\n")
            f.write("\n")
        
        # Add deployment recommendations
        if testing_results:
            f.write("Deployment Recommendations:\n")
            
            # Find best overall (balance of accuracy and speed)
            best_overall = max(testing_results.items(), 
                             key=lambda x: x[1]['accuracy'] * (x[1]['avg_fps'] / 10))
            f.write(f"  Best Overall: {best_overall[0]}\n")
            f.write(f"    Accuracy: {best_overall[1]['accuracy']:.4f}\n")
            f.write(f"    FPS: {best_overall[1]['avg_fps']:.1f}\n")
            
            # Find fastest
            fastest = max(testing_results.items(), key=lambda x: x[1]['avg_fps'])
            f.write(f"  Fastest: {fastest[0]} ({fastest[1]['avg_fps']:.1f} FPS)\n")
            
            # Find most accurate
            most_accurate = max(testing_results.items(), key=lambda x: x[1]['accuracy'])
            f.write(f"  Most Accurate: {most_accurate[0]} ({most_accurate[1]['accuracy']:.4f})\n")
    
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
            'result_dir': args.result_dir
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
    """Main pipeline function"""
    
    # Parse arguments
    args = parse_arguments()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    print("RPi VideoGasNet Pipeline")
    print("=" * 50)
    print("Binary Classification: No Leak (0) vs Leak (1-7)")
    print("=" * 50)
    print(f"Models: {', '.join(args.model_types)}")
    print(f"Resolutions: {', '.join(args.resolutions)}")
    print(f"Data: {args.data_dir}")
    print(f"Results: {args.result_dir}")
    print("=" * 50)
    
    # Create results directory
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Training phase
    training_results = {}
    if not args.skip_training:
        print("\n[INFO] Starting training phase...")
        training_results = train_all_models(args, timestamp)
        
        if training_results:
            print(f"[SUCCESS] Training completed. {len(training_results)} models trained.")
        else:
            print("[WARNING] No models were successfully trained!")
    else:
        print("\n[INFO] Skipping training phase, looking for existing models...")
        training_results = find_existing_models(args)
    
    # Testing phase
    testing_results = {}
    if not args.skip_testing and training_results:
        print("\n[INFO] Starting testing phase...")
        testing_results = test_all_models(args, training_results, timestamp)
        
        if testing_results:
            print(f"[SUCCESS] Testing completed. {len(testing_results)} models tested.")
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
            
            print(f"\n" + "="*50)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*50)
            print(f"Pipeline ID: {timestamp}")
            print(f"Summary: {summary_path}")
            if json_path:
                print(f"Data: {json_path}")
            
            if testing_results:
                # Show best result
                best = max(testing_results.items(), 
                         key=lambda x: x[1]['avg_fps'] * x[1]['accuracy'])
                print(f"\nBest Overall Performance:")
                print(f"  Model: {best[0]}")
                print(f"  Accuracy: {best[1]['accuracy']:.4f}")
                print(f"  FPS: {best[1]['avg_fps']:.2f}")
                print(f"  Inference: {best[1]['avg_inference_ms']:.2f}ms")
            
            print("="*50)
            
        except Exception as e:
            print(f"[ERROR] Failed to generate summary: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[WARNING] No results to summarize")
    
    print(f"\n[INFO] RPi pipeline completed: {timestamp}")


if __name__ == "__main__":
    main()