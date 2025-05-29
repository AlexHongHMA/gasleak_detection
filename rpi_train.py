"""
rpi_train.py - Training module for RPi-optimized models
Following original train.py structure with correct data generators
"""

import os
import time
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import classification_report

from rpi_optimized_model import get_rpi_model, quantize_model, FrameSampler
from src.loader.loader import DataGenerator


def create_representative_dataset(data_gen, num_samples=100):
    """Create representative dataset for quantization"""
    
    def representative_data_gen():
        for i in range(min(num_samples, len(data_gen))):
            try:
                batch_x, _ = data_gen[i]
                if len(batch_x) > 0:
                    # Use first sample from batch
                    sample = batch_x[0:1].astype(np.float32)
                    yield [sample]
            except:
                continue
    
    return representative_data_gen


def train_model(model_type, data_dir, output_dir, input_shape, epochs=50, 
                batch_size=16, target_frames=8, teacher_model_path=None):
    """
    Train a single RPi model for binary classification
    
    Args:
        model_type: Type of model to train
        data_dir: Directory containing training data
        output_dir: Directory to save results
        input_shape: Input shape for the model
        epochs: Number of training epochs
        batch_size: Training batch size
        target_frames: Number of frames to process
        teacher_model_path: Path to teacher model for knowledge distillation
    
    Returns:
        Dictionary with training results
    """
    
    print(f"\n[INFO] Training {model_type} model for binary classification...")
    print(f"[INFO] Input shape: {input_shape}")
    print(f"[INFO] Target frames: {target_frames}")
    
    # Create data generators using the existing DataGenerator
    train_gen = DataGenerator(
        data_dir=os.path.join(data_dir, "train"),
        batch_size=batch_size,
        shuffle=True,
        binary_all_leak=True,  # Binary classification: 0=no leak, 1=leak
        balance_classes=True,
        training=True,
        resize=True,
        target_height=input_shape[1],
        target_width=input_shape[2]
    )
    
    val_gen = DataGenerator(
        data_dir=os.path.join(data_dir, "val"),
        batch_size=batch_size,
        shuffle=False,
        binary_all_leak=True,
        balance_classes=False,  # Don't balance validation set
        training=False,  # No augmentation for validation
        resize=True,
        target_height=input_shape[1],
        target_width=input_shape[2]
    )
    
    print(f"[INFO] Training batches: {len(train_gen)}")
    print(f"[INFO] Validation batches: {len(val_gen)}")
    
    # Create model for binary classification
    model = get_rpi_model(
        model_type=model_type,
        input_shape=input_shape,
        target_frames=target_frames,
        num_classes=2  # Binary classification
    )
    
    print(f"[INFO] Model parameters: {model.count_params():,}")
    
    # Setup callbacks
    timestamp = datetime.now().strftime("%Y%m%d")
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_dir, f"rpi_{model_type}_best_{timestamp}.keras"),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks = [model_checkpoint, early_stopping, reduce_lr]
    
    # Train model
    print(f"[INFO] Starting training for {epochs} epochs...")
    start_time = time.time()
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Save final model
    final_model_path = os.path.join(output_dir, f"rpi_{model_type}_final_{timestamp}.keras")
    model.save(final_model_path)
    
    # Get final metrics
    final_accuracy = max(history.history['val_accuracy'])
    
    # Quantize model for RPi deployment
    quantized_path = None
    if not os.environ.get('SKIP_QUANTIZATION'):
        try:
            print("[INFO] Creating quantized version for RPi deployment...")
            quantized_path = os.path.join(output_dir, f"rpi_{model_type}_quantized_{timestamp}.tflite")
            
            # Create representative dataset for quantization
            repr_gen = create_representative_dataset(train_gen, num_samples=50)
            
            quantized_path = quantize_model(
                model=model,
                output_path=quantized_path,
                representative_data_generator=repr_gen
            )
            
            if quantized_path:
                print(f"[INFO] Quantized model saved: {quantized_path}")
            
        except Exception as e:
            print(f"[WARNING] Quantization failed: {e}")
            quantized_path = None
    
    # Save training history
    history_path = os.path.join(output_dir, f"training_history_{timestamp}.json")
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, values in history.history.items():
            serializable_history[key] = [float(v) for v in values]
        json.dump(serializable_history, f, indent=2)
    
    # Generate training summary
    summary_path = os.path.join(output_dir, f"training_summary_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"RPi Model Training Summary - {timestamp}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Classification: Binary (No Leak vs Leak)\n")
        f.write(f"Input Shape: {input_shape}\n")
        f.write(f"Target Frames: {target_frames}\n")
        f.write(f"Total Parameters: {model.count_params():,}\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write(f"Best Validation Accuracy: {final_accuracy:.4f}\n\n")
        
        f.write("Training Configuration:\n")
        f.write(f"  Epochs: {epochs}\n")
        f.write(f"  Batch Size: {batch_size}\n")
        f.write(f"  Training Batches: {len(train_gen)}\n")
        f.write(f"  Validation Batches: {len(val_gen)}\n\n")
        
        f.write("Files Generated:\n")
        f.write(f"  Final Model: {os.path.basename(final_model_path)}\n")
        f.write(f"  Best Model: rpi_{model_type}_best_{timestamp}.keras\n")
        if quantized_path:
            f.write(f"  Quantized Model: {os.path.basename(quantized_path)}\n")
        f.write(f"  Training History: {os.path.basename(history_path)}\n")
    
    print(f"[INFO] Training summary saved: {summary_path}")
    
    # Return results
    results = {
        'model_type': model_type,
        'classification_type': 'binary',
        'model_path': final_model_path,
        'best_model_path': os.path.join(output_dir, f"rpi_{model_type}_best_{timestamp}.keras"),
        'quantized_path': quantized_path,
        'final_accuracy': float(final_accuracy),
        'training_time': training_time,
        'total_params': model.count_params(),
        'output_dir': output_dir,
        'history_path': history_path,
        'summary_path': summary_path
    }
    
    print(f"[SUCCESS] Training completed in {training_time:.2f}s")
    print(f"[INFO] Best validation accuracy: {final_accuracy:.4f}")
    
    return results


def train_all_models(args, timestamp):
    """Train all requested models"""
    
    print("\n" + "="*50)
    print("RPi MODEL TRAINING PHASE (RTX 4060)")
    print("="*50)
    print("Training for RPi4 deployment with binary classification")
    print("No Leak (0) vs Leak (1-7 mapped to 1)")
    print("="*50)
    
    all_results = {}
    
    for resolution_str in args.resolutions:
        target_height, target_width = map(int, resolution_str.split('x'))
        input_shape = (15, target_height, target_width, 1)
        
        print(f"\n--- Training for resolution {resolution_str} ---")
        
        for model_type in args.model_types:
            # Create output directory
            output_dir = os.path.join(args.result_dir, f"rpi_{model_type}_{resolution_str}")
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                # Train model
                result = train_model(
                    model_type=model_type,
                    data_dir=args.data_dir,
                    output_dir=output_dir,
                    input_shape=input_shape,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    target_frames=args.target_frames,
                    teacher_model_path=args.teacher_model
                )
                
                # Store results
                key = f"{model_type}_{resolution_str}"
                all_results[key] = result
                
                print(f"[SUCCESS] {key} training completed")
                
            except Exception as e:
                print(f"[ERROR] Training failed for {model_type}_{resolution_str}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n[INFO] Training phase completed. Trained {len(all_results)} models.")
    return all_results