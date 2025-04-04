# train.py

import os
import tensorflow as tf
from src.model.model import cnn_3d_model
from src.loader.loader import DataGenerator, DataGeneratorThreeClass
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)
import datetime

def train_model(train_dir, val_dir, model_save_path, best_model_path, output_base_dir, batch_size, epochs, 
               target_height, target_width, three_class_mode=False, distance_filter=None):
    """
    Train a model for gas leak classification.
    
    Args:
        train_dir: Directory containing training data
        val_dir: Directory containing validation data
        model_save_path: Path to save the final model
        best_model_path: Path to save the best model during training
        output_base_dir: Base directory for outputs
        batch_size: Batch size for training
        epochs: Number of epochs to train
        target_height: Target image height
        target_width: Target image width
        three_class_mode: If True, train for three-class classification (small/medium/large leaks),
                          otherwise train for binary classification (leak/no-leak)
        distance_filter: Filter data by imaging distance: '46' for 4.6m, '69' for 6.9m, or None for all data
    
    Returns:
        model: Trained model
        history: Training history
    """
    mode_name = "three_class" if three_class_mode else "binary"
    num_classes = 3 if three_class_mode else 2
    
    # Add distance information to mode name if a filter is applied
    distance_info = f"_distance_{distance_filter}m" if distance_filter else ""
    print(f"[INFO] Starting {mode_name}{distance_info} classification training...")
    
    # 1) Build the model with appropriate number of output classes
    model = cnn_3d_model(input_shape=(15, target_height, target_width, 1), num_classes=num_classes)
    
    # 2) Create data generators - use appropriate generator based on mode
    if three_class_mode:
        train_gen = DataGeneratorThreeClass(
            data_dir=train_dir,
            batch_size=batch_size,
            shuffle=True,
            balance_classes=True,
            training=True,
            resize=True,
            target_height=target_height,
            target_width=target_width,
            distance_filter=distance_filter
        )
        
        val_gen = DataGeneratorThreeClass(
            data_dir=val_dir,
            batch_size=batch_size,
            shuffle=False,
            balance_classes=False,
            training=False,
            resize=True,
            target_height=target_height,
            target_width=target_width,
            distance_filter=distance_filter
        )
    else:
        # Original binary classification mode
        train_gen = DataGenerator(
            data_dir=train_dir,
            batch_size=batch_size,
            shuffle=True,
            binary_all_leak=True,
            balance_classes=True,
            training=True,
            resize=True,
            target_height=target_height,
            target_width=target_width
        )
        
        val_gen = DataGenerator(
            data_dir=val_dir,
            batch_size=batch_size,
            shuffle=False,
            binary_all_leak=True,
            balance_classes=False,
            training=False,
            resize=True,
            target_height=target_height,
            target_width=target_width
        )
    
    # 3) Optional: Warmup execution on a single batch
    try:
        warmup_data, warmup_labels = next(iter(train_gen))
        # Perform one forward+backward pass on that batch
        model.train_on_batch(warmup_data, warmup_labels)
        print("[INFO] Warmup pass completed.")
    except StopIteration:
        print("[WARNING] Train generator is empty. Skipping warmup.")
    
    # 4) Define callbacks
    checkpoint_cb = ModelCheckpoint(
        filepath=best_model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    earlystop_cb = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_scheduler_cb = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=5,
        verbose=1,
        min_lr=1e-7
    )
    
    callbacks_list = [checkpoint_cb, earlystop_cb, lr_scheduler_cb]
    
    # 5) Train
    print(f"[INFO] Starting {mode_name}{distance_info} model training: {epochs} epochs with batch size {batch_size}")
    print(f"[INFO] Input shape: (15, {target_height}, {target_width}, 1), Output classes: {num_classes}")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # 6) Save final model
    save_dir = os.path.dirname(model_save_path)
    os.makedirs(save_dir, exist_ok=True)
    model.save(model_save_path)
    print(f"[INFO] {mode_name.capitalize()}{distance_info} model saved to {model_save_path}")
    print(f"[INFO] Best model (lowest val_loss) saved to {best_model_path}")
    
    # 7) Evaluate validation set
    print(f"\n[INFO] Evaluating {mode_name}{distance_info} model on validation set...")
    results = model.evaluate(val_gen, verbose=1)
    print(f"{mode_name.capitalize()}{distance_info} model - Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}")
    
    # 8) Save evaluation results to text file with current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    distance_suffix = f"_distance_{distance_filter}m" if distance_filter else ""
    eval_result_path = os.path.join(output_base_dir, f"{mode_name}{distance_suffix}_eval_{current_date}_{target_height}x{target_width}.txt")
    
    with open(eval_result_path, 'w') as f:
        f.write(f"{mode_name.capitalize()}{distance_info} Evaluation Date: {current_date}\n")
        f.write(f"Model Type: {mode_name.capitalize()} Classification\n")
        f.write(f"Num Classes: {num_classes}\n")
        if distance_filter:
            f.write(f"Distance Filter: {distance_filter}m\n")
        if three_class_mode:
            f.write("Class Mapping:\n")
            f.write("  - Class 0: Small Leak (original classes 0-2)\n")
            f.write("  - Class 1: Medium Leak (original classes 3-5)\n")
            f.write("  - Class 2: Large Leak (original classes 6-7)\n")
        f.write(f"Loss: {results[0]:.4f}\n")
        f.write(f"Accuracy: {results[1]:.4f}\n")
    
    print(f"[INFO] Evaluation results saved to {eval_result_path}")
    
    # 9) Save training history
    history_dir = os.path.join(output_base_dir, "history")
    os.makedirs(history_dir, exist_ok=True)
    
    history_file = os.path.join(history_dir, f"{mode_name}{distance_suffix}_history_{current_date}_{target_height}x{target_width}.txt")
    with open(history_file, 'w') as f:
        f.write(f"{mode_name.capitalize()}{distance_info} Training History\n")
        f.write("="*30 + "\n\n")
        f.write(f"Resolution: {target_height}x{target_width}\n")
        if distance_filter:
            f.write(f"Distance Filter: {distance_filter}m\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Epochs Trained: {len(history.history['loss'])}\n\n")
        
        for epoch in range(len(history.history['loss'])):
            f.write(f"Epoch {epoch+1}:\n")
            f.write(f"  Loss: {history.history['loss'][epoch]:.4f}\n")
            f.write(f"  Accuracy: {history.history['accuracy'][epoch]:.4f}\n")
            f.write(f"  Val Loss: {history.history['val_loss'][epoch]:.4f}\n")
            f.write(f"  Val Accuracy: {history.history['val_accuracy'][epoch]:.4f}\n")
            f.write("\n")
    
    print(f"[INFO] Training history saved to {history_file}")
    
    return model, history

# Keep the original function name for backward compatibility
def train_binary(train_dir, val_dir, model_save_path, best_model_path, output_base_dir, batch_size, epochs, 
                target_height, target_width):
    """Wrapper around train_model for backward compatibility"""
    return train_model(train_dir, val_dir, model_save_path, best_model_path, output_base_dir, 
                      batch_size, epochs, target_height, target_width, three_class_mode=False)
