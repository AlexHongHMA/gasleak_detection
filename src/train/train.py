# train.py

import os
import tensorflow as tf
from src.model.model import cnn_3d_model
from src.loader.loader import DataGenerator
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)

def train_binary(train_dir, val_dir, model_save_path, best_model_path, batch_size, epochs):
    # 1) Build the model
    #    Input shape = (T, H, W, C), e.g. (15, 240, 320, 1)
    model = cnn_3d_model(input_shape=(15, 240, 320, 1), num_classes=2)

    # 2) Create data generators with balanced sampling
    train_gen = DataGenerator(
        data_dir=train_dir,
        batch_size=batch_size,
        shuffle=True,
        binary_all_leak=True,
        balance_classes=True  # Enable class balancing
    )
    val_gen = DataGenerator(
        data_dir=val_dir,
        batch_size=batch_size,
        shuffle=False,
        binary_all_leak=True,
        balance_classes=True  # Enable class balancing
    )

    # Calculate class weights
    no_leak_samples = sum(1 for path, label in train_gen.filepaths if label == 0)
    leak_samples = sum(1 for path, label in train_gen.filepaths if label != 0)
    total_samples = no_leak_samples + leak_samples
    
    class_weights = {
        0: total_samples / (2 * no_leak_samples),
        1: total_samples / (2 * leak_samples)
    }
    
    print(f"[INFO] Class distribution:")
    print(f"  No leak (0): {no_leak_samples} samples")
    print(f"  Leak (1): {leak_samples} samples")
    print(f"[INFO] Using class weights: {class_weights}")

    # 3) Optional: Warmup execution on a single batch
    #    This helps the model do a forward/backward pass so the graph is "built" and GPU is primed
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
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks_list,
        class_weight=class_weights,
        verbose=1
    )

    # 6) Save final model
    model.save(model_save_path)
    print(f"[INFO] Model saved to {model_save_path}")
    print("[INFO] Best model (lowest val_loss) saved to", best_model_path)