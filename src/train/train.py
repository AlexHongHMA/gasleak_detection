# train.py

import os
import tensorflow as tf
from src.model.model import cnn_3d_model
from src.loader.loader import DataGenerator
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)
import datetime

def train_binary(train_dir, val_dir, model_save_path, best_model_path,output_base_dir, batch_size, epochs):
    # 1) Build the model
    #    Input shape = (T, H, W, C), e.g. (15, 240, 320, 1)
    model = cnn_3d_model(input_shape=(15, 240, 320, 1), num_classes=2)

    # 2) Create data generators with balanced sampling
    train_gen = DataGenerator(
        data_dir=train_dir,
        batch_size=batch_size,
        shuffle=True,
        binary_all_leak=True,
        balance_classes=True,  # Enable class balancing
        training=True  # Set to True for training
    )
    val_gen = DataGenerator(
        data_dir=val_dir,
        batch_size=batch_size,
        shuffle=False,
        binary_all_leak=True,
        balance_classes=True,  # Enable class balancing
        training=False  # Set to False for validation
    )

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
        verbose=1
    )

    # 6) Save final model
    model.save(model_save_path)
    print(f"[INFO] Model saved to {model_save_path}")
    print("[INFO] Best model (lowest val_loss) saved to", best_model_path)

    # # 8) Evaluate validation set
    # print("\n[INFO] Evaluating original model on validation set...")
    # results = model.evaluate(val_gen, verbose=1)
    # print(f"Original model - Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}")
    
    # # 9) Save evaluation results to text file with current date
    # current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    # eval_result_path = os.path.join(output_base_dir, f"eval_results_{current_date}.txt")
    
    # with open(eval_result_path, 'w') as f:
    #     f.write(f"Evaluation Date: {current_date}\n")
    #     f.write(f"Loss: {results[0]:.4f}\n")
    #     f.write(f"Accuracy: {results[1]:.4f}\n")
    
    # print(f"[INFO] Evaluation results saved to {eval_result_path}")
    
