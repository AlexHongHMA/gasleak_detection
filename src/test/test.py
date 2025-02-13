import os
import numpy as np
import tensorflow as tf
from src.model.model import cnn_3d_model
from src.loader.loader import DataGenerator

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time 
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def save_confusion_matrix(y_true, y_pred, class_names, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'confusion_matrix_{timestamp}.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")

def save_classification_report(y_true, y_pred, output_dir):
    report = classification_report(y_true, y_pred, digits=4)

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'classification_report_{timestamp}.txt')

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Classification report saved to {output_path}")


def evaluate_all_leak_vs_no_leak(model_path, test_dir, batch_size, output_dir):
    """
    Evaluate class 0 vs [1..7] in one go (i.e., binary_all_leak=True).
    """
    print("[INFO] Evaluate no-leak(0) vs. leak(1..7)...")
    model = cnn_3d_model(input_shape=(15, 240, 320, 1), num_classes=2)
    model.load_weights(model_path)

    print("[INFO] Warmup GPU with a dummy input")
    dummy_input = np.random.rand(1, 15, 240, 320, 1).astype(np.float32)
    _ = model.predict(dummy_input, verbose=1)

    test_gen = DataGenerator(
        data_dir=test_dir,
        batch_size=batch_size,
        shuffle=False,
        binary_all_leak=True,
        balance_classes=True  # Enable class balancing
    )   

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

        start_time = time.time()
        preds = model.predict(X_batch, verbose=1)
        y_pred.extend(preds.argmax(axis=-1))
        y_true.extend(y_batch)

    # Save results
    class_names = ['No Leak', 'Leak']
    save_confusion_matrix(y_true, y_pred, class_names, output_dir)
    save_classification_report(y_true, y_pred, output_dir)

    return y_true, y_pred

def evaluate_per_leak_class(model_path, test_dir, batch_size, output_dir):
    """
    Perform 10-fold testing for each leak class vs. no leak (0..k).
    Only saves the final accuracy table.
    """
    print("[INFO] Evaluate each leak class vs. no-leak (0), with 10-fold testing...")

    # Build a single model & load weights once
    base_model = cnn_3d_model(input_shape=(15, 240, 320, 1), num_classes=2)
    base_model.load_weights(model_path)

    final_results = {i: [] for i in range(1, 8)}  # Store accuracies for each class

    for leak_class in range(1, 8):
        print(f"\n--- 0 vs {leak_class} (10-fold) ---")
        
        # Create balanced generator for this leak class
        temp_gen = DataGenerator(
            data_dir=test_dir,
            batch_size=batch_size,
            shuffle=False,
            binary_all_leak=False,
            binary_pair=(0, leak_class),
            balance_classes=True
        )

        if len(temp_gen.filepaths) == 0:
            print(f"Skipping 0 vs {leak_class} - no data found.")
            continue

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
                balance_classes=True
            )
            
            y_true, y_pred = [], []
            for X_batch, y_batch in fold_gen:
                if len(X_batch) == 0:
                    break
                preds = base_model.predict(X_batch, verbose=0)
                y_pred.extend(preds.argmax(axis=-1))
                y_true.extend(y_batch)
            
            if len(y_true) > 0:
                acc = accuracy_score(y_true, y_pred)
                fold_accuracies.append(acc)
                print(f"Fold {fold_i+1}/10 | Accuracy: {acc:.4f}")
        
        # Store and print class summary
        final_results[leak_class] = fold_accuracies
        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        print(f"\n0 vs {leak_class} | Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    # Generate final table
    generate_final_table(final_results, output_dir)
    return final_results

def generate_final_table(final_results, output_dir):
    """
    Generate the final accuracy comparison table from 10-fold results.
    """
    # Calculate mean accuracies
    table_data = {}
    for leak_class in range(1, 8):
        accuracies = final_results[leak_class]
        if len(accuracies) == 0:
            table_data[leak_class] = 0.0
        else:
            table_data[leak_class] = np.mean(accuracies) * 100  # Convert to percentage
    
    # Create formatted table
    table_filename = os.path.join(output_dir, "accuracy_comparison_table.txt")
    with open(table_filename, "w") as f:
        f.write("Table 2.5: Accuracy comparison in the leak vs. non-leak binary problem\n")
        f.write("Architecture |  0-1   0-2   0-3   0-4   0-5   0-6   0-7\n")
        f.write("------------------------------------------------------\n")
        
        row_str = "VideoGasNet  |"
        for leak_class in range(1, 8):
            acc = table_data[leak_class]
            row_str += f" {acc:6.1f}% "
        f.write(row_str + "\n")
    
    print(f"\nFinal table saved to: {table_filename}")

def test_binary(test_dir, model_path, batch_size, output_base_dir="./result", method_name="unknown"):
    """
    Master function to test the binary model in 2 ways:
     1) all leak vs. no leak
     2) 10-fold each leak class vs. no leak
    Called from main.py
    """
    # Create method-specific output directory
    output_dir = os.path.join(output_base_dir, method_name)
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories for different tests
    all_leak_dir = os.path.join(output_dir, "all_leak_test")
    fold_test_dir = os.path.join(output_dir, "10fold_test")
    os.makedirs(all_leak_dir, exist_ok=True)
    os.makedirs(fold_test_dir, exist_ok=True)

    print(f"\n_______ Testing all leaks vs no-leak for {method_name} _______")
    evaluate_all_leak_vs_no_leak(model_path, test_dir, batch_size, all_leak_dir)

    print(f"\n_______ 10-Fold testing for each leak class vs no-leak for {method_name} _______")
    final_results = evaluate_per_leak_class(model_path, test_dir, batch_size, fold_test_dir)

    # Generate final table
    print(f"\n_______ Generating final table for {method_name} _______")
    generate_final_table(final_results, output_dir)

    # Create a summary file
    summary_path = os.path.join(output_dir, f"{method_name}_test_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Test Summary for {method_name}\n")
        f.write("================================\n\n")
        f.write(f"All leak vs no-leak results: {all_leak_dir}\n")
        f.write(f"10-fold test results: {fold_test_dir}\n")
        f.write(f"Final table location: {output_dir}\n")
