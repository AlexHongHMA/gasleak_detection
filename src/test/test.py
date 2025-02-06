# import os
# import numpy as np
# import tensorflow as tf
# from src.model.model import cnn_3d_model
# from src.loader.loader import DataGenerator
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from datetime import datetime
# import matplotlib.pyplot as plt
# import seaborn as sns

# def save_confusion_matrix(y_true, y_pred, class_names, output_dir):
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=class_names, yticklabels=class_names)
#     plt.title('Confusion Matrix')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
    
#     # Create output directory if not exists
#     os.makedirs(output_dir, exist_ok=True)
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_path = os.path.join(output_dir, f'confusion_matrix_{timestamp}.png')
#     plt.savefig(output_path)
#     plt.close()
#     print(f"Confusion matrix saved to {output_path}")

# def save_classification_report(y_true, y_pred, output_dir):
#     report = classification_report(y_true, y_pred, digits=4)
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_path = os.path.join(output_dir, f'classification_report_{timestamp}.txt')
    
#     with open(output_path, 'w') as f:
#         f.write(report)
    
#     print(f"Classification report saved to {output_path}")

# def evaluate_all_leak_vs_no_leak(model_path, test_dir, batch_size, output_dir):

#     print("[INFO] Evaluate no-leak(0) vs. leak(1..7)...")
#     model = cnn_3d_model(input_shape=(15, 240, 320, 1), num_classes=2)
#     model.load_weights(model_path)

#     print("[INFO]Warmup GPU")
#     dummy_input = np.random.rand(1, 15, 240, 320, 1).astype(np.float32)
#     _ = model.predict(dummy_input, verbose=0)

#     test_gen = DataGenerator(
#         data_dir=test_dir,
#         batch_size=batch_size,
#         shuffle=False,
#         binary_all_leak=True
#     )

#     y_true, y_pred = [], []
#     for X_batch, y_batch in test_gen:
#         # Skip empty batches that cause errors
#         if len(X_batch) == 0:
#             continue
            
#         preds = model.predict(X_batch, verbose=0)
#         y_pred.extend(preds.argmax(axis=-1))
#         y_true.extend(y_batch)

#     # Save results
#     class_names = ['No Leak', 'Leak']
#     save_confusion_matrix(y_true, y_pred, class_names, output_dir)
#     save_classification_report(y_true, y_pred, output_dir)
    
#     return y_true, y_pred

# def evaluate_per_leak_class(model_path, test_dir, batch_size, output_dir):
#     print("[INFO] Evaluate each leak class vs. no-leak...")
#     model = cnn_3d_model(input_shape=(15, 240, 320, 1), num_classes=2)
#     model.load_weights(model_path)

#     for leak_class in range(1, 8):
#         test_gen = DataGenerator(
#             data_dir=test_dir,
#             batch_size=batch_size,
#             shuffle=False,
#             binary_all_leak=False,
#             binary_pair=(0, leak_class)
#         )

#         if not test_gen.filepaths:
#             print(f"Skipping 0 vs {leak_class} - no data")
#             continue

#         y_true, y_pred = [], []
#         for X_batch, y_batch in test_gen:
#             if len(X_batch) == 0:
#                 continue
                
#             preds = model.predict(X_batch, verbose=0)
#             y_pred.extend(preds.argmax(axis=-1))
#             y_true.extend(y_batch)

#         if len(y_true) == 0:
#             print(f"No samples found for 0 vs {leak_class}")
#             continue

#         print(f"\n--- 0 vs {leak_class} ---")
#         class_names = ['No Leak', f'Leak {leak_class}']
#         save_confusion_matrix(y_true, y_pred, class_names, 
#                              os.path.join(output_dir, f'class_{leak_class}'))
#         save_classification_report(y_true, y_pred, 
#                                   os.path.join(output_dir, f'class_{leak_class}'))

# def test_binary(test_dir, model_path, batch_size, output_base_dir="./result"):
#     # Create timestamped output directory
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_dir = os.path.join(output_base_dir, timestamp)
    
#     print("_______Testing all leaks vs no-leak_______")
#     evaluate_all_leak_vs_no_leak(model_path, test_dir, batch_size, output_dir)
    
#     print("\n_______Testing per-class leaks vs no-leak_______")
#     evaluate_per_leak_class(model_path, test_dir, batch_size, output_dir)

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

# def evaluate_all_leak_vs_no_leak(model_path, test_dir, batch_size, output_dir):
#     """
#     Evaluate class 0 vs [1..7] in one go (i.e., binary_all_leak=True).
#     Note: This can be slow if the test dataset is huge and we iterate in a Python loop.
#     """
#     print("[INFO] Evaluate no-leak(0) vs. leak(1..7)...")
#     model = cnn_3d_model(input_shape=(15, 240, 320, 1), num_classes=2)
#     model.load_weights(model_path)

#     print("[INFO] Warmup GPU with a dummy input")
#     dummy_input = np.random.rand(1, 15, 240, 320, 1).astype(np.float32)
#     _ = model.predict(dummy_input, verbose=1)

#     test_gen = DataGenerator(
#         data_dir=test_dir,
#         batch_size=batch_size,
#         shuffle=False,
#         binary_all_leak=True  # merges classes [1..7] => label=1, class 0 => 0
#     )   

#     total_batches = len(test_gen)
#     print(f"[INFO] Total batches to process: {total_batches}")

#     # Verify generator output shape before prediction loop
#     try:
#         test_batch, test_labels = test_gen[0]
#         print(f"\n[SHAPE TEST] First batch shape: {test_batch.shape}")
#         print(f"Sample data range: {test_batch.min()} to {test_batch.max()}")
#     except IndexError:
#         print("[ERROR] Test generator is empty!")
#         return

#     y_true, y_pred = [], []
#     print("[INFO] Testing Start")
#     processed_samples = 0
#     start_time = time.time()
#     # This loop can be slow if the dataset is large because we call model.predict per batch
#     for batch_idx, (X_batch, y_batch) in enumerate(test_gen):
#         print(f"\nProcessing batch of shape {X_batch.shape}")
        
#         if len(X_batch) == 0:
#             print("[WARNING] Empty batch detected!")
#             continue

#                 # === Progress Tracking ===
#         processed_samples += len(X_batch)
#         elapsed = time.time() - start_time
#         samples_per_sec = processed_samples / elapsed if elapsed > 0 else 0
        
#         print(f"\nBatch {batch_idx+1}/{total_batches} | "
#               f"Samples: {processed_samples}/{len(test_gen.filepaths)} | "
#               f"Speed: {samples_per_sec:.1f} samples/sec | "
#               f"Elapsed: {elapsed:.1f}s")
        
#         print(f"First sample min/max: {X_batch[0].min():.2f}/{X_batch[0].max():.2f}")
#         start_time = time.time()
#         preds = model.predict(X_batch, verbose=1)
#         print(f"Prediction time: {time.time()-start_time:.2f}s")
#         y_pred.extend(preds.argmax(axis=-1))
#         y_true.extend(y_batch)

#     # Save results
#     class_names = ['No Leak', 'Leak']
#     save_confusion_matrix(y_true, y_pred, class_names, output_dir)
#     save_classification_report(y_true, y_pred, output_dir)

#     return y_true, y_pred

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
        binary_all_leak=True  # merges classes [1..7] => label=1, class 0 => 0
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


# def evaluate_per_leak_class(model_path, test_dir, batch_size, output_dir):
#     """
#     Perform 10-fold testing for each leak class vs. no leak (0..k).
#     We gather all data in memory for 0 vs k, then do KFold(n_splits=10).
#     For each fold, we do a simple test pass and save confusion matrix/report.
#     """
#     print("[INFO] Evaluate each leak class vs. no-leak (0), with 10-fold testing...")

#     # Build a single model & load weights once
#     base_model = cnn_3d_model(input_shape=(15, 240, 320, 1), num_classes=2)
#     base_model.load_weights(model_path)

#     # For each leak class in [1..7], gather data, do KFold(10)
#     for leak_class in range(1, 8):
#         print(f"\n--- 0 vs {leak_class} (10-fold) ---")

#         # 1) Collect X,y in memory
#         temp_gen = DataGenerator(
#             data_dir=test_dir,
#             batch_size=batch_size,
#             shuffle=False,
#             binary_all_leak=False,
#             binary_pair=(0, leak_class)
#         )

#         # Check if we actually have any files for this binary pair (0 vs leak_class)
#         print(f"[INFO] Loaded {len(temp_gen.filepaths)} samples for 0 vs {leak_class}")

#         # If no data, skip
#         if len(temp_gen.filepaths) == 0:
#             print(f"Skipping 0 vs {leak_class} - no data found.")
#             continue

#         X_list, y_list = [], []
#         for X_batch, y_batch in temp_gen:
#             if len(X_batch) == 0:
#                 print("[WARNING] Empty batch detected during data collection. Skipping batch.")
#                 continue
#             X_list.append(X_batch)
#             y_list.append(y_batch)

#         if len(X_list) == 0:
#             print(f"No samples for 0 vs {leak_class}. Skipping.")
#             continue

#         # Concatenate all data collected
#         X_all = np.concatenate(X_list, axis=0)
#         y_all = np.concatenate(y_list, axis=0)
        
#         print(f"[INFO] 0 vs {leak_class} => Collected {len(X_all)} samples in memory.")

#         if len(X_all) < 10:
#             print(f"[WARNING] Not enough data for KFold (0 vs {leak_class}). Skipping.")
#             continue

#         # 2) KFold(10) -> We do 10 subsets
#         kf = KFold(n_splits=10, shuffle=True, random_state=42)
#         fold_index = 0

#         # Create a parent dir for "class_k" results
#         class_output_dir = os.path.join(output_dir, f'class_{leak_class}')
#         os.makedirs(class_output_dir, exist_ok=True)

#         for train_idx, test_idx in kf.split(X_all):
#             fold_index += 1
#             X_fold = X_all[test_idx]
#             y_fold = y_all[test_idx]

#             print(f"\n[Fold {fold_index}/10] Processing {len(X_fold)} samples")

#             if len(X_fold) == 0:
#                 print(f"[WARNING] Fold {fold_index} for 0 vs {leak_class} has 0 samples.")
#                 continue
            
#             # ======== CRITICAL FIX: Add batch_size to predict ========
#             try:
#                 # Process in smaller batches to avoid GPU OOM
#                 preds = base_model.predict(X_fold, batch_size=32, verbose=1)
#             except tf.errors.ResourceExhaustedError:
#                 print(f"[ERROR] GPU OOM in fold {fold_index}. Reducing batch size to 16...")
#                 preds = base_model.predict(X_fold, batch_size=16, verbose=1)
            
#             y_pred_fold = preds.argmax(axis=-1)

#             # 4) Save confusion matrix & classification report for this fold
#             fold_dir = os.path.join(class_output_dir, f'fold_{fold_index}')
#             class_names = ['No Leak', f'Leak {leak_class}']
#             save_confusion_matrix(y_fold, y_pred_fold, class_names, fold_dir)
#             save_classification_report(y_fold, y_pred_fold, fold_dir)

def evaluate_per_leak_class(model_path, test_dir, batch_size, output_dir):
    """
    Perform 10-fold testing for each leak class vs. no leak (0) and generate final table.
    """
    print("[INFO] Evaluate each leak class vs. no-leak (0), with 10-fold testing...")
    
    # Load model once
    model = cnn_3d_model(input_shape=(15, 240, 320, 1), num_classes=2)
    model.load_weights(model_path)
    
    # Dictionary to store results for final table
    final_results = {i: [] for i in range(1, 8)}  # Keys: 1-7 (leak classes)
    
    for leak_class in range(1, 8):
        print(f"\n--- 0 vs {leak_class} (10-fold) ---")
        
        # 1) Get all test files for classes 0 and leak_class
        test_files = []
        test_labels = []
        for class_label in [0, leak_class]:
            class_dir = os.path.join(test_dir, str(class_label))
            if not os.path.exists(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.endswith('.npz'):
                    test_files.append(os.path.join(class_dir, fname))
                    test_labels.append(0 if class_label == 0 else 1)
        
        if len(test_files) == 0:
            print(f"Skipping 0 vs {leak_class} - no data.")
            final_results[leak_class] = [0.0] * 10  # Handle missing data
            continue
            
        # 2) Convert to numpy arrays for KFold splitting
        test_files = np.array(test_files)
        test_labels = np.array(test_labels)
        
        # 3) 10-fold cross-validation
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        fold_accuracies = []
        
        for fold_i, (_, test_idx) in enumerate(kf.split(test_files)):
            # Get fold-specific test data
            fold_files = test_files[test_idx]
            fold_labels = test_labels[test_idx]
            
            # Create DataGenerator for this fold with explicit file list
            fold_gen = DataGenerator(
                data_dir=test_dir,
                batch_size=batch_size,
                shuffle=False,
                binary_all_leak=False,
                binary_pair=(0, leak_class),
                explicit_files=fold_files.tolist()
            )
            
            # Batch-safe evaluation
            y_true, y_pred = [], []
            for X_batch, y_batch in fold_gen:
                if len(X_batch) == 0:  # Skip empty batches
                    break
                preds = model.predict(X_batch, verbose=0)
                y_pred.extend(preds.argmax(axis=-1))
                y_true.extend(y_batch)
                
            # Calculate fold accuracy
            if len(y_true) > 0:  # Handle edge case of empty fold
                acc = accuracy_score(y_true, y_pred)
            else:
                acc = 0.0
                print(f"[WARNING] Empty fold {fold_i+1} for 0 vs {leak_class}")
            
            fold_accuracies.append(acc)
            print(f"Fold {fold_i+1}/10 | Accuracy: {acc:.4f}")
            
            # Save fold results
            fold_dir = os.path.join(output_dir, f'class_{leak_class}', f'fold_{fold_i+1}')
            os.makedirs(fold_dir, exist_ok=True)
            save_confusion_matrix(y_true, y_pred, ['No Leak', f'Leak {leak_class}'], fold_dir)
            save_classification_report(y_true, y_pred, fold_dir)
        
        # Store results for final table
        final_results[leak_class] = fold_accuracies
        
        # Print class summary
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

def test_binary(test_dir, model_path, batch_size, output_base_dir="./result"):
    """
    Master function to test the binary model in 2 ways:
     1) all leak vs. no leak
     2) 10-fold each leak class vs. no leak
    Called from main.py
    """
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, timestamp)

    # print("_______ Testing all leaks vs no-leak _______")
    # evaluate_all_leak_vs_no_leak(model_path, test_dir, batch_size, output_dir)

    print("\n_______ 10-Fold testing for each leak class vs no-leak _______")
    final_results  = evaluate_per_leak_class(model_path, test_dir, batch_size, output_dir)

    # Generate final table
    print("\n_______ Generating final table _______")
    generate_final_table(final_results, output_dir)

