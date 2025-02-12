import numpy as np
import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import (EarlyStopping,
                                        ReduceLROnPlateau,
                                        ModelCheckpoint)
import math


class DataGenerator(Sequence):
    def __init__(self, data_dir, batch_size=16, shuffle=True, binary_all_leak=True, binary_pair=None, explicit_files=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.binary_all_leak = binary_all_leak
        self.binary_pair = binary_pair
        self.filepaths = []

        # Validate directory structure
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory {data_dir} does not exist!")
        
        if explicit_files is not None:
            # Directly use provided files (for fold-specific loading)
            self.filepaths = []
            for path in explicit_files:
                class_label = int(os.path.basename(os.path.dirname(path)))
                self.filepaths.append((path, class_label))
        else:
        # Collect all .npz in subfolders 0..7
            for class_label_str in sorted(os.listdir(data_dir)):
                subdir = os.path.join(data_dir, class_label_str)
                if not os.path.isdir(subdir):
                    print(f"[WARNING] Skipping non-directory: {subdir}")
                    continue
                try:
                    class_label = int(class_label_str)
                except:
                    print(f"[WARNING] Invalid class folder name: {class_label_str}")
                    continue

                if self.binary_all_leak:
                    if class_label < 0 or class_label > 7:  # Only allow classes 0 (no-leak) and 1-7 (leak)
                        continue

                elif self.binary_pair is not None:
                    if class_label not in self.binary_pair:  # e.g. (0,3) => only keep if label=0 or label=3
                        continue

                for fname in os.listdir(subdir):
                    if fname.endswith('.npz'):
                        self.filepaths.append((os.path.join(subdir, fname), class_label))

        # After collecting filepaths, check how many valid files were loaded
        if len(self.filepaths) == 0:
            raise RuntimeError(f"No valid .npz files found in {data_dir} "
                            f"with binary_all_leak={binary_all_leak}, "
                            f"binary_pair={binary_pair}")

        print(f"[INFO] Loaded {len(self.filepaths)} samples "
              f"(binary_all_leak={binary_all_leak})")

        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.filepaths)

    def __len__(self):
        if len(self.filepaths) == 0:
            return 0
        return math.ceil(len(self.filepaths) / self.batch_size)

    def __getitem__(self, idx):
        batch_slice = self.filepaths[idx * self.batch_size: (idx + 1) * self.batch_size]

        if not batch_slice:
            print(f"[DEBUG] Empty batch detected at batch {idx}.")
            return np.empty((0, 15, 240, 320, 1)), np.array([], dtype=np.int32)

        X_list, y_list = [], []

        for (path, orig_label) in batch_slice:
            try:
                with np.load(path) as data:
                    frames = data['segment'].astype(np.float32)

                    # Validate shape (15, 240, 320, 1)
                    if frames.shape != (15, 240, 320, 1):
                        print(f"[ERROR] Invalid shape in {path}: {frames.shape}. Expected (15, 240, 320, 1)")
                        continue

                    # Ensure channel dimension exists (some .npz might save as (15,240,320))
                    if frames.ndim == 3:
                        frames = np.expand_dims(frames, axis=-1)
                        print(f"[WARNING] Added channel dim to {path}")
            except Exception as e:
                print(f"[ERROR] Corrupted file {path}: {str(e)}")
                continue

            # Map label
            if self.binary_all_leak:
                label = 0 if orig_label == 0 else 1
            else:
                if self.binary_pair is not None:
                    if orig_label == self.binary_pair[0]:
                        label = 0
                    else:
                        label = 1
                else:
                    label = orig_label  # fallback

            X_list.append(frames)
            y_list.append(label)

        if not X_list:  # If no valid data in batch, skip it
            print(f"[DEBUG] No valid data found in batch {idx}. Skipping.")
            return np.empty((0, 15, 240, 320, 1)), np.array([], dtype=np.int32)

        X_batch = np.array(X_list, dtype=np.float32)  # (B, T, H, W, C)
        y_batch = np.array(y_list, dtype=np.int32)

        # print(f"[DEBUG] Batch {idx} shape: {X_batch.shape}")  # Should be (batch_size, 15,240,320,1)
        return X_batch, y_batch