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
from src.augmentation.video_augment import VideoAugmenter


class DataGenerator(Sequence):
    def __init__(self, data_dir, batch_size=16, shuffle=True, binary_all_leak=True, binary_pair=None, explicit_files=None, balance_classes=True, training=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.binary_all_leak = binary_all_leak
        self.binary_pair = binary_pair
        self.balance_classes = balance_classes
        self.filepaths = []
        self.augmenter = VideoAugmenter(p=0.5)
        self.training = training  # Add this flag

        # Handle explicit files differently since they're already tuples of (path, label)
        if explicit_files is not None:
            self.filepaths = explicit_files  # These are already (path, label) tuples
        else:
            # Collect all .npz files in subfolders 0..7
            for class_label_str in sorted(os.listdir(data_dir)):
                subdir = os.path.join(data_dir, class_label_str)
                if not os.path.isdir(subdir):
                    continue
                try:
                    class_label = int(class_label_str)
                except:
                    continue

                if self.binary_all_leak:
                    if class_label < 0 or class_label > 7:
                        continue
                elif self.binary_pair is not None:
                    if class_label not in self.binary_pair:
                        continue

                for fname in os.listdir(subdir):
                    if fname.endswith('.npz'):
                        self.filepaths.append((os.path.join(subdir, fname), class_label))

        # Separate no-leak and leak files
        self.no_leak_files = [(p, l) for p, l in self.filepaths if l == 0]
        self.leak_files = [(p, l) for p, l in self.filepaths if l != 0]
        
        if self.balance_classes and training:
            # If no_leak is minority class, duplicate it with augmentation
            if len(self.no_leak_files) < len(self.leak_files):
                # Calculate how many times we need to duplicate
                target_size = len(self.leak_files)
                while len(self.no_leak_files) < target_size:
                    # Add copies from original no_leak files
                    remaining_needed = target_size - len(self.no_leak_files)
                    # Take the minimum between remaining needed and original size
                    num_to_add = min(remaining_needed, len(self.no_leak_files))
                    self.no_leak_files.extend(self.no_leak_files[:num_to_add])
            
            # Combine balanced files
            self.filepaths = self.no_leak_files + self.leak_files
            
        print(f"[INFO] Class distribution after balancing:")
        print(f"  No leak (0): {len(self.no_leak_files)} samples")
        print(f"  Leak (1-7): {len(self.leak_files)} samples")

        if len(self.filepaths) == 0:
            raise RuntimeError(f"No valid .npz files found in {data_dir}")

        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.filepaths)

    def __len__(self):
        if len(self.filepaths) == 0:
            return 0
        return math.ceil(len(self.filepaths) / self.batch_size)

    def __getitem__(self, idx):
        batch_slice = self.filepaths[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_list, y_list = [], []
        
        for path, orig_label in batch_slice:
            try:
                with np.load(path) as data:
                    frames = data['segment'].astype(np.float32)  # Using 'segment' as key
                    
                    # Ensure correct shape
                    if frames.shape != (15, 240, 320, 1):
                        frames = frames.reshape(15, 240, 320, 1)
                    
                    # Apply augmentation for no-leak class during training
                    if self.training and orig_label == 0:
                        frames = self.augmenter.apply_augmentation(frames)
                        
                    # Verify shape after augmentation
                    assert frames.shape == (15, 240, 320, 1), f"Invalid shape after processing: {frames.shape}"
                    
                    # Convert label to binary if needed
                    if self.binary_all_leak:
                        label = 1 if orig_label > 0 else 0
                    elif self.binary_pair is not None:
                        label = 1 if orig_label == self.binary_pair[1] else 0
                    else:
                        label = orig_label
                        
                    X_list.append(frames)
                    y_list.append(label)
            except Exception as e:
                print(f"[ERROR] Failed to load {path}: {str(e)}")
                continue
                
        if not X_list:
            return np.empty((0, 15, 240, 320, 1)), np.array([], dtype=np.int32)
            
        X_batch = np.array(X_list)
        y_batch = np.array(y_list)
        
        return X_batch, y_batch