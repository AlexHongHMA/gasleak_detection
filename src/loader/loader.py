import numpy as np
import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
import cv2

import math
from src.augmentation.video_augment import VideoAugmenter


class DataGenerator(Sequence):
    def __init__(self, data_dir, batch_size=16, shuffle=True, binary_all_leak=True, 
                 binary_pair=None, explicit_files=None, balance_classes=True, training=True, 
                 resize=True, target_height=120, target_width=160):
        super().__init__()
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

class DataGeneratorThreeClass(Sequence):
    def __init__(self, data_dir, batch_size=16, shuffle=True, explicit_files=None, 
                 balance_classes=True, training=True, resize=True, 
                 target_height=120, target_width=160, distance_filter=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.balance_classes = balance_classes
        self.filepaths = []
        self.training = training
        self.resize = resize
        self.target_height = target_height
        self.target_width = target_width
        self.distance_filter = distance_filter
        self.augmenter = VideoAugmenter(p=0.5)

        # Handle explicit files differently since they're already tuples of (path, label)
        if explicit_files is not None:
            self.filepaths = explicit_files  # These are already (path, label) tuples
            # Apply distance filtering to explicit files if needed
            if self.distance_filter is not None:
                self.filepaths = [(p, l) for p, l in self.filepaths 
                                  if self._check_distance_in_filename(p)]
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

                # Skip folders that aren't in the expected class range
                if class_label < 0 or class_label > 7:
                    continue

                for fname in os.listdir(subdir):
                    if fname.endswith('.npz'):
                        # Apply distance filtering
                        if self.distance_filter is not None and not self._check_distance_in_filename(os.path.join(subdir, fname)):
                            continue
                        
                        # Map original classes to three-class categories
                        mapped_label = self._map_to_three_class(class_label)
                        self.filepaths.append((os.path.join(subdir, fname), mapped_label))

        # Separate files by leak size category
        self.small_leak_files = [(p, l) for p, l in self.filepaths if l == 0]  # Small leaks
        self.medium_leak_files = [(p, l) for p, l in self.filepaths if l == 1]  # Medium leaks
        self.large_leak_files = [(p, l) for p, l in self.filepaths if l == 2]  # Large leaks
        
        if self.balance_classes and training:
            # Determine the target size (maximum class count for balancing)
            target_size = max(len(self.small_leak_files), 
                             len(self.medium_leak_files), 
                             len(self.large_leak_files))
            
            # Balance small leak class if needed
            if len(self.small_leak_files) < target_size and len(self.small_leak_files) > 0:
                # Calculate how many times we need to duplicate
                while len(self.small_leak_files) < target_size:
                    # Add copies from original small_leak files
                    remaining_needed = target_size - len(self.small_leak_files)
                    # Take the minimum between remaining needed and original size
                    num_to_add = min(remaining_needed, len(self.small_leak_files))
                    self.small_leak_files.extend(self.small_leak_files[:num_to_add])
            
            # Balance medium leak class if needed
            if len(self.medium_leak_files) < target_size and len(self.medium_leak_files) > 0:
                while len(self.medium_leak_files) < target_size:
                    remaining_needed = target_size - len(self.medium_leak_files)
                    num_to_add = min(remaining_needed, len(self.medium_leak_files))
                    self.medium_leak_files.extend(self.medium_leak_files[:num_to_add])
            
            # Balance large leak class if needed
            if len(self.large_leak_files) < target_size and len(self.large_leak_files) > 0:
                while len(self.large_leak_files) < target_size:
                    remaining_needed = target_size - len(self.large_leak_files)
                    num_to_add = min(remaining_needed, len(self.large_leak_files))
                    self.large_leak_files.extend(self.large_leak_files[:num_to_add])
            
            # Combine the balanced classes
            self.filepaths = self.small_leak_files + self.medium_leak_files + self.large_leak_files
            
        # Print class distribution
        distance_info = f" at distance {self.distance_filter}m" if self.distance_filter else ""
        print(f"[INFO] Three-class distribution{distance_info} after balancing:")
        print(f"  Small leak (0-2): {len(self.small_leak_files)} samples")
        print(f"  Medium leak (3-5): {len(self.medium_leak_files)} samples")
        print(f"  Large leak (6-7): {len(self.large_leak_files)} samples")

        if len(self.filepaths) == 0:
            raise RuntimeError(f"No valid .npz files found in {data_dir}{distance_info}")

        self.on_epoch_end()

    def _check_distance_in_filename(self, filepath):
        """Check if a file matches the distance filter."""
        if self.distance_filter is None:
            return True
            
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        
        # Typical filename format: "segment0_1468_69_380"
        # The distance code is usually in the third part
        if len(parts) >= 3:
            try:
                # Extract the distance code (e.g., "69" for 6.9m)
                if parts[2] == self.distance_filter:
                    return True
            except (IndexError, ValueError):
                pass
        return False

    def _map_to_three_class(self, original_label):
        """
        Map original class labels (0-7) to three-class categories:
        0: Small leak (classes 0-2)
        1: Medium leak (classes 3-5)
        2: Large leak (classes 6-7)
        """
        if 0 <= original_label <= 2:
            return 0  # Small leak
        elif 3 <= original_label <= 5:
            return 1  # Medium leak
        elif 6 <= original_label <= 7:
            return 2  # Large leak
        else:
            raise ValueError(f"Invalid original class label: {original_label}")
        
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
        
        for path, label in batch_slice:
            try:
                with np.load(path) as data:
                    frames = data['segment'].astype(np.float32)  # Using 'segment' as key
                    
                    # Ensure correct shape
                    if frames.shape != (15, 240, 320, 1):
                        frames = frames.reshape(15, 240, 320, 1)
                    
                    # Resize if needed
                    if self.resize and (self.target_height != 240 or self.target_width != 320):
                        resized_frames = np.zeros((15, self.target_height, self.target_width, 1), dtype=np.float32)
                        for j in range(15):
                            resized_frames[j, :, :, 0] = cv2.resize(
                                frames[j, :, :, 0], 
                                (self.target_width, self.target_height), 
                                interpolation=cv2.INTER_AREA
                            )
                        frames = resized_frames
                    
                    # Apply augmentation during training
                    if self.training:
                        frames = self.augmenter.apply_augmentation(frames)
                        
                    # Verify shape after processing
                    expected_shape = (15, self.target_height, self.target_width, 1) if self.resize else (15, 240, 320, 1)
                    assert frames.shape == expected_shape, f"Invalid shape after processing: {frames.shape}"
                    
                    X_list.append(frames)
                    y_list.append(label)
            except Exception as e:
                print(f"[ERROR] Failed to load {path}: {str(e)}")
                continue
                
        if not X_list:
            expected_height = self.target_height if self.resize else 240
            expected_width = self.target_width if self.resize else 320
            return np.empty((0, 15, expected_height, expected_width, 1)), np.array([], dtype=np.int32)
            
        X_batch = np.array(X_list)
        y_batch = np.array(y_list)
        
        return X_batch, y_batch

class DataGeneratorEightClass(Sequence):
    def __init__(self, data_dir, batch_size=16, shuffle=True, explicit_files=None, 
                 balance_classes=True, training=True, resize=True, 
                 target_height=120, target_width=160, distance_filter=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.balance_classes = balance_classes
        self.filepaths = []
        self.training = training
        self.resize = resize
        self.target_height = target_height
        self.target_width = target_width
        self.distance_filter = distance_filter
        self.augmenter = VideoAugmenter(p=0.5)

        # Handle explicit files differently since they're already tuples of (path, label)
        if explicit_files is not None:
            self.filepaths = explicit_files  # These are already (path, label) tuples
            # Apply distance filtering to explicit files if needed
            if self.distance_filter is not None:
                self.filepaths = [(p, l) for p, l in self.filepaths 
                                  if self._check_distance_in_filename(p)]
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

                # Skip folders that aren't in the expected class range
                if class_label < 0 or class_label > 7:
                    continue

                for fname in os.listdir(subdir):
                    if fname.endswith('.npz'):
                        # Apply distance filtering
                        if self.distance_filter is not None and not self._check_distance_in_filename(os.path.join(subdir, fname)):
                            continue
                        
                        # Use the original labels (0-7) directly for 8-class classification
                        self.filepaths.append((os.path.join(subdir, fname), class_label))

        # Separate files by class (0-7)
        self.class_files = [[] for _ in range(8)]
        for p, l in self.filepaths:
            if 0 <= l <= 7:
                self.class_files[l].append((p, l))
        
        if self.balance_classes and training:
            # Determine the target size (maximum class count for balancing)
            target_size = max([len(class_list) for class_list in self.class_files])
            
            # Balance each class to match the target size
            for class_idx in range(8):
                class_list = self.class_files[class_idx]
                if len(class_list) < target_size and len(class_list) > 0:
                    # Calculate how many times we need to duplicate
                    while len(class_list) < target_size:
                        # Add copies from original class_list
                        remaining_needed = target_size - len(class_list)
                        # Take the minimum between remaining needed and original size
                        num_to_add = min(remaining_needed, len(class_list))
                        class_list.extend(class_list[:num_to_add])
                    self.class_files[class_idx] = class_list
            
            # Combine the balanced classes
            self.filepaths = []
            for class_list in self.class_files:
                self.filepaths.extend(class_list)
            
        # Print class distribution
        distance_info = f" at distance {self.distance_filter}m" if self.distance_filter else ""
        print(f"[INFO] Eight-class distribution{distance_info} after balancing:")
        for i in range(8):
            print(f"  Class {i}: {len(self.class_files[i])} samples")

        if len(self.filepaths) == 0:
            raise RuntimeError(f"No valid .npz files found in {data_dir}{distance_info}")

        self.on_epoch_end()

    def _check_distance_in_filename(self, filepath):
        """Check if a file matches the distance filter."""
        if self.distance_filter is None:
            return True
            
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        
        # Typical filename format: "segment0_1468_69_380"
        # The distance code is usually in the third part
        if len(parts) >= 3:
            try:
                # Extract the distance code (e.g., "69" for 6.9m)
                if parts[2] == self.distance_filter:
                    return True
            except (IndexError, ValueError):
                pass
        return False
        
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
        
        for path, label in batch_slice:
            try:
                with np.load(path) as data:
                    frames = data['segment'].astype(np.float32)  # Using 'segment' as key
                    
                    # Ensure correct shape
                    if frames.shape != (15, 240, 320, 1):
                        frames = frames.reshape(15, 240, 320, 1)
                    
                    # Resize if needed
                    if self.resize and (self.target_height != 240 or self.target_width != 320):
                        resized_frames = np.zeros((15, self.target_height, self.target_width, 1), dtype=np.float32)
                        for j in range(15):
                            resized_frames[j, :, :, 0] = cv2.resize(
                                frames[j, :, :, 0], 
                                (self.target_width, self.target_height), 
                                interpolation=cv2.INTER_AREA
                            )
                        frames = resized_frames
                    
                    # Apply augmentation during training
                    if self.training:
                        frames = self.augmenter.apply_augmentation(frames)
                        
                    # Verify shape after processing
                    expected_shape = (15, self.target_height, self.target_width, 1) if self.resize else (15, 240, 320, 1)
                    assert frames.shape == expected_shape, f"Invalid shape after processing: {frames.shape}"
                    
                    X_list.append(frames)
                    y_list.append(label)
            except Exception as e:
                print(f"[ERROR] Failed to load {path}: {str(e)}")
                continue
                
        if not X_list:
            expected_height = self.target_height if self.resize else 240
            expected_width = self.target_width if self.resize else 320
            return np.empty((0, 15, expected_height, expected_width, 1)), np.array([], dtype=np.int32)
            
        X_batch = np.array(X_list)
        y_batch = np.array(y_list)
        
        return X_batch, y_batch
