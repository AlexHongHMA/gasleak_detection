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
        self.training = training
        self.resize = resize
        self.target_height = target_height
        self.target_width = target_width
        
        # Initialize augmenter with the target dimensions
        self.augmenter = VideoAugmenter(p=0.5, target_height=target_height, target_width=target_width)

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

                # Skip folders that aren't in the expected class range
                if class_label < 0 or class_label > 7:
                    continue
                    
                # In binary_pair mode, only include specified pair classes
                if self.binary_pair is not None:
                    if class_label not in self.binary_pair:
                        continue

                for fname in os.listdir(subdir):
                    if fname.endswith('.npz'):
                        self.filepaths.append((os.path.join(subdir, fname), class_label))

        # Separate into no-leak and leak files
        self.no_leak_files = [(p, l) for p, l in self.filepaths if l == 0]
        self.leak_files = [(p, l) for p, l in self.filepaths if l != 0]
        
        if self.balance_classes and training:
            # If no_leak is minority class, duplicate it
            if len(self.no_leak_files) < len(self.leak_files) and len(self.no_leak_files) > 0:
                # Calculate how many times we need to duplicate
                target_size = len(self.leak_files)
                self.no_leak_files = self._balance_class(self.no_leak_files, target_size)
            
            # Combine balanced files
            self.filepaths = self.no_leak_files + self.leak_files
            
        print(f"[INFO] Binary class distribution after balancing:")
        print(f"  No leak (0): {len(self.no_leak_files)} samples")
        print(f"  Leak (1-7): {len(self.leak_files)} samples")

        if len(self.filepaths) == 0:
            raise RuntimeError(f"No valid .npz files found in {data_dir}")

        self.on_epoch_end()

    def _balance_class(self, file_list, target_size):
        """Helper method to balance a specific class to reach the target size"""
        balanced_list = file_list.copy()
        while len(balanced_list) < target_size:
            # Add copies from original files
            remaining_needed = target_size - len(balanced_list)
            # Take the minimum between remaining needed and original size
            num_to_add = min(remaining_needed, len(file_list))
            balanced_list.extend(file_list[:num_to_add])
        return balanced_list
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.filepaths)

    def __len__(self):
        if len(self.filepaths) == 0:
            return 0
        return math.ceil(len(self.filepaths) / self.batch_size)

    def __getitem__(self, idx):
        batch_slice = self.filepaths[idx * self.batch_size:(idx + 1) * self.batch_size]
        # Pre-allocate arrays for better memory efficiency
        X_batch = np.zeros((len(batch_slice), 15, self.target_height, self.target_width, 1), dtype=np.float32)
        y_batch = np.zeros(len(batch_slice), dtype=np.int32)
        
        valid_samples = 0
        for i, (path, orig_label) in enumerate(batch_slice):
            try:
                # Use memory mapping for faster file access
                data = np.load(path, mmap_mode='r')
                frames = data['segment'].astype(np.float32)
                
                # Ensure correct shape for original frames
                if frames.shape != (15, 240, 320, 1):
                    frames = frames.reshape(15, 240, 320, 1)
                
                # Resize if needed - batch process all frames at once when possible
                if self.resize and (self.target_height != 240 or self.target_width != 320):
                    if not self.training or orig_label != 0:  # Skip augmentation during inference
                        # Process all frames at once for better efficiency
                        for j in range(15):
                            X_batch[valid_samples, j, :, :, 0] = cv2.resize(
                                frames[j, :, :, 0], 
                                (self.target_width, self.target_height), 
                                interpolation=cv2.INTER_AREA
                            )
                    else:
                        # Only use augmentation during training for class 0
                        resized_frames = np.zeros((15, self.target_height, self.target_width, 1), dtype=np.float32)
                        for j in range(15):
                            resized_frames[j, :, :, 0] = cv2.resize(
                                frames[j, :, :, 0], 
                                (self.target_width, self.target_height), 
                                interpolation=cv2.INTER_AREA
                            )
                        frames = self.augmenter.apply_augmentation(resized_frames)
                        X_batch[valid_samples] = frames
                else:
                    X_batch[valid_samples] = frames
                
                # Convert label according to the selected mode
                if self.binary_all_leak:
                    label = 1 if orig_label > 0 else 0
                elif self.binary_pair is not None:
                    label = 1 if orig_label == self.binary_pair[1] else 0
                else:
                    label = orig_label
                    
                y_batch[valid_samples] = label
                valid_samples += 1
                
                # Explicitly close the .npz file to free resources
                data.close()
                
            except Exception as e:
                print(f"[ERROR] Failed to load {path}: {str(e)}")
                continue
                
        if valid_samples == 0:
            return np.empty((0, 15, self.target_height, self.target_width, 1)), np.array([], dtype=np.int32)
            
        # Return only valid samples
        return X_batch[:valid_samples], y_batch[:valid_samples]

class DataGeneratorThreeClass(Sequence):
    def __init__(self, data_dir, batch_size=16, shuffle=True, explicit_files=None, 
                 balance_classes=True, training=True, resize=True, 
                 target_height=120, target_width=160, distance_filter=None):
        """
        Data generator for three-class leak classification.
        
        Args:
            data_dir: Directory containing training/testing data
            batch_size: Batch size for training/inference
            shuffle: Whether to shuffle data between epochs
            explicit_files: Explicit list of (path, label) tuples to use instead of reading from data_dir
            balance_classes: Whether to balance classes during training
            training: Whether this generator is used for training or inference
            resize: Whether to resize frames to target dimensions
            target_height: Target height for resizing
            target_width: Target width for resizing
            distance_filter: Filter data by imaging distance: '46' for 4.6m, '69' for 6.9m, or None for all data
        """
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
        
        # Initialize augmenter with the target dimensions
        self.augmenter = VideoAugmenter(p=0.5, target_height=target_height, target_width=target_width)

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
                self.small_leak_files = self._balance_class(self.small_leak_files, target_size)
            
            # Balance medium leak class if needed
            if len(self.medium_leak_files) < target_size and len(self.medium_leak_files) > 0:
                self.medium_leak_files = self._balance_class(self.medium_leak_files, target_size)
            
            # Balance large leak class if needed
            if len(self.large_leak_files) < target_size and len(self.large_leak_files) > 0:
                self.large_leak_files = self._balance_class(self.large_leak_files, target_size)
            
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

    def _balance_class(self, file_list, target_size):
        """Helper method to balance a specific class to reach the target size"""
        balanced_list = file_list.copy()
        while len(balanced_list) < target_size:
            # Add copies from original files
            remaining_needed = target_size - len(balanced_list)
            # Take the minimum between remaining needed and original size
            num_to_add = min(remaining_needed, len(file_list))
            balanced_list.extend(file_list[:num_to_add])
        return balanced_list
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.filepaths)

    def __len__(self):
        if len(self.filepaths) == 0:
            return 0
        return math.ceil(len(self.filepaths) / self.batch_size)

    def __getitem__(self, idx):
        batch_slice = self.filepaths[idx * self.batch_size:(idx + 1) * self.batch_size]
        # Pre-allocate arrays for better memory efficiency
        X_batch = np.zeros((len(batch_slice), 15, self.target_height, self.target_width, 1), dtype=np.float32)
        y_batch = np.zeros(len(batch_slice), dtype=np.int32)
        
        valid_samples = 0
        for i, (path, label) in enumerate(batch_slice):
            try:
                # Use memory mapping for faster file access
                data = np.load(path, mmap_mode='r')
                frames = data['segment'].astype(np.float32)
                
                # Ensure correct shape for original frames
                if frames.shape != (15, 240, 320, 1):
                    frames = frames.reshape(15, 240, 320, 1)
                
                # Resize if needed - batch process all frames at once when possible
                if self.resize and (self.target_height != 240 or self.target_width != 320):
                    if not self.training:  # Skip augmentation during inference
                        # Process all frames at once for better efficiency
                        for j in range(15):
                            X_batch[valid_samples, j, :, :, 0] = cv2.resize(
                                frames[j, :, :, 0], 
                                (self.target_width, self.target_height), 
                                interpolation=cv2.INTER_AREA
                            )
                    else:
                        # Only use augmentation during training
                        resized_frames = np.zeros((15, self.target_height, self.target_width, 1), dtype=np.float32)
                        for j in range(15):
                            resized_frames[j, :, :, 0] = cv2.resize(
                                frames[j, :, :, 0], 
                                (self.target_width, self.target_height), 
                                interpolation=cv2.INTER_AREA
                            )
                        # Apply augmentation
                        frames = self.augmenter.apply_augmentation(resized_frames)
                        X_batch[valid_samples] = frames
                else:
                    X_batch[valid_samples] = frames
                
                # The label is already mapped to three-class format when creating filepaths
                y_batch[valid_samples] = label
                valid_samples += 1
                
                # Explicitly close the .npz file to free resources
                data.close()
                
            except Exception as e:
                print(f"[ERROR] Failed to load {path}: {str(e)}")
                continue
                
        if valid_samples == 0:
            return np.empty((0, 15, self.target_height, self.target_width, 1)), np.array([], dtype=np.int32)
            
        # Return only valid samples
        return X_batch[:valid_samples], y_batch[:valid_samples]


'''
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.binary_all_leak = binary_all_leak
        self.binary_pair = binary_pair
        self.balance_classes = balance_classes
        self.filepaths = []
        self.training = training
        self.resize = resize
        self.target_height = target_height
        self.target_width = target_width
        
        # Initialize augmenter with the target dimensions
        self.augmenter = VideoAugmenter(p=0.5, target_height=target_height, target_width=target_width)

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

                # Skip folders that aren't in the expected class range
                if class_label < 0 or class_label > 7:
                    continue
                    
                # In binary_pair mode, only include specified pair classes
                if self.binary_pair is not None:
                    if class_label not in self.binary_pair:
                        continue

                for fname in os.listdir(subdir):
                    if fname.endswith('.npz'):
                        self.filepaths.append((os.path.join(subdir, fname), class_label))

        # Separate into no-leak and leak files
        self.no_leak_files = [(p, l) for p, l in self.filepaths if l == 0]
        self.leak_files = [(p, l) for p, l in self.filepaths if l != 0]
        
        if self.balance_classes and training:
            # If no_leak is minority class, duplicate it
            if len(self.no_leak_files) < len(self.leak_files) and len(self.no_leak_files) > 0:
                # Calculate how many times we need to duplicate
                target_size = len(self.leak_files)
                self.no_leak_files = self._balance_class(self.no_leak_files, target_size)
            
            # Combine balanced files
            self.filepaths = self.no_leak_files + self.leak_files
            
        print(f"[INFO] Binary class distribution after balancing:")
        print(f"  No leak (0): {len(self.no_leak_files)} samples")
        print(f"  Leak (1-7): {len(self.leak_files)} samples")

        if len(self.filepaths) == 0:
            raise RuntimeError(f"No valid .npz files found in {data_dir}")

        self.on_epoch_end()

    def _balance_class(self, file_list, target_size):
        """Helper method to balance a specific class to reach the target size"""
        balanced_list = file_list.copy()
        while len(balanced_list) < target_size:
            # Add copies from original files
            remaining_needed = target_size - len(balanced_list)
            # Take the minimum between remaining needed and original size
            num_to_add = min(remaining_needed, len(file_list))
            balanced_list.extend(file_list[:num_to_add])
        return balanced_list
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.filepaths)

    def __len__(self):
        if len(self.filepaths) == 0:
            return 0
        return math.ceil(len(self.filepaths) / self.batch_size)

    def __getitem__(self, idx):
        batch_slice = self.filepaths[idx * self.batch_size:(idx + 1) * self.batch_size]
        # Pre-allocate arrays for better memory efficiency
        X_batch = np.zeros((len(batch_slice), 15, self.target_height, self.target_width, 1), dtype=np.float32)
        y_batch = np.zeros(len(batch_slice), dtype=np.int32)
        
        valid_samples = 0
        for i, (path, orig_label) in enumerate(batch_slice):
            try:
                # Use memory mapping for faster file access
                data = np.load(path, mmap_mode='r')
                frames = data['segment'].astype(np.float32)
                
                # Ensure correct shape for original frames
                if frames.shape != (15, 240, 320, 1):
                    frames = frames.reshape(15, 240, 320, 1)
                
                # Resize if needed - batch process all frames at once when possible
                if self.resize and (self.target_height != 240 or self.target_width != 320):
                    if not self.training or orig_label != 0:  # Skip augmentation during inference
                        # Process all frames at once for better efficiency
                        for j in range(15):
                            X_batch[valid_samples, j, :, :, 0] = cv2.resize(
                                frames[j, :, :, 0], 
                                (self.target_width, self.target_height), 
                                interpolation=cv2.INTER_AREA
                            )
                    else:
                        # Only use augmentation during training for class 0
                        resized_frames = np.zeros((15, self.target_height, self.target_width, 1), dtype=np.float32)
                        for j in range(15):
                            resized_frames[j, :, :, 0] = cv2.resize(
                                frames[j, :, :, 0], 
                                (self.target_width, self.target_height), 
                                interpolation=cv2.INTER_AREA
                            )
                        frames = self.augmenter.apply_augmentation(resized_frames)
                        X_batch[valid_samples] = frames
                else:
                    X_batch[valid_samples] = frames
                
                # Convert label according to the selected mode
                if self.binary_all_leak:
                    label = 1 if orig_label > 0 else 0
                elif self.binary_pair is not None:
                    label = 1 if orig_label == self.binary_pair[1] else 0
                else:
                    label = orig_label
                    
                y_batch[valid_samples] = label
                valid_samples += 1
                
                # Explicitly close the .npz file to free resources
                data.close()
                
            except Exception as e:
                print(f"[ERROR] Failed to load {path}: {str(e)}")
                continue
                
        if valid_samples == 0:
            return np.empty((0, 15, self.target_height, self.target_width, 1)), np.array([], dtype=np.int32)
            
        # Return only valid samples
        return X_batch[:valid_samples], y_batch[:valid_samples]
'''