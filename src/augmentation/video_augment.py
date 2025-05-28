import numpy as np
import cv2

class VideoAugmenter:
    def __init__(self, p=0.5, target_height=240, target_width=320):
        self.p = p
    
    def random_flip_horizontal(self, frames):
        """Apply random horizontal flip"""
        if np.random.random() > self.p:
            return frames
        return np.flip(frames, axis=2).reshape(15, 240, 320, 1)
    
    def random_rotation(self, frames, max_angle=10):
        """Apply random rotation"""
        if np.random.random() > self.p:
            return frames
        angle = np.random.uniform(-max_angle, max_angle)
        center = (frames.shape[2] // 2, frames.shape[1] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        rotated_frames = np.zeros_like(frames)
        for i in range(frames.shape[0]):
            rotated_frames[i, :, :, 0] = cv2.warpAffine(
                frames[i, :, :, 0], 
                matrix, 
                (frames.shape[2], frames.shape[1])
            )
        return rotated_frames.reshape(15, 240, 320, 1)
    
    def apply_augmentation(self, frames):
        """Apply a random combination of augmentations"""
        # Ensure input shape is correct
        if frames.shape != (15, 240, 320, 1):
            frames = frames.reshape(15, 240, 320, 1)
            
        augmentations = [
            self.random_flip_horizontal,
            self.random_rotation
        ]
        
        # Apply 1-2 random augmentations
        num_augs = np.random.randint(1, 3)
        selected_augs = np.random.choice(augmentations, num_augs, replace=False)
        
        augmented_frames = frames.copy()
        for aug in selected_augs:
            augmented_frames = aug(augmented_frames)
            
        # Final shape check
        if augmented_frames.shape != (15, 240, 320, 1):
            augmented_frames = augmented_frames.reshape(15, 240, 320, 1)
            
        return augmented_frames 

# class VideoAugmenter:
#     def __init__(self, p=0.5):
#         self.p = p
    
#     def random_flip_horizontal(self, frames):
#         """Apply random horizontal flip"""
#         if np.random.random() > self.p:
#             return frames
#         return np.flip(frames, axis=2).reshape(15, 240, 320, 1)
    
#     def random_rotation(self, frames, max_angle=10):
#         """Apply random rotation"""
#         if np.random.random() > self.p:
#             return frames
#         angle = np.random.uniform(-max_angle, max_angle)
#         center = (frames.shape[2] // 2, frames.shape[1] // 2)
#         matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
#         rotated_frames = np.zeros_like(frames)
#         for i in range(frames.shape[0]):
#             rotated_frames[i, :, :, 0] = cv2.warpAffine(
#                 frames[i, :, :, 0], 
#                 matrix, 
#                 (frames.shape[2], frames.shape[1])
#             )
#         return rotated_frames.reshape(15, 240, 320, 1)
    
#     def apply_augmentation(self, frames):
#         """Apply a random combination of augmentations"""
#         # Ensure input shape is correct
#         if frames.shape != (15, 240, 320, 1):
#             frames = frames.reshape(15, 240, 320, 1)
            
#         augmentations = [
#             self.random_flip_horizontal,
#             self.random_rotation
#         ]
        
#         # Apply 1-2 random augmentations
#         num_augs = np.random.randint(1, 3)
#         selected_augs = np.random.choice(augmentations, num_augs, replace=False)
        
#         augmented_frames = frames.copy()
#         for aug in selected_augs:
#             augmented_frames = aug(augmented_frames)
            
#         # Final shape check
#         if augmented_frames.shape != (15, 240, 320, 1):
#             augmented_frames = augmented_frames.reshape(15, 240, 320, 1)
            
#         return augmented_frames 


# class VideoAugmenter:
#     def __init__(self, p=0.5, target_height=240, target_width=320):
    #     self.p = p
    #     self.target_height = target_height
    #     self.target_width = target_width
    
    # def random_flip_horizontal(self, frames):
    #     """Apply random horizontal flip"""
    #     if np.random.random() > self.p:
    #         return frames
    #     return np.flip(frames, axis=2).reshape(15, self.target_height, self.target_width, 1)
    
    # def random_rotation(self, frames, max_angle=10):
    #     """Apply random rotation"""
    #     if np.random.random() > self.p:
    #         return frames
    #     angle = np.random.uniform(-max_angle, max_angle)
    #     center = (frames.shape[2] // 2, frames.shape[1] // 2)
    #     matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
    #     rotated_frames = np.zeros_like(frames)
    #     for i in range(frames.shape[0]):
    #         rotated_frames[i, :, :, 0] = cv2.warpAffine(
    #             frames[i, :, :, 0], 
    #             matrix, 
    #             (frames.shape[2], frames.shape[1])
    #         )
    #     return rotated_frames.reshape(15, self.target_height, self.target_width, 1)
    
    # def apply_augmentation(self, frames):
    #     """Apply a random combination of augmentations"""
    #     # Ensure input shape is correct
    #     if frames.shape != (15, self.target_height, self.target_width, 1):
    #         frames = frames.reshape(15, self.target_height, self.target_width, 1)
            
    #     augmentations = [
    #         self.random_flip_horizontal,
    #         self.random_rotation
    #     ]
        
    #     # Apply 1-2 random augmentations
    #     num_augs = np.random.randint(1, 3)
    #     selected_augs = np.random.choice(augmentations, num_augs, replace=False)
        
    #     augmented_frames = frames.copy()
    #     for aug in selected_augs:
    #         augmented_frames = aug(augmented_frames)
            
    #     # Final shape check
    #     if augmented_frames.shape != (15, self.target_height, self.target_width, 1):
    #         augmented_frames = augmented_frames.reshape(15, self.target_height, self.target_width, 1)
            
    #     return augmented_frames 