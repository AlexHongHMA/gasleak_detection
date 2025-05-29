"""
rpi_model.py - RPi-optimized model architectures
Clean model definitions following the original model.py structure
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import os


@tf.keras.utils.register_keras_serializable()
class FrameSampler(layers.Layer):
    """Custom layer to sample frames from video sequences"""
    
    def __init__(self, target_frames=8, sampling_method='uniform', **kwargs):
        super(FrameSampler, self).__init__(**kwargs)
        self.target_frames = target_frames
        self.sampling_method = sampling_method
    
    def call(self, inputs):
        # inputs shape: (batch, sequence_length, height, width, channels)
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.shape(inputs)[1]
        
        if self.sampling_method == 'uniform':
            # Uniform sampling across the sequence
            indices = tf.linspace(0.0, tf.cast(sequence_length - 1, tf.float32), self.target_frames)
            indices = tf.cast(tf.round(indices), tf.int32)
        else:  # 'random'
            indices = tf.random.uniform([self.target_frames], 0, sequence_length, dtype=tf.int32)
            indices = tf.sort(indices)
        
        # Gather frames
        sampled = tf.gather(inputs, indices, axis=1)
        return sampled
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'target_frames': self.target_frames,
            'sampling_method': self.sampling_method
        })
        return config


def create_ultra_fast_model(input_shape=(15, 120, 160, 1), target_frames=5, num_classes=2):
    """Ultra-fast model for maximum FPS on RPi4"""
    
    input_layer = layers.Input(shape=input_shape)
    sampled_frames = FrameSampler(target_frames=target_frames)(input_layer)
    
    # Minimal CNN architecture
    features = layers.TimeDistributed(
        models.Sequential([
            layers.DepthwiseConv2D((3, 3), padding='same', activation='relu'),
            layers.Conv2D(8, (1, 1), activation='relu'),
            layers.MaxPooling2D((4, 4)),
            layers.GlobalAveragePooling2D()
        ])
    )(sampled_frames)
    
    # Minimal LSTM
    lstm_out = layers.LSTM(12, dropout=0.2)(features)
    
    # Output layer
    output = layers.Dense(num_classes, activation='softmax')(lstm_out)
    
    model = models.Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_lightweight_model(input_shape=(15, 120, 160, 1), target_frames=8, num_classes=2):
    """Lightweight model balancing speed and accuracy"""
    
    sequence_length, height, width, channels = input_shape
    
    input_layer = layers.Input(shape=input_shape)
    sampled_frames = FrameSampler(target_frames=target_frames)(input_layer)
    
    # Adaptive CNN based on resolution
    if height * width >= 240 * 320:  # Large resolution
        cnn_filters = [16, 24, 32]
        lstm_units = 32
        pooling_sizes = [(2, 2), (4, 4), (2, 2)]
    else:  # Smaller resolution
        cnn_filters = [12, 16, 24]
        lstm_units = 24
        pooling_sizes = [(2, 2), (2, 2), (2, 2)]
    
    # Build CNN layers dynamically
    cnn_layers = []
    for i, filters in enumerate(cnn_filters):
        cnn_layers.extend([
            layers.DepthwiseConv2D((3, 3), padding='same', activation='relu'),
            layers.Conv2D(filters, (1, 1), activation='relu'),
            layers.MaxPooling2D(pooling_sizes[i])
        ])
    
    # Add global pooling at the end
    cnn_layers.append(layers.GlobalAveragePooling2D())
    
    # Apply CNN to each frame
    features = layers.TimeDistributed(models.Sequential(cnn_layers))(sampled_frames)
    
    # LSTM processing
    lstm_out = layers.LSTM(lstm_units, dropout=0.3)(features)
    
    # Output layer
    output = layers.Dense(num_classes, activation='softmax')(lstm_out)
    
    model = models.Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_mobilenet_model(input_shape=(15, 120, 160, 1), target_frames=8, num_classes=2):
    """MobileNet-based model for balanced performance"""
    
    sequence_length, height, width, channels = input_shape
    
    input_layer = layers.Input(shape=input_shape)
    sampled_frames = FrameSampler(target_frames=target_frames)(input_layer)
    
    # Create base MobileNetV2 without top layers
    base_model = MobileNetV2(
        input_shape=(height, width, 3),  # MobileNet expects 3 channels
        include_top=False,
        weights=None,  # Don't use pretrained weights for grayscale
        pooling='avg'
    )
    
    # Adapt for single channel input
    mobilenet_layers = []
    
    # Convert single channel to 3 channels
    mobilenet_layers.append(layers.Conv2D(3, (1, 1), activation='relu'))
    
    # Add MobileNet layers (simplified)
    mobilenet_layers.extend([
        layers.DepthwiseConv2D((3, 3), padding='same', activation='relu'),
        layers.Conv2D(16, (1, 1), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.DepthwiseConv2D((3, 3), padding='same', activation='relu'),
        layers.Conv2D(32, (1, 1), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.DepthwiseConv2D((3, 3), padding='same', activation='relu'),
        layers.Conv2D(64, (1, 1), activation='relu'),
        layers.GlobalAveragePooling2D()
    ])
    
    # Apply to each frame
    features = layers.TimeDistributed(models.Sequential(mobilenet_layers))(sampled_frames)
    
    # LSTM size based on resolution
    lstm_units = min(64, max(32, (height * width) // 10000))
    
    lstm_out = layers.LSTM(
        lstm_units, 
        dropout=0.3, 
        recurrent_dropout=0.2
    )(features)
    
    # Dense layers
    dense1 = layers.Dense(lstm_units * 2, activation='relu')(lstm_out)
    dropout1 = layers.Dropout(0.4)(dense1)
    
    dense2 = layers.Dense(lstm_units, activation='relu')(dropout1)
    dropout2 = layers.Dropout(0.3)(dense2)
    
    output = layers.Dense(num_classes, activation='softmax')(dropout2)
    
    model = models.Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_rpi_model(model_type="lightweight", input_shape=(15, 120, 160, 1), 
                  target_frames=8, num_classes=2):
    """
    Main function to create RPi-optimized models
    
    Args:
        model_type: "ultra", "lightweight", or "mobilenet"
        input_shape: (sequence, height, width, channels)
        target_frames: Number of frames to process
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    
    print(f"[INFO] Creating {model_type} model for shape: {input_shape}")
    
    if model_type == "ultra":
        return create_ultra_fast_model(input_shape, min(target_frames, 5), num_classes)
    elif model_type == "lightweight":
        return create_lightweight_model(input_shape, target_frames, num_classes)
    elif model_type == "mobilenet":
        return create_mobilenet_model(input_shape, target_frames, num_classes)
    else:
        raise ValueError("model_type must be 'ultra', 'lightweight', or 'mobilenet'")


def quantize_model(keras_model_path, output_path=None):
    """
    Convert Keras model to TensorFlow Lite with quantization
    
    Args:
        keras_model_path: Path to the saved Keras model
        output_path: Path to save the TFLite model (optional)
    
    Returns:
        Path to the TFLite model
    """
    
    try:
        # Load the model with custom objects
        custom_objects = {'FrameSampler': FrameSampler}
        model = tf.keras.models.load_model(keras_model_path, custom_objects=custom_objects)
        
        print(f"[INFO] Loaded model for quantization: {keras_model_path}")
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Enable optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Determine output path
        if output_path is None:
            output_path = keras_model_path.replace('.keras', '.tflite').replace('.h5', '.tflite')
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"[INFO] Quantized model saved: {output_path}")
        
        # Compare sizes
        original_size = os.path.getsize(keras_model_path)
        quantized_size = os.path.getsize(output_path)
        compression_ratio = original_size / quantized_size
        
        print(f"[INFO] Size reduction: {original_size/1024:.1f}KB -> {quantized_size/1024:.1f}KB "
              f"(compression: {compression_ratio:.1f}x)")
        
        return output_path
        
    except Exception as e:
        print(f"[ERROR] Quantization failed: {e}")
        return None