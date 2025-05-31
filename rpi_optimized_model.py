"""
rpi_optimized_model.py - RPi-optimized model architectures with ResNet and EfficientNet
Optimized for knowledge distillation and RPi4 deployment
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os


class FrameSampler(layers.Layer):
    """Enhanced frame sampler optimized for knowledge distillation"""
    
    def __init__(self, target_frames=8, sampling_method='uniform', **kwargs):
        super(FrameSampler, self).__init__(**kwargs)
        self.target_frames = target_frames
        self.sampling_method = sampling_method
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.shape(inputs)[1]
        
        if self.sampling_method == 'uniform':
            indices = tf.linspace(0.0, tf.cast(sequence_length - 1, tf.float32), self.target_frames)
            indices = tf.cast(tf.round(indices), tf.int32)
        elif self.sampling_method == 'adaptive':
            # Focus on middle frames where gas leaks are more prominent
            start_idx = tf.maximum(0, sequence_length // 4)
            end_idx = tf.minimum(sequence_length, 3 * sequence_length // 4)
            indices = tf.linspace(tf.cast(start_idx, tf.float32), tf.cast(end_idx - 1, tf.float32), self.target_frames)
            indices = tf.cast(tf.round(indices), tf.int32)
        else:  # random
            indices = tf.random.uniform([self.target_frames], 0, sequence_length, dtype=tf.int32)
            indices = tf.sort(indices)
        
        sampled = tf.gather(inputs, indices, axis=1)
        return sampled
    
    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer"""
        batch_size, sequence_length, height, width, channels = input_shape
        return (batch_size, self.target_frames, height, width, channels)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'target_frames': self.target_frames,
            'sampling_method': self.sampling_method
        })
        return config


class SimpleFrameSampler(layers.Layer):
    """Simple frame sampler that takes every nth frame"""
    
    def __init__(self, target_frames=10, **kwargs):
        super().__init__(**kwargs)
        self.target_frames = target_frames
    
    def call(self, inputs):
        # inputs shape: (batch, sequence, height, width, channels)
        sequence_length = tf.shape(inputs)[1]
        
        # Use TensorFlow operations instead of Python conditionals
        indices = tf.cond(
            tf.less_equal(sequence_length, self.target_frames),
            lambda: tf.range(sequence_length),  # If sequence is shorter, use all frames
            lambda: tf.cast(tf.linspace(0.0, tf.cast(sequence_length - 1, tf.float32), self.target_frames), tf.int32)
        )
        
        return tf.gather(inputs, indices, axis=1)
    
    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer"""
        batch_size, sequence_length, height, width, channels = input_shape
        
        if sequence_length is not None and sequence_length <= self.target_frames:
            output_sequence_length = sequence_length
        else:
            output_sequence_length = self.target_frames
            
        return (batch_size, output_sequence_length, height, width, channels)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'target_frames': self.target_frames
        })
        return config


class ResidualBlock(layers.Layer):
    """Enhanced ResidualBlock with improved normalization and activation"""
    
    def __init__(self, filters, strides=1, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        
        # Main path layers
        self.conv1 = layers.Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        
        # Shortcut path
        self.shortcut = None
        if strides != 1:
            self.shortcut_conv = layers.Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)
            self.shortcut_bn = layers.BatchNormalization()
            self.shortcut = True
    
    def call(self, inputs, training=None):
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Shortcut connection
        if self.shortcut:
            shortcut = self.shortcut_conv(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
        else:
            shortcut = inputs
        
        # Add shortcut and apply activation
        x = tf.nn.relu(x + shortcut)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'strides': self.strides
        })
        return config


class EfficientBlock(layers.Layer):
    """Enhanced EfficientNet-style block with squeeze-and-excitation"""
    
    def __init__(self, filters, kernel_size=(3, 3), expand_ratio=2, se_ratio=0.25, **kwargs):
        super(EfficientBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        
        # Expansion
        expanded_filters = int(filters * expand_ratio)
        self.expand_conv = layers.Conv2D(expanded_filters, (1, 1), padding='same', use_bias=False)
        self.expand_bn = layers.BatchNormalization()
        
        # Depthwise
        self.depthwise_conv = layers.DepthwiseConv2D(kernel_size, padding='same', use_bias=False)
        self.depthwise_bn = layers.BatchNormalization()
        
        # SE block
        se_filters = max(1, int(filters * se_ratio))
        self.se_squeeze = layers.GlobalAveragePooling2D()
        self.se_reduce = layers.Dense(se_filters, activation='relu')
        self.se_expand = layers.Dense(expanded_filters, activation='sigmoid')
        
        # Output
        self.project_conv = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)
        self.project_bn = layers.BatchNormalization()
    
    def call(self, inputs, training=None):
        # Expansion
        x = self.expand_conv(inputs)
        x = self.expand_bn(x, training=training)
        x = tf.nn.relu(x)
        
        # Depthwise
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x, training=training)
        x = tf.nn.relu(x)
        
        # Squeeze-and-Excitation
        se = self.se_squeeze(x)
        se = self.se_reduce(se)
        se = self.se_expand(se)
        se = tf.expand_dims(tf.expand_dims(se, 1), 1)
        x = x * se
        
        # Project
        x = self.project_conv(x)
        x = self.project_bn(x, training=training)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'expand_ratio': self.expand_ratio,
            'se_ratio': self.se_ratio
        })
        return config


def create_resnet_model(input_shape=(15, 120, 160, 1), target_frames=10, num_classes=2):
    """
    Simplified ResNet-based model that avoids complex custom layers
    This version should save and load reliably
    """
    
    input_layer = layers.Input(shape=input_shape)
    sampled_frames = FrameSampler(target_frames=target_frames, sampling_method='adaptive')(input_layer)
    
    # Simplified CNN without custom ResidualBlocks to avoid weight issues
    def create_simple_cnn():
        model = tf.keras.Sequential([
            # Initial convolution
            layers.Conv2D(16, (7, 7), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
            
            # First block (no residual connections to avoid complexity)
            layers.Conv2D(16, (3, 3), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(16, (3, 3), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            # Second block with downsampling
            layers.Conv2D(24, (3, 3), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(24, (3, 3), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            # Third block with downsampling
            layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(32, (3, 3), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            # Global pooling
            layers.GlobalAveragePooling2D(),
            layers.Dense(48, activation='relu'),
            layers.Dropout(0.3)
        ], name="simple_resnet_cnn")
        return model
    
    # Apply CNN to each frame
    features = layers.TimeDistributed(create_simple_cnn())(sampled_frames)
    
    # Temporal processing
    temporal_features = layers.Bidirectional(
        layers.GRU(32, dropout=0.3, recurrent_dropout=0.2, return_sequences=True)
    )(features)
    
    # Self-attention mechanism
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=16)(
        temporal_features, temporal_features
    )
    attention_pooled = layers.GlobalAveragePooling1D()(attention)
    
    # Classification head
    dense1 = layers.Dense(64, activation='relu')(attention_pooled)
    dropout1 = layers.Dropout(0.4)(dense1)
    
    dense2 = layers.Dense(32, activation='relu')(dropout1)
    dropout2 = layers.Dropout(0.3)(dense2)
    
    output = layers.Dense(num_classes, activation='softmax')(dropout2)
    
    model = models.Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_efficientnet_model(input_shape=(15, 120, 160, 1), target_frames=10, num_classes=2):
    """
    EfficientNet-based model optimized for RPi4
    EfficientNet is designed for efficiency while maintaining accuracy
    """
    
    input_layer = layers.Input(shape=input_shape)
    sampled_frames = FrameSampler(target_frames=target_frames, sampling_method='adaptive')(input_layer)
    
    # EfficientNet-style CNN
    cnn_layers = [
        # Stem
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        # EfficientNet blocks
        EfficientBlock(16, (3, 3), expand_ratio=1, se_ratio=0.25),
        EfficientBlock(16, (3, 3), expand_ratio=2, se_ratio=0.25),
        
        EfficientBlock(24, (3, 3), expand_ratio=2, se_ratio=0.25),
        EfficientBlock(24, (3, 3), expand_ratio=2, se_ratio=0.25),
        
        EfficientBlock(32, (5, 5), expand_ratio=2, se_ratio=0.25),
        EfficientBlock(32, (5, 5), expand_ratio=2, se_ratio=0.25),
        
        # Head
        layers.Conv2D(64, (1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3)
    ]
    
    # Apply CNN to each frame
    features = layers.TimeDistributed(
        models.Sequential(cnn_layers, name="efficientnet_feature_extractor")
    )(sampled_frames)
    
    # Temporal processing optimized for efficiency
    lstm_out = layers.LSTM(48, dropout=0.3, recurrent_dropout=0.2)(features)
    
    # Classification head
    dense1 = layers.Dense(64, activation='relu')(lstm_out)
    dropout1 = layers.Dropout(0.4)(dense1)
    
    dense2 = layers.Dense(32, activation='relu')(dropout1)
    dropout2 = layers.Dropout(0.3)(dense2)
    
    output = layers.Dense(num_classes, activation='softmax')(dropout2)
    
    model = models.Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_enhanced_lightweight_model(input_shape=(15, 120, 160, 1), target_frames=12, num_classes=2):
    """
    Enhanced lightweight model - improved version that works well with knowledge distillation
    This is based on the successful lightweight model that reached 98% with teacher
    """
    
    sequence_length, height, width, channels = input_shape
    
    input_layer = layers.Input(shape=input_shape)
    sampled_frames = FrameSampler(target_frames=target_frames, sampling_method='adaptive')(input_layer)
    
    # Enhanced CNN with better architecture for knowledge distillation
    if height * width >= 19200:  # 120x160 or larger
        cnn_layers = [
            # Block 1
            layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(20, (1, 1), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),
            
            # Block 2
            layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(32, (1, 1), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),
            
            # Block 3
            layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(48, (1, 1), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),
            
            # Global pooling and feature extraction
            layers.GlobalAveragePooling2D(),
            layers.Dense(40, activation='relu'),
            layers.Dropout(0.25)
        ]
        lstm_units = 40
    else:
        cnn_layers = [
            # Simplified for smaller resolutions
            layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(16, (1, 1), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),
            
            layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(24, (1, 1), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),
            
            layers.GlobalAveragePooling2D(),
            layers.Dense(24, activation='relu'),
            layers.Dropout(0.25)
        ]
        lstm_units = 24
    
    # Apply CNN to each frame
    features = layers.TimeDistributed(
        models.Sequential(cnn_layers, name="enhanced_lightweight_cnn")
    )(sampled_frames)
    
    # Enhanced temporal processing for knowledge distillation
    # Bidirectional processing for better feature learning from teacher
    temporal_features = layers.Bidirectional(
        layers.GRU(lstm_units // 2, dropout=0.3, recurrent_dropout=0.2)
    )(features)
    
    # Enhanced classification head optimized for distillation
    dense1 = layers.Dense(lstm_units, activation='relu')(temporal_features)
    dropout1 = layers.Dropout(0.4)(dense1)
    
    dense2 = layers.Dense(lstm_units // 2, activation='relu')(dropout1)
    dropout2 = layers.Dropout(0.3)(dense2)
    
    output = layers.Dense(num_classes, activation='softmax')(dropout2)
    
    model = models.Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_rpi_model(model_type="enhanced_lightweight", input_shape=(15, 120, 160, 1), 
                  target_frames=10, num_classes=2):
    """
    Main function to create RPi-optimized models
    
    Args:
        model_type: "resnet", "efficientnet", "enhanced_lightweight"
        input_shape: (sequence, height, width, channels)
        target_frames: Number of frames to process
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    
    print(f"[INFO] Creating {model_type} model for shape: {input_shape}")
    
    if model_type == "resnet":
        return create_resnet_model(input_shape, target_frames, num_classes)
    elif model_type == "efficientnet":
        return create_efficientnet_model(input_shape, target_frames, num_classes)
    elif model_type == "enhanced_lightweight":
        return create_enhanced_lightweight_model(input_shape, target_frames, num_classes)
    else:
        raise ValueError("model_type must be 'resnet', 'efficientnet', or 'enhanced_lightweight'")


def quantize_model(keras_model_path, output_path=None):
    """
    Convert Keras model to TensorFlow Lite with quantization
    Enhanced for RPi4 deployment with better error handling and GRU compatibility
    """
    
    try:
        # Load the model with custom objects
        custom_objects = {
            'FrameSampler': FrameSampler,
            'SimpleFrameSampler': SimpleFrameSampler,
            'ResidualBlock': ResidualBlock,
            'EfficientBlock': EfficientBlock
        }
        
        print(f"[INFO] Loading model for quantization: {keras_model_path}")
        model = tf.keras.models.load_model(keras_model_path, custom_objects=custom_objects)
        
        print(f"[INFO] Model loaded successfully")
        print(f"[INFO] Model input shape: {model.input_shape}")
        print(f"[INFO] Model output shape: {model.output_shape}")
        
        # Determine output path
        if output_path is None:
            output_path = keras_model_path.replace('.keras', '.tflite').replace('.h5', '.tflite')
        
        # Convert to TensorFlow Lite with optimizations
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Enable optimizations for RPi4
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Enhanced settings for models with bidirectional GRU/LSTM
        converter.experimental_enable_resource_variables = True
        converter._experimental_lower_tensor_list_ops = False
        
        # Use SELECT_TF_OPS to support complex operations
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        print("[INFO] Using TFLite with SELECT_TF_OPS for GRU/LSTM compatibility")
        
        # Set representative dataset for quantization if possible
        try:
            # Create a simple representative dataset using random data
            def representative_data_gen():
                for _ in range(10):
                    # Generate random input data matching the model's input shape
                    input_shape = model.input_shape
                    if input_shape[0] is None:  # Remove batch dimension
                        input_shape = input_shape[1:]
                    
                    data = np.random.random((1,) + input_shape).astype(np.float32)
                    yield [data]
            
            converter.representative_dataset = representative_data_gen
            print("[INFO] Representative dataset configured for quantization")
            
        except Exception as e:
            print(f"[WARNING] Could not set up representative dataset: {e}")
        
        # Convert the model
        print("[INFO] Converting model to TensorFlow Lite...")
        tflite_model = converter.convert()
        
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
        print(f"[ERROR] TFLite conversion failed: {e}")
        
        # Try fallback conversion without quantization
        try:
            print("[WARNING] Attempting fallback conversion without integer quantization...")
            
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.experimental_enable_resource_variables = True
            converter._experimental_lower_tensor_list_ops = False
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            
            # Only use basic optimizations
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            fallback_path = output_path.replace('.tflite', '_fallback.tflite')
            tflite_model = converter.convert()
            
            with open(fallback_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"[INFO] Fallback TFLite model saved: {fallback_path}")
            return fallback_path
            
        except Exception as fallback_error:
            print(f"[ERROR] Fallback conversion also failed: {fallback_error}")
            import traceback
            traceback.print_exc()
            return None