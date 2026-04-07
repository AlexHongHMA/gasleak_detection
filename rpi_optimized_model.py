"""
rpi_optimized_model.py - Complete RPi-optimized model architectures with VideoGasNet integration
Enhanced ResNet and EfficientNet models based on VideoGasNet's GasNet-2 foundation
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


def create_resnet_model(input_shape=(15, 120, 160, 1), target_frames=10, num_classes=2):
    """
    Complete ResNet + GasNet-2 Hybrid Architecture
    
    REASONING:
    1. Domain Expertise First: Start with VideoGasNet's proven Conv-Pool structures for gas detection
    2. ResNet Enhancement: Add skip connections around these structures to enable deeper learning
    3. Unified Design: Seamlessly integrate ResNet principles throughout, not just stack on top
    4. RPi4 Optimization: Keep parameter count reasonable while maximizing architectural benefits
    
    ARCHITECTURE FLOW:
    VideoGasNet Foundation + ResNet Skip Connections → Enhanced Temporal Processing → Classification
    
    KEY INNOVATIONS:
    - Skip connections around GasNet-2 Conv-Pool structures
    - ResNet-style bottleneck after proven gas detection layers
    - Residual connections in temporal processing
    - Deep integration, not shallow stacking
    """
    
    sequence_length, height, width, channels = input_shape
    
    print(f"[INFO] Creating ResNet-Enhanced GasNet-2 Hybrid")
    print(f"[INFO] Strategy: Deep ResNet integration with VideoGasNet foundation")
    print(f"[INFO] Key: Skip connections enable deeper gas pattern learning")
    
    input_layer = layers.Input(shape=input_shape)
    sampled_frames = FrameSampler(target_frames=target_frames, sampling_method='adaptive')(input_layer)
    
    # Automatic scaling based on resolution (VideoGasNet used 240x320)
    total_pixels = height * width
    scale_factor = max(0.25, min(1.0, total_pixels / (240 * 320)))
    
    # Filter scaling: maintain VideoGasNet proportions but scale for resolution
    base_filters = max(4, int(8 * scale_factor))  # Base filter count
    gasnet_filters_1 = base_filters // 2  # GasNet Conv-Pool 1: 4 filters (scaled)
    gasnet_filters_2 = base_filters      # GasNet Conv-Pool 2: 8 filters (scaled) 
    resnet_filters_1 = base_filters * 2  # ResNet enhancement 1: 16 filters (scaled)
    resnet_filters_2 = base_filters * 3  # ResNet enhancement 2: 24 filters (scaled)
    
    print(f"[INFO] Filter progression: {gasnet_filters_1} → {gasnet_filters_2} → {resnet_filters_1} → {resnet_filters_2}")
    print(f"[INFO] Resolution scale factor: {scale_factor:.2f}")
    
    def create_resnet_enhanced_gasnet_cnn():
        """
        ResNet-Enhanced GasNet-2 CNN with deep integration
        Each major structure has ResNet-style skip connections for deeper learning
        """
        
        cnn_input = layers.Input(shape=(height, width, channels))
        
        # ========== ResNet-Style Stem (prepare for gas detection) ==========
        x = layers.Conv2D(gasnet_filters_1, (3, 3), padding='same', use_bias=False, 
                         name="stem_conv")(cnn_input)
        x = layers.BatchNormalization(name="stem_bn")(x)
        stem_features = layers.ReLU(name="stem_relu")(x)
        
        # ========== GasNet-2 Conv-Pool Structure 1 with ResNet Enhancement ==========
        # Main path: VideoGasNet's proven Conv-Pool Structure 1
        x = layers.Conv2D(gasnet_filters_1, (3, 3), activation='relu', padding='same', 
                         kernel_regularizer=tf.keras.regularizers.l2(0.001), 
                         name="gasnet1_conv")(stem_features)
        x = layers.BatchNormalization(name="gasnet1_bn")(x)
        x = layers.Dropout(0.1, name="gasnet1_dropout")(x)
        x = layers.MaxPooling2D((2, 2), padding='same', name="gasnet1_pool")(x)
        
        # ResNet-style skip connection around Conv-Pool 1
        # Adjust stem_features to match pooled dimensions and channels
        stem_pooled = layers.MaxPooling2D((2, 2), padding='same', name="stem_pool")(stem_features)
        if stem_pooled.shape[-1] != gasnet_filters_1:
            stem_projected = layers.Conv2D(gasnet_filters_1, (1, 1), padding='same', 
                                         use_bias=False, name="stem_projection")(stem_pooled)
            stem_projected = layers.BatchNormalization(name="stem_projection_bn")(stem_projected)
        else:
            stem_projected = stem_pooled
        
        # ResNet addition: stem features + Conv-Pool 1 features
        gasnet1_with_skip = layers.Add(name="gasnet1_skip_add")([x, stem_projected])
        gasnet1_out = layers.ReLU(name="gasnet1_skip_relu")(gasnet1_with_skip)
        
        # ========== GasNet-2 Conv-Pool Structure 2 with ResNet Enhancement ==========
        # Main path: VideoGasNet's proven Conv-Pool Structure 2
        x = layers.Conv2D(gasnet_filters_2, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001),
                         name="gasnet2_conv")(gasnet1_out)
        x = layers.BatchNormalization(name="gasnet2_bn")(x)
        x = layers.Dropout(0.1, name="gasnet2_dropout")(x)
        x = layers.MaxPooling2D((2, 2), padding='same', name="gasnet2_pool")(x)
        
        # ResNet-style skip connection around Conv-Pool 2
        gasnet1_pooled = layers.MaxPooling2D((2, 2), padding='same', name="gasnet1_to_2_pool")(gasnet1_out)
        if gasnet1_pooled.shape[-1] != gasnet_filters_2:
            gasnet1_projected = layers.Conv2D(gasnet_filters_2, (1, 1), padding='same',
                                            use_bias=False, name="gasnet1_to_2_projection")(gasnet1_pooled)
            gasnet1_projected = layers.BatchNormalization(name="gasnet1_to_2_projection_bn")(gasnet1_projected)
        else:
            gasnet1_projected = gasnet1_pooled
        
        # ResNet addition: previous features + Conv-Pool 2 features  
        gasnet2_with_skip = layers.Add(name="gasnet2_skip_add")([x, gasnet1_projected])
        gasnet2_out = layers.ReLU(name="gasnet2_skip_relu")(gasnet2_with_skip)
        
        # ========== ResNet Bottleneck Enhancement (learn complex patterns) ==========
        # Now that we have gas-specific features, use ResNet bottleneck to learn complex patterns
        
        # Bottleneck Block 1: Learn higher-level gas patterns
        # 1x1 conv to reduce dimensions
        x = layers.Conv2D(resnet_filters_1 // 2, (1, 1), padding='same', use_bias=False,
                         name="bottleneck1_reduce")(gasnet2_out)
        x = layers.BatchNormalization(name="bottleneck1_reduce_bn")(x)
        x = layers.ReLU(name="bottleneck1_reduce_relu")(x)
        
        # 3x3 conv for pattern learning
        x = layers.Conv2D(resnet_filters_1 // 2, (3, 3), padding='same', use_bias=False,
                         name="bottleneck1_conv")(x)
        x = layers.BatchNormalization(name="bottleneck1_conv_bn")(x)
        x = layers.ReLU(name="bottleneck1_conv_relu")(x)
        
        # 1x1 conv to expand dimensions
        x = layers.Conv2D(resnet_filters_1, (1, 1), padding='same', use_bias=False,
                         name="bottleneck1_expand")(x)
        x = layers.BatchNormalization(name="bottleneck1_expand_bn")(x)
        
        # Skip connection for bottleneck 1
        if gasnet2_out.shape[-1] != resnet_filters_1:
            gasnet2_projected = layers.Conv2D(resnet_filters_1, (1, 1), padding='same',
                                            use_bias=False, name="gasnet2_to_bottleneck1_projection")(gasnet2_out)
            gasnet2_projected = layers.BatchNormalization(name="gasnet2_to_bottleneck1_projection_bn")(gasnet2_projected)
        else:
            gasnet2_projected = gasnet2_out
        
        bottleneck1_out = layers.Add(name="bottleneck1_add")([x, gasnet2_projected])
        bottleneck1_out = layers.ReLU(name="bottleneck1_relu")(bottleneck1_out)
        
        # Bottleneck Block 2: Learn even more complex patterns with downsampling
        # 1x1 conv to reduce dimensions  
        x = layers.Conv2D(resnet_filters_2 // 2, (1, 1), padding='same', use_bias=False,
                         name="bottleneck2_reduce")(bottleneck1_out)
        x = layers.BatchNormalization(name="bottleneck2_reduce_bn")(x)
        x = layers.ReLU(name="bottleneck2_reduce_relu")(x)
        
        # 3x3 conv with stride for downsampling
        x = layers.Conv2D(resnet_filters_2 // 2, (3, 3), strides=2, padding='same', use_bias=False,
                         name="bottleneck2_conv")(x)
        x = layers.BatchNormalization(name="bottleneck2_conv_bn")(x)
        x = layers.ReLU(name="bottleneck2_conv_relu")(x)
        
        # 1x1 conv to expand dimensions
        x = layers.Conv2D(resnet_filters_2, (1, 1), padding='same', use_bias=False,
                         name="bottleneck2_expand")(x)
        x = layers.BatchNormalization(name="bottleneck2_expand_bn")(x)
        
        # Skip connection with downsampling
        bottleneck1_downsampled = layers.Conv2D(resnet_filters_2, (1, 1), strides=2, padding='same',
                                               use_bias=False, name="bottleneck1_to_2_projection")(bottleneck1_out)
        bottleneck1_downsampled = layers.BatchNormalization(name="bottleneck1_to_2_projection_bn")(bottleneck1_downsampled)
        
        bottleneck2_out = layers.Add(name="bottleneck2_add")([x, bottleneck1_downsampled])
        bottleneck2_out = layers.ReLU(name="bottleneck2_relu")(bottleneck2_out)
        
        # ========== Feature Consolidation ==========
        # Global pooling to create fixed-size representation
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(bottleneck2_out)
        
        # Dense layer with residual connection for feature refinement
        dense_input = x
        x = layers.Dense(max(64, int(96 * scale_factor)), activation='relu', name="feature_dense1")(x)
        x = layers.Dropout(0.3, name="feature_dropout1")(x)
        
        # Another dense layer 
        x = layers.Dense(max(32, int(48 * scale_factor)), activation='relu', name="feature_dense2")(x)
        
        # Residual connection in feature space (if dimensions match)
        if dense_input.shape[-1] == x.shape[-1]:
            x = layers.Add(name="feature_residual")([dense_input, x])
        
        cnn_output = layers.Dropout(0.25, name="feature_dropout2")(x)
        
        return models.Model(inputs=cnn_input, outputs=cnn_output, name="resnet_enhanced_gasnet2_cnn")
    
    # Apply enhanced CNN to each frame
    frame_features = layers.TimeDistributed(create_resnet_enhanced_gasnet_cnn(), name="frame_feature_extraction")(sampled_frames)
    
    # ========== Enhanced Temporal Processing with ResNet Concepts ==========
    # Use bidirectional processing to capture temporal patterns in both directions
    temporal_forward = layers.GRU(32, dropout=0.3, recurrent_dropout=0.2, 
                                 return_sequences=True, name="temporal_gru_forward")(frame_features)
    temporal_backward = layers.GRU(32, dropout=0.3, recurrent_dropout=0.2, 
                                  return_sequences=True, go_backwards=True, name="temporal_gru_backward")(frame_features)
    
    # Combine bidirectional features
    temporal_combined = layers.Concatenate(name="temporal_combine")([temporal_forward, temporal_backward])
    
    # Temporal attention mechanism (learn which frames are most important)
    attention_weights = layers.Dense(64, activation='tanh', name="attention_dense")(temporal_combined)
    attention_weights = layers.Dense(1, activation='softmax', name="attention_weights")(attention_weights)
    
    # Apply attention to features
    attended_features = layers.Multiply(name="attention_apply")([temporal_combined, attention_weights])
    temporal_output = layers.GlobalAveragePooling1D(name="temporal_pool")(attended_features)
    
    # ========== Classification Head with Residual Connections ==========
    # Multi-layer classification with skip connections
    classifier_input = temporal_output
    
    # First classification layer
    x = layers.Dense(64, activation='relu', name="classifier_dense1")(classifier_input)
    x = layers.Dropout(0.4, name="classifier_dropout1")(x)
    
    # Second classification layer
    x = layers.Dense(32, activation='relu', name="classifier_dense2")(x)
    x = layers.Dropout(0.3, name="classifier_dropout2")(x)
    
    # Residual connection in classifier (if dimensions allow)
    if classifier_input.shape[-1] == 32:
        x = layers.Add(name="classifier_residual")([classifier_input, x])
    
    # Final output
    output = layers.Dense(num_classes, activation='softmax', name="classification_output")(x)
    
    # Create and compile model
    model = models.Model(inputs=input_layer, outputs=output, name="ResNet_Enhanced_GasNet2")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),  # Slightly lower LR for deeper network
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print architecture summary
    total_params = model.count_params()
    print(f"[INFO] ResNet-Enhanced GasNet-2 created successfully")
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Estimated model size: {(total_params * 4) / (1024*1024):.2f} MB")
    print(f"[INFO] Key features: Skip connections, bottleneck blocks, temporal attention")
    
    return model


def create_efficientnet_model(input_shape=(15, 120, 160, 1), target_frames=10, num_classes=2):
    """
    Complete EfficientNet + GasNet-2 Hybrid Architecture
    
    REASONING:
    1. Domain Expertise First: Start with VideoGasNet's proven Conv-Pool structures for gas detection
    2. Spatial Attention Enhancement: SE blocks are perfect for gas plume localization - integrate deeply
    3. Parameter Efficiency: Use depthwise separable convolutions for RPi4 efficiency
    4. Inverted Residuals: Add efficient pattern learning blocks after proven gas detection layers
    5. Compound Enhancement: Scale the architecture intelligently based on input resolution
    
    ARCHITECTURE FLOW:
    VideoGasNet Foundation + SE Attention + Efficient Convolutions → Enhanced Temporal → Classification
    
    KEY INNOVATIONS:
    - SE blocks after each major structure for gas plume spatial attention
    - Depthwise separable convolutions for efficiency
    - Inverted residual blocks for advanced pattern learning
    - Multi-scale spatial attention throughout
    - Attention-aware temporal processing
    """
    
    sequence_length, height, width, channels = input_shape
    
    print(f"[INFO] Creating EfficientNet-Enhanced GasNet-2 Hybrid")
    print(f"[INFO] Strategy: Spatial attention + efficiency with VideoGasNet foundation")
    print(f"[INFO] Key: SE blocks for gas plume localization + efficient convolutions")
    
    input_layer = layers.Input(shape=input_shape)
    sampled_frames = FrameSampler(target_frames=target_frames, sampling_method='adaptive')(input_layer)
    
    # Intelligent scaling based on resolution and EfficientNet principles
    total_pixels = height * width
    scale_factor = max(0.25, min(1.0, total_pixels / (240 * 320)))
    
    # EfficientNet-style compound scaling: scale width, depth, and resolution together
    width_multiplier = scale_factor
    depth_multiplier = max(0.5, scale_factor)  # Don't go too shallow
    
    # Filter progression with EfficientNet-style scaling
    base_filters = max(8, int(16 * width_multiplier))
    gasnet_filters_1 = base_filters // 2      # GasNet Conv-Pool 1: 8 filters (scaled)
    gasnet_filters_2 = base_filters           # GasNet Conv-Pool 2: 16 filters (scaled)
    efficient_filters_1 = base_filters * 2   # EfficientNet block 1: 32 filters (scaled)
    efficient_filters_2 = base_filters * 3   # EfficientNet block 2: 48 filters (scaled)
    
    # SE ratio for attention strength
    se_ratio = 0.25
    
    print(f"[INFO] Filter progression: {gasnet_filters_1} → {gasnet_filters_2} → {efficient_filters_1} → {efficient_filters_2}")
    print(f"[INFO] Width multiplier: {width_multiplier:.2f}, SE ratio: {se_ratio}")
    
    def create_se_block(input_tensor, filters, ratio=0.25, name_prefix="se"):
        """
        Squeeze-and-Excitation block for spatial attention
        Critical for gas plume localization in VideoGasNet context
        """
        se_filters = max(1, int(filters * ratio))
        
        # Squeeze: Global information aggregation
        se = layers.GlobalAveragePooling2D(name=f"{name_prefix}_squeeze")(input_tensor)
        se = layers.Reshape((1, 1, filters), name=f"{name_prefix}_reshape")(se)
        
        # Excitation: Channel attention learning
        se = layers.Dense(se_filters, activation='relu', name=f"{name_prefix}_reduce")(se)
        se = layers.Dense(filters, activation='sigmoid', name=f"{name_prefix}_excite")(se)
        
        # Apply attention
        output = layers.Multiply(name=f"{name_prefix}_apply")([input_tensor, se])
        return output
    
    def create_efficient_inverted_residual(input_tensor, output_filters, expand_ratio=2, 
                                         stride=1, name_prefix="inv_res"):
        """
        EfficientNet-style inverted residual block with depthwise separable convolutions
        Efficient pattern learning after gas-specific feature extraction
        """
        input_filters = input_tensor.shape[-1]
        expanded_filters = int(input_filters * expand_ratio)
        
        # Expansion phase (only if expand_ratio > 1)
        if expand_ratio > 1:
            x = layers.Conv2D(expanded_filters, (1, 1), padding='same', use_bias=False,
                             name=f"{name_prefix}_expand")(input_tensor)
            x = layers.BatchNormalization(name=f"{name_prefix}_expand_bn")(x)
            x = layers.ReLU(name=f"{name_prefix}_expand_relu")(x)
        else:
            x = input_tensor
        
        # Depthwise convolution (efficient spatial processing)
        x = layers.DepthwiseConv2D((3, 3), strides=stride, padding='same', use_bias=False,
                                  name=f"{name_prefix}_depthwise")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_depthwise_bn")(x)
        x = layers.ReLU(name=f"{name_prefix}_depthwise_relu")(x)
        
        # SE block for spatial attention (critical for gas detection)
        x = create_se_block(x, expanded_filters, ratio=se_ratio, name_prefix=f"{name_prefix}_se")
        
        # Project back to output dimensions
        x = layers.Conv2D(output_filters, (1, 1), padding='same', use_bias=False,
                         name=f"{name_prefix}_project")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_project_bn")(x)
        
        # Residual connection (only if dimensions match and no stride)
        if stride == 1 and input_filters == output_filters:
            x = layers.Add(name=f"{name_prefix}_add")([input_tensor, x])
        
        return x
    
    def create_efficientnet_enhanced_gasnet_cnn():
        """
        EfficientNet-Enhanced GasNet-2 CNN with deep spatial attention integration
        Each structure enhanced with SE blocks for gas plume focus
        """
        
        cnn_input = layers.Input(shape=(height, width, channels))
        
        # ========== EfficientNet-Style Stem ==========
        # Efficient stem that prepares for gas detection
        x = layers.Conv2D(gasnet_filters_1, (3, 3), strides=1, padding='same', use_bias=False,
                         name="stem_conv")(cnn_input)
        x = layers.BatchNormalization(name="stem_bn")(x)
        stem_features = layers.ReLU(name="stem_relu")(x)
        
        # Early SE attention for initial spatial focus
        stem_with_attention = create_se_block(stem_features, gasnet_filters_1, 
                                            ratio=se_ratio, name_prefix="stem_se")
        
        # ========== GasNet-2 Conv-Pool Structure 1 with SE Enhancement ==========
        # VideoGasNet's proven Conv-Pool Structure 1 enhanced with spatial attention
        x = layers.Conv2D(gasnet_filters_1, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001),
                         name="gasnet1_conv")(stem_with_attention)
        x = layers.BatchNormalization(name="gasnet1_bn")(x)
        x = layers.Dropout(0.1, name="gasnet1_dropout")(x)
        
        # SE attention before pooling (focus on gas-relevant spatial regions)
        x = create_se_block(x, gasnet_filters_1, ratio=se_ratio, name_prefix="gasnet1_se")
        
        x = layers.MaxPooling2D((2, 2), padding='same', name="gasnet1_pool")(x)
        gasnet1_out = x
        
        # ========== GasNet-2 Conv-Pool Structure 2 with SE Enhancement ==========
        # VideoGasNet's proven Conv-Pool Structure 2 enhanced with spatial attention
        x = layers.Conv2D(gasnet_filters_2, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001),
                         name="gasnet2_conv")(gasnet1_out)
        x = layers.BatchNormalization(name="gasnet2_bn")(x)
        x = layers.Dropout(0.1, name="gasnet2_dropout")(x)
        
        # Enhanced SE attention (stronger focus on gas plumes)
        x = create_se_block(x, gasnet_filters_2, ratio=se_ratio * 1.5, name_prefix="gasnet2_se")
        
        x = layers.MaxPooling2D((2, 2), padding='same', name="gasnet2_pool")(x)
        gasnet2_out = x
        
        # ========== EfficientNet Inverted Residual Block 1 ==========
        # Learn more complex patterns from gas-specific features
        efficient_block1 = create_efficient_inverted_residual(
            gasnet2_out, 
            output_filters=efficient_filters_1,
            expand_ratio=3,  # EfficientNet typically uses 3-6
            stride=1,
            name_prefix="efficient_block1"
        )
        
        # ========== EfficientNet Inverted Residual Block 2 with Downsampling ==========
        # Even more complex patterns with spatial downsampling
        efficient_block2 = create_efficient_inverted_residual(
            efficient_block1,
            output_filters=efficient_filters_2,
            expand_ratio=4,  # Higher expansion for more complex patterns
            stride=2,        # Downsample for computational efficiency
            name_prefix="efficient_block2"
        )
        
        # ========== Final Spatial Attention and Feature Consolidation ==========
        # Global attention to focus on most important spatial regions
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(efficient_block2)
        
        # Final SE-style attention in feature space
        feature_attention = layers.Dense(max(16, int(efficient_filters_2 * 0.25)), 
                                       activation='relu', name="final_attention_reduce")(x)
        feature_attention = layers.Dense(x.shape[-1], activation='sigmoid', 
                                       name="final_attention_excite")(feature_attention)
        x = layers.Multiply(name="final_attention_apply")([x, feature_attention])
        
        # Feature refinement with residual connection
        feature_input = x
        x = layers.Dense(max(48, int(72 * width_multiplier)), activation='relu', 
                        name="feature_dense")(x)
        x = layers.Dropout(0.3, name="feature_dropout")(x)
        
        # Add residual if dimensions match
        if feature_input.shape[-1] == x.shape[-1]:
            x = layers.Add(name="feature_residual")([feature_input, x])
        
        cnn_output = x
        
        return models.Model(inputs=cnn_input, outputs=cnn_output, 
                          name="efficientnet_enhanced_gasnet2_cnn")
    
    # Apply enhanced CNN to each frame
    frame_features = layers.TimeDistributed(create_efficientnet_enhanced_gasnet_cnn(), 
                                          name="frame_feature_extraction")(sampled_frames)
    
    # ========== Attention-Aware Temporal Processing ==========
    # Use LSTM with attention mechanism for temporal understanding
    lstm_features = layers.LSTM(48, dropout=0.3, recurrent_dropout=0.2, 
                               return_sequences=True, name="temporal_lstm")(frame_features)
    
    # Temporal self-attention (learn which time steps are most important)
    # Query, Key, Value are all the same (self-attention)
    temporal_attention = layers.MultiHeadAttention(
        num_heads=4, 
        key_dim=12,
        name="temporal_self_attention"
    )(lstm_features, lstm_features)
    
    # Combine LSTM and attention features
    temporal_combined = layers.Add(name="temporal_combine")([lstm_features, temporal_attention])
    
    # Global temporal pooling with attention weights
    temporal_weights = layers.Dense(1, activation='softmax', name="temporal_attention_weights")(temporal_combined)
    weighted_temporal = layers.Multiply(name="temporal_weight_apply")([temporal_combined, temporal_weights])
    temporal_output = layers.GlobalAveragePooling1D(name="temporal_pool")(weighted_temporal)
    
    # ========== Classification Head with Attention ==========
    # Multi-layer classification with attention mechanisms
    
    # First classification layer with attention
    x = layers.Dense(64, activation='relu', name="classifier_dense1")(temporal_output)
    x = layers.Dropout(0.4, name="classifier_dropout1")(x)
    
    # Internal attention in classification
    classifier_attention = layers.Dense(16, activation='relu', name="classifier_attention_reduce")(x)
    classifier_attention = layers.Dense(64, activation='sigmoid', name="classifier_attention_excite")(classifier_attention)
    x = layers.Multiply(name="classifier_attention_apply")([x, classifier_attention])
    
    # Second classification layer
    x = layers.Dense(32, activation='relu', name="classifier_dense2")(x)
    x = layers.Dropout(0.3, name="classifier_dropout2")(x)
    
    # Final output
    output = layers.Dense(num_classes, activation='softmax', name="classification_output")(x)
    
    # Create and compile model
    model = models.Model(inputs=input_layer, outputs=output, name="EfficientNet_Enhanced_GasNet2")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Standard LR for efficient architecture
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print architecture summary
    total_params = model.count_params()
    print(f"[INFO] EfficientNet-Enhanced GasNet-2 created successfully")
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Estimated model size: {(total_params * 4) / (1024*1024):.2f} MB")
    print(f"[INFO] Key features: SE blocks, depthwise convolutions, spatial attention")
    print(f"[INFO] Efficiency: {total_pixels / total_params:.2f} pixels per parameter")
    
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
    Main function to create RPi-optimized models with VideoGasNet integration
    
    Args:
        model_type: "resnet" (GasNet-2+ResNet), "efficientnet" (GasNet-2+EfficientNet), "enhanced_lightweight"
        input_shape: (sequence, height, width, channels)
        target_frames: Number of frames to process
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    
    print(f"[INFO] Creating {model_type} model for shape: {input_shape}")
    
    if model_type == "resnet":
        print("[INFO] Using GasNet-2 + ResNet hybrid architecture")
        return create_resnet_model(input_shape, target_frames, num_classes)
    elif model_type == "efficientnet":
        print("[INFO] Using GasNet-2 + EfficientNet hybrid architecture")
        return create_efficientnet_model(input_shape, target_frames, num_classes)
    elif model_type == "enhanced_lightweight":
        return create_enhanced_lightweight_model(input_shape, target_frames, num_classes)
    else:
        raise ValueError("model_type must be 'resnet' (GasNet-2+ResNet), 'efficientnet' (GasNet-2+EfficientNet), or 'enhanced_lightweight'")


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