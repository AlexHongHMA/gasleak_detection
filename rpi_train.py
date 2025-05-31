"""
rpi_train.py - Enhanced training module for RPi models with knowledge distillation
Optimized for ResNet, EfficientNet, and Enhanced Lightweight models
"""

import os
import time
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import classification_report

from rpi_optimized_model import get_rpi_model, quantize_model
from src.loader.loader import DataGenerator


class KnowledgeDistillationLoss(tf.keras.losses.Loss):
    """Knowledge distillation loss combining hard and soft targets"""
    
    def __init__(self, alpha=0.7, temperature=4.0, name="knowledge_distillation_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.temperature = temperature
    
    def call(self, y_true, y_pred):
        # y_pred should be [student_logits, teacher_logits] when using distillation
        if isinstance(y_pred, list) and len(y_pred) == 2:
            student_logits, teacher_logits = y_pred[0], y_pred[1]
            
            # Hard loss (student vs true labels)
            hard_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, student_logits)
            
            # Soft loss (student vs teacher)
            student_soft = tf.nn.softmax(student_logits / self.temperature)
            teacher_soft = tf.nn.softmax(teacher_logits / self.temperature)
            soft_loss = tf.keras.losses.categorical_crossentropy(teacher_soft, student_soft)
            soft_loss *= (self.temperature ** 2)
            
            # Combined loss
            total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
            return total_loss
        else:
            # Standard loss when no teacher available
            return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)


class TeacherStudentTrainer:
    """Enhanced trainer for knowledge distillation with proper model saving"""
    
    def __init__(self, teacher_model_path=None, temperature=4.0, alpha=0.7):
        self.teacher_model_path = teacher_model_path
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_model = None
        
        if teacher_model_path and os.path.exists(teacher_model_path):
            self._load_teacher_model()
    
    def _load_teacher_model(self):
        """Load the teacher model"""
        try:
            print(f"[INFO] Loading teacher model: {self.teacher_model_path}")
            self.teacher_model = tf.keras.models.load_model(self.teacher_model_path)
            print(f"[INFO] Teacher model loaded successfully")
            print(f"[INFO] Teacher model parameters: {self.teacher_model.count_params():,}")
        except Exception as e:
            print(f"[WARNING] Failed to load teacher model: {e}")
            self.teacher_model = None
    
    def create_distillation_model(self, student_model):
        """Create a combined model for distillation training"""
        if self.teacher_model is None:
            return student_model
        
        # Freeze teacher model
        for layer in self.teacher_model.layers:
            layer.trainable = False
        
        # Create combined model that outputs both student and teacher predictions
        inputs = student_model.input
        student_output = student_model(inputs)
        teacher_output = self.teacher_model(inputs)
        
        # Combined outputs for distillation loss
        combined_output = [student_output, teacher_output]
        
        distillation_model = tf.keras.Model(inputs=inputs, outputs=combined_output)
        
        # Compile with custom loss
        distillation_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
            loss=KnowledgeDistillationLoss(alpha=self.alpha, temperature=self.temperature),
            metrics=['accuracy']
        )
        
        return distillation_model, student_model
    
    def train_with_distillation(self, student_model, train_gen, val_gen, epochs, best_model_path):
        """Train student model with knowledge distillation and proper saving"""
        if self.teacher_model is None:
            print("[INFO] No teacher model available, training with standard loss")
            
            # Setup callbacks for standard training
            callbacks = self._create_callbacks(student_model, best_model_path, monitor='val_accuracy')
            
            return student_model.fit(
                train_gen,
                epochs=epochs,
                validation_data=val_gen,
                callbacks=callbacks,
                verbose=1
            )
        
        print("[INFO] Training with knowledge distillation")
        
        # Create distillation model
        distillation_model, original_student = self.create_distillation_model(student_model)
        
        # Custom callback to save only the student model
        class StudentModelCheckpoint(tf.keras.callbacks.Callback):
            def __init__(self, student_model, filepath, monitor='val_accuracy', 
                        save_best_only=True, verbose=1):
                super().__init__()
                self.student_model = student_model
                self.filepath = filepath
                self.monitor = monitor
                self.save_best_only = save_best_only
                self.verbose = verbose
                self.best = -np.Inf if 'acc' in monitor else np.Inf
                
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                current = logs.get(self.monitor)
                
                if current is None:
                    if self.verbose > 0:
                        print(f'\nEpoch {epoch + 1}: {self.monitor} not available, saving model anyway.')
                    self.student_model.save(self.filepath)
                    return
                
                if 'acc' in self.monitor:
                    # For accuracy, higher is better
                    if current > self.best:
                        self.best = current
                        if self.verbose > 0:
                            print(f'\nEpoch {epoch + 1}: {self.monitor} improved from {self.best:.5f} to {current:.5f}, saving student model to {self.filepath}')
                        self.student_model.save(self.filepath)
                else:
                    # For loss, lower is better
                    if current < self.best:
                        self.best = current
                        if self.verbose > 0:
                            print(f'\nEpoch {epoch + 1}: {self.monitor} improved from {self.best:.5f} to {current:.5f}, saving student model to {self.filepath}')
                        self.student_model.save(self.filepath)
        
        # Setup callbacks for knowledge distillation
        callbacks = [
            StudentModelCheckpoint(
                student_model=original_student,
                filepath=best_model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=False,  # We handle this manually
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train with distillation
        history = distillation_model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Load the best student model
        if os.path.exists(best_model_path):
            print(f"[INFO] Loading best student model from {best_model_path}")
            # Load with custom objects to ensure compatibility
            from rpi_optimized_model import FrameSampler, ResidualBlock, EfficientBlock
            custom_objects = {
                'FrameSampler': FrameSampler,
                'ResidualBlock': ResidualBlock,
                'EfficientBlock': EfficientBlock,
            }
            best_student = tf.keras.models.load_model(best_model_path, custom_objects=custom_objects)
            
            # Replace the original student model weights with the best weights
            student_model.set_weights(best_student.get_weights())
        
        return history
    
    def _create_callbacks(self, model, best_model_path, monitor='val_accuracy'):
        """Create standard callbacks for non-distillation training"""
        return [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=best_model_path,
                monitor=monitor,
                mode='max' if 'acc' in monitor else 'min',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]


def create_representative_dataset(data_gen, num_samples=100):
    """Create representative dataset for quantization"""
    
    def representative_data_gen():
        sample_count = 0
        for i in range(len(data_gen)):
            try:
                batch_x, _ = data_gen[i]
                if len(batch_x) > 0:
                    for sample in batch_x:
                        if sample_count >= num_samples:
                            return
                        yield [np.expand_dims(sample, axis=0).astype(np.float32)]
                        sample_count += 1
            except:
                continue
    
    return representative_data_gen


def train_model(model_type, data_dir, output_dir, input_shape, epochs=50, 
                batch_size=16, target_frames=10, teacher_model_path=None, create_tflite=True):
    """
    Train a single RPi model for binary classification with optional knowledge distillation
    
    Args:
        model_type: Type of model to train ('resnet', 'efficientnet', 'enhanced_lightweight')
        data_dir: Directory containing training data
        output_dir: Directory to save results
        input_shape: Input shape for the model
        epochs: Number of training epochs
        batch_size: Training batch size
        target_frames: Number of frames to process
        teacher_model_path: Path to teacher model for knowledge distillation
        create_tflite: Whether to create TFLite model (default: True)
    
    Returns:
        Dictionary with training results
    """
    
    print(f"\n[INFO] Training {model_type} model for binary classification...")
    print(f"[INFO] Input shape: {input_shape}")
    print(f"[INFO] Target frames: {target_frames}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data generators
    train_gen = DataGenerator(
        data_dir=os.path.join(data_dir, "train"),
        batch_size=batch_size,
        shuffle=True,
        binary_all_leak=True,
        balance_classes=True,
        training=True,
        resize=True,
        target_height=input_shape[1],
        target_width=input_shape[2]
    )
    
    val_gen = DataGenerator(
        data_dir=os.path.join(data_dir, "val"),
        batch_size=batch_size,
        shuffle=False,
        binary_all_leak=True,
        balance_classes=False,
        training=False,
        resize=True,
        target_height=input_shape[1],
        target_width=input_shape[2]
    )
    
    print(f"[INFO] Training batches: {len(train_gen)}")
    print(f"[INFO] Validation batches: {len(val_gen)}")
    
    # Create student model
    student_model = get_rpi_model(
        model_type=model_type,
        input_shape=input_shape,
        target_frames=target_frames,
        num_classes=2
    )
    
    print(f"[INFO] Student model parameters: {student_model.count_params():,}")
    
    # Setup knowledge distillation trainer
    trainer = TeacherStudentTrainer(
        teacher_model_path=teacher_model_path,
        temperature=4.0,
        alpha=0.7
    )
    
    # Setup file paths
    timestamp = datetime.now().strftime("%Y%m%d")
    best_model_path = os.path.join(output_dir, f"rpi_{model_type}_best_{timestamp}.keras")
    final_model_path = os.path.join(output_dir, f"rpi_{model_type}_final_{timestamp}.keras")
    
    # Train model
    print(f"[INFO] Starting training for {epochs} epochs...")
    if trainer.teacher_model is not None:
        print(f"[INFO] Using knowledge distillation with teacher model")
    
    start_time = time.time()
    
    try:
        history = trainer.train_with_distillation(
            student_model=student_model,
            train_gen=train_gen,
            val_gen=val_gen,
            epochs=epochs,
            best_model_path=best_model_path
        )
        
        training_time = time.time() - start_time
        
        # Save final model
        print(f"[INFO] Saving final model to {final_model_path}")
        student_model.save(final_model_path)
        
        # Get final metrics
        if hasattr(history, 'history') and 'val_accuracy' in history.history:
            final_accuracy = max(history.history['val_accuracy'])
        else:
            # Evaluate manually if accuracy not in history
            try:
                final_accuracy = student_model.evaluate(val_gen, verbose=0)[1]
            except:
                final_accuracy = 0.0
        
        print(f"[INFO] Best validation accuracy: {final_accuracy:.4f}")
        
        # Quantize model for RPi deployment (with user choice)
        quantized_path = None
        if create_tflite:
            try:
                print("[INFO] Creating quantized version for RPi deployment...")
                quantized_path = os.path.join(output_dir, f"rpi_{model_type}_quantized_{timestamp}.tflite")
                
                # Use the best model for quantization if it exists
                model_to_quantize = best_model_path if os.path.exists(best_model_path) else final_model_path
                quantized_path = quantize_model(model_to_quantize, quantized_path)
                
                if quantized_path:
                    print(f"[INFO] Quantized model saved: {quantized_path}")
                
            except Exception as e:
                print(f"[ERROR] Quantization failed: {e}")
                import traceback
                traceback.print_exc()
                quantized_path = None
        else:
            print("[INFO] Skipping TFLite model creation (user choice)")
        
        # Save training history
        if hasattr(history, 'history'):
            history_path = os.path.join(output_dir, f"training_history_{timestamp}.json")
            with open(history_path, 'w') as f:
                serializable_history = {}
                for key, values in history.history.items():
                    serializable_history[key] = [float(v) for v in values]
                json.dump(serializable_history, f, indent=2)
        else:
            history_path = None
        
        # Generate training summary
        summary_path = os.path.join(output_dir, f"training_summary_{timestamp}.txt")
        with open(summary_path, 'w') as f:
            f.write(f"RPi Model Training Summary - {timestamp}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Classification: Binary (No Leak vs Leak)\n")
            f.write(f"Input Shape: {input_shape}\n")
            f.write(f"Target Frames: {target_frames}\n")
            f.write(f"Total Parameters: {student_model.count_params():,}\n")
            f.write(f"Training Time: {training_time:.2f} seconds\n")
            f.write(f"Best Validation Accuracy: {final_accuracy:.4f}\n")
            
            if trainer.teacher_model is not None:
                f.write(f"Knowledge Distillation: Enabled\n")
                f.write(f"Teacher Model: {teacher_model_path}\n")
                f.write(f"Teacher Parameters: {trainer.teacher_model.count_params():,}\n")
                f.write(f"Temperature: {trainer.temperature}\n")
                f.write(f"Alpha: {trainer.alpha}\n")
            else:
                f.write(f"Knowledge Distillation: Disabled\n")
            
            f.write(f"\nTraining Configuration:\n")
            f.write(f"  Epochs: {epochs}\n")
            f.write(f"  Batch Size: {batch_size}\n")
            f.write(f"  Training Batches: {len(train_gen)}\n")
            f.write(f"  Validation Batches: {len(val_gen)}\n\n")
            
            f.write("Files Generated:\n")
            f.write(f"  Final Model: {os.path.basename(final_model_path)}\n")
            f.write(f"  Best Model: {os.path.basename(best_model_path)}\n")
            if quantized_path:
                f.write(f"  Quantized Model: {os.path.basename(quantized_path)}\n")
            if history_path:
                f.write(f"  Training History: {os.path.basename(history_path)}\n")
        
        print(f"[INFO] Training summary saved: {summary_path}")
        
        # Return results
        results = {
            'model_type': model_type,
            'classification_type': 'binary',
            'model_path': final_model_path,
            'best_model_path': best_model_path,
            'quantized_path': quantized_path,
            'final_accuracy': float(final_accuracy),
            'training_time': training_time,
            'total_params': student_model.count_params(),
            'output_dir': output_dir,
            'history_path': history_path,
            'summary_path': summary_path,
            'knowledge_distillation': trainer.teacher_model is not None,
            'teacher_model_path': teacher_model_path if trainer.teacher_model is not None else None
        }
        
        print(f"[SUCCESS] Training completed in {training_time:.2f}s")
        print(f"[INFO] Best validation accuracy: {final_accuracy:.4f}")
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Return minimal results in case of failure
        return {
            'model_type': model_type,
            'classification_type': 'binary',
            'model_path': None,
            'best_model_path': None,
            'quantized_path': None,
            'final_accuracy': 0.0,
            'training_time': 0.0,
            'total_params': 0,
            'output_dir': output_dir,
            'history_path': None,
            'summary_path': None,
            'knowledge_distillation': False,
            'teacher_model_path': None
        }
    
    finally:
        # Clean up GPU memory
        tf.keras.backend.clear_session()
        import gc
        gc.collect()


def train_all_models(args, timestamp):
    """Train all requested models with enhanced support for knowledge distillation"""
    
    print("\n" + "="*60)
    print("RPi MODEL TRAINING PHASE - Enhanced with Knowledge Distillation")
    print("="*60)
    print("Training RPi-optimized models for binary classification")
    print("Architectures: ResNet, EfficientNet, Enhanced Lightweight")
    print("No Leak (0) vs Leak (1-7 mapped to 1)")
    print("="*60)
    
    # Ask user about TFLite creation
    create_tflite = True
    if hasattr(args, 'interactive') and args.interactive:
        response = input("\nCreate TFLite models for RPi deployment? (y/n, default: y): ").strip().lower()
        create_tflite = response != 'n'
        
        if create_tflite:
            print("[INFO] TFLite models will be created after training")
        else:
            print("[INFO] TFLite model creation will be skipped")
    
    all_results = {}
    
    for resolution_str in args.resolutions:
        target_height, target_width = map(int, resolution_str.split('x'))
        input_shape = (15, target_height, target_width, 1)
        
        print(f"\n--- Training for resolution {resolution_str} ---")
        
        for model_type in args.model_types:
            # Create output directory
            output_dir = os.path.join(args.result_dir, f"rpi_{model_type}_{resolution_str}")
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                print(f"\n[INFO] Starting {model_type} training...")
                
                # Train model with optional knowledge distillation
                result = train_model(
                    model_type=model_type,
                    data_dir=args.data_dir,
                    output_dir=output_dir,
                    input_shape=input_shape,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    target_frames=args.target_frames,
                    teacher_model_path=args.teacher_model,
                    create_tflite=create_tflite
                )
                
                # Store results
                key = f"{model_type}_{resolution_str}"
                all_results[key] = result
                
                print(f"[SUCCESS] {key} training completed")
                print(f"[INFO] Final accuracy: {result['final_accuracy']:.4f}")
                if result['knowledge_distillation']:
                    print(f"[INFO] Knowledge distillation was used")
                
            except Exception as e:
                print(f"[ERROR] Training failed for {model_type}_{resolution_str}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n[INFO] Training phase completed. Trained {len(all_results)} models.")
    
    # Print summary of results
    if all_results:
        print(f"\n[SUMMARY] Training Results:")
        for model_name, result in all_results.items():
            distillation_status = "with KD" if result['knowledge_distillation'] else "standard"
            print(f"  {model_name}: {result['final_accuracy']:.4f} ({distillation_status})")
    
    return all_results