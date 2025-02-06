import os
from src.preprocess.preprocess import process_video
from src.train.train import train_binary
from src.test.test import test_binary
import tensorflow as tf

def main():
    # Gather all .mp4 files from ./data/raw/
    video_dir = './data/raw/'
    video_paths = [
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.endswith('.mp4')
    ]
    
    xlss_path = "/mnt/d/GasVid_Dataset/gasleak_detection/resources/GasVid_Logging_File.xlsx"
    output_dir = "./data/processed"

    # # Loop over each video path
    # for single_video_path in video_paths:
    #     process_video(single_video_path, xlss_path, output_dir)

        # Paths
    train_dir = "./data/processed/train"
    val_dir   = "./data/processed/val"
    test_dir = "./data/processed/test"
    model_save_path = "./result/binary/cnn_3d_binaryAllLeak.keras"
    batch_size = 32
    epochs = 100
    best_model_path = "./result/best_model/cnn_3d_binaryAllLeak.keras"
    # train_binary(train_dir, val_dir, model_save_path, best_model_path, batch_size, epochs)
    test_binary(test_dir, model_save_path, batch_size)

if __name__ == "__main__":
    # Disable XLA and mixed precision
    tf.config.optimizer.set_jit(False)
    tf.keras.mixed_precision.set_global_policy('float32')

    # Force GPU memory growth to prevent OOM errors
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    main()
