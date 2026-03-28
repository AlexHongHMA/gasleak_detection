# Optimised 3D-CNN for Real-Time Infrared Natural Gas Leak Classification - Balancing Accuracy and Computational Cost

[![IEEE](https://img.shields.io/badge/IEEE-10.1109/ICoICT66265.2025.11192973-00629B.svg)](https://doi.org/10.1109/ICoICT66265.2025.11192973)
[![Best Paper](https://img.shields.io/badge/ICoICT%202025-Best%20Paper%20Award-FFD700.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://www.python.org/)
[![TensorFlow 2.18](https://img.shields.io/badge/TensorFlow-2.18-FF6F00.svg)](https://www.tensorflow.org/)

This repository contains the source code for the paper:

> H. M. Aung and S. Wongsa, "Optimised 3D-CNN for Real-Time Infrared Natural Gas Leak Classification: Balancing Accuracy and Computational Cost," *2025 International Conference on Information and Communication Technology (ICoICT)*, 2025.
>
> **DOI:** [10.1109/ICoICT66265.2025.11192973](https://doi.org/10.1109/ICoICT66265.2025.11192973)
>
> **Best Paper Award** at ICoICT 2025

## Overview

Infrared imaging is vital for real-time methane leak detection in industrial environments, yet the computational complexity of 3D Convolutional Neural Networks (3D-CNNs) like VideoGasNet limits their deployment on resource-constrained systems.

This work proposes an optimised preprocessing pipeline for VideoGasNet that:

- **Replaces** the computationally intensive moving median background subtraction (210-frame window) with a **running average** approach
- **Incorporates image downscaling** at half (120x160) and quarter (60x80) resolutions
- **Integrates optical flow** analysis (Farneback's algorithm) to reduce false positives from environmental motion (windsock, clouds)

All code is built from scratch using clues from the reference papers below. Our proposed 3D-CNN hyperparameters differ from the original VideoGasNet model.

## Key Results

| Background Subtraction Method | Overall Accuracy | Processing Time/Frame |
|-------------------------------|:----------------:|:---------------------:|
| Running Average (ours)        | **99.71%**       | **35.88 ms**          |
| Moving Average (original)     | 99.58%           | 107.39 ms             |
| Custom GMM                    | 99.65%           | 41.26 ms              |

- Running Average achieves **~3x faster** processing than the original Moving Average method
- **Zero false positives** with the Running Average at full resolution
- At half resolution (120x160), accuracy remains **99.27%** with **40% smaller** model size
- At quarter resolution (60x80), accuracy is still **above 98%** with up to **66% reduction** in preprocessing time

| Resolution | Parameters       | Model Size | Overall Accuracy |
|------------|:----------------:|:----------:|:----------------:|
| 240x320    | 413,386 (1.58 MB)| 4934 KB    | 99.71%           |
| 120x160    | 249,546 (975 KB) | 3014 KB    | 99.27%           |
| 60x80      | 188,106 (735 KB) | 2294 KB    | 98.14%           |

For full details, please refer to the [paper](https://doi.org/10.1109/ICoICT66265.2025.11192973).

## References

1. W. M. Haynes, D. R. Lide, and T. J. Bruno, Eds., *CRC Handbook of Chemistry and Physics: A Ready-Reference Book of Chemical and Physical Data*, 97th ed. Boca Raton, FL: CRC Press, 2016, sections 16-26: Flammability of Chemical Substances.

2. A. P. Ravikumar, J. Wang, and A. R. Brandt, "Are optical gas imaging technologies effective for methane leak detection?" *Environmental Science & Technology*, vol. 51, p. 718, 2016.

3. J. Wang, L. P. Tchapmi, A. P. Ravikumar, M. McGuire, C. S. Bell, D. Zimmerle, S. Savarese, and A. R. Brandt, "Machine vision for natural gas methane emissions detection using an infrared camera," *Applied Energy*, vol. 257, p. 113998, 2020. DOI: [10.1016/j.apenergy.2019.113998](https://doi.org/10.1016/j.apenergy.2019.113998)

4. J. Wang, J. Ji, A. P. Ravikumar, S. Savarese, and A. R. Brandt, "VideoGasNet: Deep learning for natural gas methane leak classification using an infrared camera," *Energy*, vol. 238, p. 121516, 2022. DOI: [10.1016/j.energy.2021.121516](https://doi.org/10.1016/j.energy.2021.121516)

5. J. Wang, Y. Lin, Q. Zhao, D. Luo, S. Chen, W. Chen, and X. Peng, "Invisible gas detection: An rgb-thermal cross attention network and a new benchmark," *Computer Vision and Image Understanding*, vol. 248, p. 104099, 2024.

6. Z. Yi and F. Liangzhong, "Moving object detection based on running average background and temporal difference," in *2010 IEEE International Conference on Intelligent Systems and Knowledge Engineering*, 2010, pp. 270-272.

7. G. Farneback, "Two-frame motion estimation based on polynomial expansion," in *Image Analysis*, J. Bigun and T. Gustavsson, Eds. Berlin, Heidelberg: Springer Berlin Heidelberg, 2003, pp. 363-370.

8. Z. Zivkovic, "Improved adaptive Gaussian mixture model for background subtraction," in *Proceedings of the 17th International Conference on Pattern Recognition (ICPR)*, vol. 2. IEEE, 2004, pp. 28-31.

9. Z. Zivkovic and F. Van der Heijden, "Efficient adaptive density estimation per image pixel for the task of background subtraction," *Pattern Recognition Letters*, vol. 27, pp. 773-780, 2006.

Our contribution focuses on binary classification (leak vs. no-leak) using the VideoGasNet data and a modified 3D-CNN architecture. Three-class and eight-class classification modes are under development.

## Project Structure

```
gasleak_detection/
├── data/raw/                            # Raw Data from GasVid Video Data
├── main.py                              # Main entry point (preprocessing, training, testing)
├── pyproject.toml                       # Project dependencies (uv/pip)
├── src/
│   ├── augmentation/
│   │   └── video_augment.py             # Data augmentation (flip, rotation)
│   ├── loader/
│   │   └── loader.py                    # Data generators for training/testing
│   ├── model/
│   │   └── model.py                     # 3D-CNN model architecture
│   ├── preprocess/
│   │   ├── preprocess.py                # Moving Average background subtraction
│   │   ├── runavg_preprocess.py         # Running Average background subtraction (proposed)
│   │   └── mog2_preprocess.py           # Custom Gaussian Mixture Model
│   ├── test/
│   │   └── test.py                      # Model evaluation and metrics
│   └── train/
│       └── train.py                     # Model training pipeline
└── resources/
    └── GasVid_Logging_File.xlsx         # Video metadata
```

## 3D-CNN Architecture

The model consists of four convolutional-pooling blocks followed by fully connected layers:

```
Input: 15 x H x W x 1 (15 frames, grayscale)
  → Conv3D(4, 5x5x5) → LayerNorm → MaxPool3D(3x3x3) → Dropout(0.1)
  → Conv3D(8, 5x5x5) → LayerNorm → MaxPool3D(3x3x3) → Dropout(0.1)
  → Conv3D(16, 5x5x5) → LayerNorm → MaxPool3D(3x3x3) → Dropout(0.1)
  → Conv3D(32, 5x5x5) → LayerNorm → MaxPool3D(3x3x3) → Dropout(0.1)
  → Flatten
  → Dense(640) → Dropout(0.3)
  → Dense(128) → Dropout(0.3)
  → Dense(num_classes, softmax)
```

Trained with Adam optimiser and sparse categorical cross-entropy loss.

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- NVIDIA GPU with CUDA support (recommended)
- GasVid dataset (place raw `.mp4` videos in `./data/raw/`)

### Installation

```bash
# Clone the repository
git clone https://github.com/AlexHongHMA/gasleak_detection.git
cd gasleak_detection

# Install dependencies with uv
uv sync
```

### Usage

All commands are run through `main.py` via `uv run`.

#### Full Pipeline (preprocess + train + test)

```bash
# Run with Running Average (default, recommended)
uv run python main.py --methods runavg --resolutions 240x320 --epochs 50 --batch-size 16

# Run with Moving Average
uv run python main.py --methods ma --resolutions 240x320 --epochs 50 --batch-size 16

# Run with Custom GMM
uv run python main.py --methods mog2 --resolutions 240x320 --epochs 50 --batch-size 16
```

#### Skip Specific Steps

```bash
# Skip preprocessing (use already processed data)
uv run python main.py --methods runavg --skip-preprocessing --resolutions 240x320

# Skip training (test with existing model)
uv run python main.py --methods runavg --skip-training --resolutions 240x320

# Skip testing
uv run python main.py --methods runavg --skip-testing --resolutions 240x320
```

#### Multiple Resolutions

```bash
# Train and test at all three resolutions
uv run python main.py --methods runavg --resolutions 240x320 120x160 60x80 --epochs 50
```

#### Three-Class Classification (under development)

```bash
uv run python main.py --methods runavg --classification-mode three_class --resolutions 240x320
```

#### Distance Filtering

```bash
# Use only 4.6m imaging distance data
uv run python main.py --methods runavg --distance-filter 46

# Use only 6.9m imaging distance data
uv run python main.py --methods runavg --distance-filter 69
```

#### All Available Arguments

| Argument                | Default     | Options                                | Description                              |
|-------------------------|-------------|----------------------------------------|------------------------------------------|
| `--methods`             | `runavg`    | `ma`, `mog2`, `runavg`                 | Background subtraction method(s)         |
| `--resolutions`         | `240x320`   | `240x320`, `120x160`, `60x80`          | Input video frame resolution(s)          |
| `--epochs`              | `50`        | Any integer                            | Number of training epochs                |
| `--batch-size`          | `16`        | Any integer                            | Batch size for training/testing          |
| `--data-dir`            | `./data`    | Any path                               | Base directory for data                  |
| `--result-dir`          | `./result`  | Any path                               | Base directory for results               |
| `--classification-mode` | `binary`    | `binary`, `three_class`                | Classification mode                      |
| `--distance-filter`     | `all`       | `46`, `69`, `all`                      | Filter by imaging distance               |
| `--skip-preprocessing`  | `false`     | Flag                                   | Skip video preprocessing                 |
| `--skip-training`       | `false`     | Flag                                   | Skip model training                      |
| `--skip-testing`        | `false`     | Flag                                   | Skip model testing                       |

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{aung2025optimised3dcnn,
  author    = {Aung, Htet Myat and Wongsa, Sarawan},
  title     = {Optimised 3D-CNN for Real-Time Infrared Natural Gas Leak Classification: Balancing Accuracy and Computational Cost},
  booktitle = {2025 International Conference on Information and Communication Technology (ICoICT)},
  year      = {2025},
  doi       = {10.1109/ICoICT66265.2025.11192973},
  publisher = {IEEE}
}
```

## Acknowledgements

- **Prof. Adam Brandt**, Precourt Institute for Energy, Stanford University — for providing the GasVid dataset
- **Mrs. Phyu Phyu Aung**, General Manager at No. 1 Refinery, Thanlyin Myanmar Petrochemical Enterprise — for expertise on natural gas properties
- **Prof. Dr. Wuttipong Kumwilaisak** — for initiating this research direction

## License

This project is licensed under the [MIT License](LICENSE).

If you have any questions, feel free to send an email to htetmyataung.ctla@gmail.com. 
