# LanBLoc: Landmark-Based Localization for GPS-Denied Environments

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LanBLoc** is a visual localization framework that uses stereo vision and deep learning-based landmark recognition for position estimation in GPS-denied environments. The system combines YOLO-based landmark detection with stereo depth estimation and trilateration to achieve accurate 2D localization.

## Table of Contents
- [Overview](#overview)
- [Algorithm Description](#algorithm-description)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)

## Overview

LanBLoc addresses the challenge of localization in environments where GPS is unavailable or unreliable (e.g., urban canyons, indoor spaces, military operations). The framework:

1. **Detects known landmarks** using a fine-tuned YOLOv11 model
2. **Estimates distances** to detected landmarks using stereo depth computation
3. **Computes position** via least-squares trilateration with LBFGS-B refinement

### Key Features
- Real-time landmark detection using YOLOv11
- Robust stereo depth estimation with median filtering
- Least-squares trilateration with iterative refinement
- Support for multiple coordinate systems (lat/lon, XYZ)
- Comprehensive evaluation tools

## Algorithm Description

### Algorithm 1: Metric Depth from Stereo Landmark ROI

Computes the distance to a detected landmark using stereo disparity:

```
Input: Rectified stereo images (I_L, I_R), focal length f, baseline B, ROI
Output: Depth map Z, landmark distance Z_lm

1. Compute disparity map D_raw using stereo matching
2. Convert to depth: Z(x,y) = (f · B) / D(x,y)
3. Extract depth values within ROI
4. Filter valid (finite) depths
5. Return median depth as robust landmark distance
```

### Algorithm 2: Landmark-Based Localization (LanBLoc-3L)

Main localization procedure:

```
Input: Stereo images, calibration, landmark database M, N_req=3
Output: Current 2D position (x, y)

1. Capture and preprocess stereo pair
2. Rectify and undistort images
3. Detect landmarks using YOLOv11
4. For each detected landmark:
   - Look up known coordinates from database
   - Estimate distance using Algorithm 1
5. If sufficient landmarks detected (≥ N_req):
   - Compute initial position via least-squares trilateration
   - Refine using LBFGS-B optimization
6. Return estimated position
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for YOLO inference)
- OpenCV with CUDA support (optional, for faster stereo matching)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/lanbloc.git
cd lanbloc
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install LanBLoc Package

```bash
pip install -e .
```

### Step 5: Download YOLO Weights (Optional)

If using a custom-trained YOLO model for landmark detection:

```bash
# Place your trained weights in the weights directory
mkdir -p weights
# Copy your yolo11_landmarks.pt to weights/
```

## Dataset Structure

LanBLoc uses the **Landmark StereoV1** dataset for validation. The dataset follows this structure:

```
landmark_stereov1/
├── trilat1/
│   ├── landmark1/
│   │   ├── location           # Landmark coordinates (lat, lon) or (x, y, z)
│   │   └── stereo_images/
│   │       ├── node_location  # Ground truth node positions
│   │       ├── stereoL/
│   │       │   ├── 1_img1.png  # Node 1, sequence 1
│   │       │   ├── 1_img2.png  # Node 1, sequence 2
│   │       │   ├── 2_img1.png  # Node 2, sequence 1
│   │       │   └── ...
│   │       └── stereoR/
│   │           ├── 1_img1.png
│   │           ├── 1_img2.png
│   │           └── ...
│   ├── landmark2/
│   ├── landmark3/
│   └── landmark4/
├── trilat2/
│   └── ...
├── ...
└── trilat10/
```

### Image Naming Convention
- Format: `<node_position>_<sequence_number>.png`
- Example: `1_img1.png` in `stereoL/` pairs with `1_img1.png` in `stereoR/`
- The prefix number indicates the node position where the stereo image was captured

### Location Files
- **landmark location**: Contains the known 3D coordinates of the landmark
- **node_location**: Contains ground truth positions for evaluation

## Usage

### Basic Usage

```python
from lanbloc import LanBLoc
from lanbloc.data import LandmarkDatabase

# Initialize landmark database
landmark_db = LandmarkDatabase.from_dataset("datasets/landmark_stereov1_corrupt")

# Initialize LanBLoc
lanbloc = LanBLoc(
    landmark_db=landmark_db,
    yolo_weights="weights/yolo11_landmarks.pt",
    camera_config="config/camera_calibration.yaml"
)

# Run localization
position = lanbloc.localize(left_image, right_image)
print(f"Estimated position: ({position.x:.2f}, {position.y:.2f})")
```

### Command Line Interface

```bash
# Run localization on a single stereo pair
python scripts/run_lanbloc.py \
    --left datasets/landmark_stereov1_corrupt/trilat1/landmark1/stereo_images/stereoL/1_img1.png\
    --right datasets/landmark_stereov1_corrupt/trilat1/landmark1/stereo_images/stereoR/1_img1.png \
    --config config/default_config.yaml

# Run evaluation on entire dataset
python scripts/evaluate.py \
    --dataset data/landmark_stereov1 \
    --config config/default_config.yaml \
    --output results/

# Visualize results
python scripts/visualize_results.py \
    --results results/evaluation.json \
    --output figures/
```

### Processing a Trilateration Set

```python
from lanbloc.data import StereoDataset
from lanbloc import LanBLoc

# Load a trilateration set
dataset = StereoDataset("data/landmark_stereov1/trilat1")

# Initialize localization system
lanbloc = LanBLoc.from_config("config/default_config.yaml")

# Process all node positions
results = []
for node_id, stereo_pairs in dataset.get_node_images():
    for left_img, right_img in stereo_pairs:
        position = lanbloc.localize(left_img, right_img)
        results.append({
            'node_id': node_id,
            'estimated_position': position
        })
```

## Configuration

### Default Configuration (config/default_config.yaml)

```yaml
# Camera calibration
camera:
  focal_length: 1081.8708395231049      # pixels
  baseline: 0.12           # meters
  image_width: 640
  image_height: 480

# Stereo matching parameters
stereo:
  algorithm: "SGBM"
  num_disparities: 128
  block_size: 11
  min_disparity: 0
  disp_scale: 16

# YOLO detection parameters
detection:
  model: "yolo11s"
  weights: "weights/yolo11_landmarks.pt"
  confidence_threshold: 0.5
  iou_threshold: 0.45

# Localization parameters
localization:
  min_landmarks: 3         # N_req
  min_valid_depth: 100     # N_min for depth validation
  optimization_method: "LBFGS-B"
  max_iterations: 1000
```

## Evaluation

### Metrics
- **RMSE**: Root Mean Square Error of position estimates
- **MAE**: Mean Absolute Error
- **Success Rate**: Percentage of successful localizations

### Running Evaluation

```bash
python scripts/evaluate.py \
    --dataset data/landmark_stereov1 \
    --ground-truth data/ground_truth.json \
    --output results/evaluation.json
```

### Example Results

| Trilateration Set | RMSE (m) | MAE (m) | Success Rate |
|-------------------|----------|---------|--------------|
| trilat1           | 0.45     | 0.38    | 100%         |
| trilat2           | 0.52     | 0.44    | 100%         |
| ...               | ...      | ...     | ...          |

## Project Structure

```
lanbloc/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
├── config/
│   └── default_config.yaml   # Default configuration
├── lanbloc/
│   ├── __init__.py
│   ├── core/
│   │   ├── stereo_depth.py   # Algorithm 1 implementation
│   │   ├── localization.py   # Algorithm 2 implementation
│   │   └── trilateration.py  # Trilateration solvers
│   ├── detection/
│   │   └── yolo_detector.py  # YOLO landmark detection
│   ├── calibration/
│   │   └── camera.py         # Camera calibration utilities
│   ├── utils/
│   │   ├── coordinates.py    # Coordinate transformations
│   │   └── visualization.py  # Plotting utilities
│   └── data/
│       ├── dataset.py        # Dataset loading
│       └── landmark_db.py    # Landmark database
├── scripts/
│   ├── run_lanbloc.py        # Main execution script
│   ├── evaluate.py           # Evaluation script
│   └── visualize_results.py  # Visualization script
├── tests/
│   └── test_lanbloc.py       # Unit tests
└── data/
    └── landmark_stereov1/    # Dataset directory
```

## Citation

If you use LanBLoc in your research, please cite:

```bibtex
@INPROCEEDINGS{10579240,
  author={Sapkota, Ganesh and Madria, Sanjay},
  booktitle={2024 IEEE 25th International Symposium on a World of Wireless, Mobile and Multimedia Networks (WoWMoM)}, 
  title={Landmark-based Localization using Stereo Vision and Deep Learning in GPS-Denied Battlefield Environment}, 
  year={2024},
  pages={209-215},
  keywords={Location awareness;Deep learning;Image recognition;Navigation;
  Wireless networks;Optimization methods;Predictive models;
  Landmark Recognition;YOLOv8;Stereo Vision;Non-GPS localization;
  DV-Hop Method;Battlefield Navigation},
  doi={10.1109/WoWMoM60985.2024.00043}}
  
```
```bibtex
@INPROCEEDINGS {10440690,
author = { Sapkota, Ganesh and Madria, Sanjay },
booktitle = { 2023 IEEE Applied Imagery Pattern Recognition Workshop (AIPR) },
title = {{ Landmark Stereo Dataset for Landmark Recognition and Moving Node Localization in a Non-GPS Battlefield Environment }},
year = {2023},
pages = {1-11},
keywords = {YOLO;Navigation;Optimization methods;
Mobile handsets;Pattern recognition;
Stereo vision;Servers},
doi = {10.1109/AIPR60534.2023.10440690},
url = {https://doi.ieeecomputersociety.org/10.1109/AIPR60534.2023.10440690},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month =sep}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLOv11 by Ultralytics
- OpenCV for stereo vision processing
- NumPy and SciPy for numerical computations

## Contact

For questions or issues, please open an issue on GitHub or contact [gsapkota@mst.edu].
