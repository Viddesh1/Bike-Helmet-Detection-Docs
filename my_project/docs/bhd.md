# Bike Rider Helmet Detection using YOLOv8

This repository contains a walk through of implementation of bike rider helmet detection using YOLOv8. As we know that bike riders who do not wear helmet may which result in fatal accidents and death in some cases. The goal is to create a ML/DL model that can detect if a person is wearing helment or not. May sure to select GPU in google colab or kaggle notebook for running through the notebook. As running through the code in local machine with no graphical processing unit will take a very significant ammount of time. <br />

### Methodology
1) Data Collection:- Collect images/videos of bike rider, helmet and no helmet for the model to train upon. <br />
2) Data pre-processing:- Pre-processing and autolabel images and videos using foundation model like DINO and SAM (Segment anything model).  <br />
3) Train YOLOv8 Model. <br />
4) Evaluate Target Model.   <br />
5) Run Inference on images and videos.  <br />

# How to setup codebase locally?
```shell
python3 -m venv .venv   # Create a virtual environment
source .venv/bin/activate   # Activate a virtual environment
git clone https://github.com/Viddesh1/Helmet_test_1.git
cd Helmet_test_1/
```

# Repository File Structure
```text
Helmet_test_1   # This repository root folder
├── .git        # For managing files by git
├── .gitignore  # Files to be not managed by git
├── Helmet_how_to_auto_train_yolov8_model_with_autodistill.ipynb    # For reference
├── kaggle_Helmet_how_to_auto_train_yolov8_model_with_autodistill.ipynb # YOLOv8 implementation
├── README.md   # This README.md file itself
└── tree_all.txt  # This file structure from tree -a > tree_all.txt command
```
# Dataset Information
For this project only images and videos are needed and no annotations files are needed. As for this project auto labelling is done by using the Ultralytics framework which uses base model naming Grounded SAM (Segment Anything Model) which is a combination of Grounding DINO (Deeper Into Neural Networks) + SAM (Segment Anything model). The target model is YOLOv8.

The datset that is used for this project can be find below:- <br />

https://www.kaggle.com/datasets/andrewmvd/helmet-detection  <br />

This dataset contains 764 images of 2 distinct classes for the objective of helmet detection.
Bounding box annotations are provided in the PASCAL VOC format. <br />
The classes are:

 - With helmet
 - Without helmet

Please generate your own kaggle api key for accessing the dataset with in google colab or kaggle notebook <br />

Please take a look at the below python code representing labels for individual ontology.
<br  />

```python
from autodistill.detection import CaptionOntology
    
    # "<description of label>": "<label_name>"
    # "bike rider": "Bike_Rider", --> label 0
    # "bike rider and passanger with helmet": "Helmet", --> label 1
    # "bike rider and passanger with no helmet": "No_Helmet" --> label 2

ontology=CaptionOntology({
    "bike rider": "Bike_Rider",
    "helmet": "Helmet",
    "no helmet": "No_Helmet"
})
```

# Dataset File Structure
```text
archive     # Root file directory of https://www.kaggle.com/datasets/andrewmvd/helmet-detection
├── annotations     # Annotations based on PASCAL VOC format as XML files
│   ├── BikesHelmets0.xml
│   ├── BikesHelmets100.xml
│   ├── BikesHelmets101.xml
├── images          # Public images for classes helmet and without helmet images
│   ├── BikesHelmets0.png
│   ├── BikesHelmets100.png
│   ├── BikesHelmets101.png
└── tree_all.txt

2 directories, 1529 files
```

## This repository output may change in the near future:-

https://drive.google.com/drive/folders/1M4FckJJeyPQTTWqgo6xWhW8L4tf0EJ4l?usp=sharing


## Output File Structure for YOLOv8
```text
YOLOv8_Helmet_V0    # Main directory of YOLOv8 output
├── dataset
│   ├── annotations # Empty file
│   ├── data.yaml   # Information related to data detection labels and train, valid path
│   ├── images      # Empty folder
│   ├── train       # Train dataset
│   │   ├── images  # Images directory path
│   │   │   ├── BikesHelmets0.jpg
│   │   │   ├── BikesHelmets100.jpg
│   │   │   ├── BikesHelmets101.jpg
│   │   ├── labels  # Annotations directory path 
│   │   │   ├── BikesHelmets0.txt
│   │   │   ├── BikesHelmets100.txt
│   │   │   ├── BikesHelmets101.txt
│   │   └── labels.cache
│   └── valid # Validation dataset for testing trained model
│       ├── images  # Validation dataset images
│       │   ├── BikesHelmets103.jpg
│       │   ├── BikesHelmets108.jpg
│       │   ├── BikesHelmets119.jpg
│       ├── labels  # Validation dataset labels
│       │   ├── BikesHelmets103.txt
│       │   ├── BikesHelmets108.txt
│       │   ├── BikesHelmets119.txt
│       └── labels.cache
├── images # All 764 images form the dataset
│   ├── BikesHelmets0.png
│   ├── BikesHelmets100.png
│   ├── BikesHelmets101.png
├── runs    # Predictions 
│   └── detect
│       ├── predict
│       │   └── he2.mp4 # Inference upon video 1
│       ├── predict2
│       │   └── test_1.mp4  # Inference upon video 2
│       ├── predict3
│       │   └── test_2.mp4  # Inference upon video 3
│       └── train
│           ├── args.yaml   # Configuration blue print for training the YOLOv8 model parameters 
│           ├── confusion_matrix.png
│           ├── events.out.tfevents.1697046331.428f98cba7b3.163.0
│           ├── F1_curve.png
│           ├── labels_correlogram.jpg
│           ├── labels.jpg
│           ├── P_curve.png
│           ├── PR_curve.png
│           ├── R_curve.png
│           ├── results.csv # All the metrics for train_loss, class_loss, lr etc
│           ├── results.png # Generated from results.csv
│           ├── train_batch0.jpg
│           ├── train_batch1.jpg
│           ├── train_batch2.jpg
│           ├── val_batch0_labels.jpg
│           ├── val_batch0_pred.jpg
│           ├── val_batch1_labels.jpg
│           ├── val_batch1_pred.jpg
│           ├── val_batch2_labels.jpg
│           ├── val_batch2_pred.jpg
│           └── weights     # Model weights after training on custom dataset
│               ├── best.pt # Best model as pytorch format
│               └── last.pt # Last model as pytorch format
├── tree_all.txt    # Generated by tree -a > tree_all.txt
├── videos          # Videos to be uploaded for preprocessing
│   ├── he2.mp4
│   ├── test_1.mp4
│   └── test_2.mp4
├── yolov8l.pt  # Default YOLOV8 large model
└── yolov8n.pt  # Default YOLOv8 nano model
```
# External packages needed for this project
```shell
!pip install -q kaggle  # For accessing dataset locally in google colab

!pip install -U ultralytics

!pip install -q \       # install require packages in quite mode
autodistill \           # Automates model distillation
autodistill-grounded-sam \ # Enhanced distillation with grounding and self-attention mechanisms.
autodistill-yolov8 \    # distilling YOLOv8 models
supervision==0.9.0      # For supervising models
```

#  Train target model - YOLOv8
```python

%cd {HOME}

from autodistill_yolov8 import YOLOv8

target_model = YOLOv8("yolov8l.pt")
target_model.train(DATA_YAML_PATH, epochs=50) #100
```

### Relevant Information of target model YOLOv8

1. **Model Download**:
   - **File**: `yolov8l.pt`
   - **Source**: Ultralytics GitHub repository :- (https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt)
   - **Description**: Downloads the pre-trained YOLOv8 large model, which is approximately 83.7 MB in size. The model is used for object detection tasks.

2. **YOLO library**:
   - **Current Version**: `8.0.81`
   - **New Version Available**: `8.0.196`
   - **Update Command**: `pip install -U ultralytics`
   - **Description**: Suggests updating the Ultralytics library to the latest version for improved features and performance.

3. **Environment Details**:
   - **Python Version**: `3.10.12`
   - **PyTorch Version**: `2.0.1+cu118`
   - **CUDA Version**: `0` (CUDA enabled device: Tesla T4 with 15102 MiB memory)
   - **Description**: Specifies the software and hardware environment used for running the YOLOv8 model.

4. **Training Configuration**:
   - **Task**: `detect`
   - **Mode**: `train`
   - **Model**: `yolov8l.pt`
   - **Data**: `/content/dataset/data.yaml`
   - **Epochs**: `50`
   - **Patience**: `50`
   - **Batch Size**: `16`
   - **Image Size**: `640`
   - **Optimizer**: `SGD`
   - **Other Parameters**: Various other hyperparameters and settings for training the model.
   - **Description**: Outlines the configuration for training the YOLOv8 model using the specified dataset, model, and hyperparameters.

5. **Font Download**:
   - **File**: `Arial.ttf`
   - **Source**: Ultralytics website (https://ultralytics.com/assets/Arial.ttf)
   - **Location**: `/root/.config/Ultralytics/Arial.ttf`
   - **Description**: Downloads a font file required for visualizations and plots generated during the training process.

6. **Model Configuration Override**:
   - **Original `nc`**: `80`
   - **New `nc`**: `3`
   - **Description**: Overrides the number of classes (`nc`) in the model configuration to 3, as specified in the dataset configuration (`data.yaml`).

7. **Optimizer**
- **Type**: SGD
- **Learning Rate**: 0.01
- **Parameter Groups**:
  - Group 1: 
    - Weight Decay: 0.0
    - Number of Parameters: 97
  - Group 2:
    - Weight Decay: 0.0005
    - Number of Parameters: 104
  - Group 3:
    - Bias: 103

8. **Augmentations**:
  - Blur:
    - Probability: 0.01
    - Blur Limit: (3, 7)
  - MedianBlur:
    - Probability: 0.01
    - Blur Limit: (3, 7)
  - ToGray:
    - Probability: 0.01
  - CLAHE (Contrast Limited Adaptive Histogram Equalization):
    - Probability: 0.01
    - Clip Limit: (1, 4.0)
    - Tile Grid Size: (8, 8)

### Model Architecture Overview

This table details the layers and parameters of the YOLOv8 model architecture after overriding the number of classes from 80 to 3. Each row represents a layer in the model, indicating the type of layer, its parameters, and specific arguments. The model is built using various modules such as convolutional layers (`Conv`), concatenation layers (`Concat`), and the `Detect` module at the end, which specifies the number of classes and their respective parameters.

The following table summarizes the architecture of the YOLOv8 model, with the number of classes (`nc`) overridden from 80 to 3. The table details the layers, parameters, and modules used in the model.

| Layer | From | Num | Params  | Module                               | Arguments              |
|-------|------|-----|---------|--------------------------------------|------------------------|
| 0     | -1   | 1   | 1,856   | ultralytics.nn.modules.Conv          | [3, 64, 3, 2]          |
| 1     | -1   | 1   | 73,984  | ultralytics.nn.modules.Conv          | [64, 128, 3, 2]        |
| 2     | -1   | 3   | 279,808 | ultralytics.nn.modules.C2f           | [128, 128, 3, True]    |
| 3     | -1   | 1   | 295,424 | ultralytics.nn.modules.Conv          | [128, 256, 3, 2]       |
| 4     | -1   | 6   | 2,101,248 | ultralytics.nn.modules.C2f         | [256, 256, 6, True]    |
| 5     | -1   | 1   | 1,180,672 | ultralytics.nn.modules.Conv        | [256, 512, 3, 2]       |
| 6     | -1   | 6   | 8,396,800 | ultralytics.nn.modules.C2f         | [512, 512, 6, True]    |
| 7     | -1   | 1   | 2,360,320 | ultralytics.nn.modules.Conv        | [512, 512, 3, 2]       |
| 8     | -1   | 3   | 4,461,568 | ultralytics.nn.modules.C2f         | [512, 512, 3, True]    |
| 9     | -1   | 1   | 656,896   | ultralytics.nn.modules.SPPF        | [512, 512, 5]          |
| 10    | -1   | 1   | 0         | torch.nn.modules.upsampling.Upsample | [None, 2, 'nearest'] |
| 11    | [-1, 6] | 1 | 0       | ultralytics.nn.modules.Concat       | [1]                    |
| 12    | -1   | 3   | 4,723,712 | ultralytics.nn.modules.C2f         | [1024, 512, 3]         |
| 13    | -1   | 1   | 0         | torch.nn.modules.upsampling.Upsample | [None, 2, 'nearest'] |
| 14    | [-1, 4] | 1 | 0       | ultralytics.nn.modules.Concat       | [1]                    |
| 15    | -1   | 3   | 1,247,744 | ultralytics.nn.modules.C2f         | [768, 256, 3]          |
| 16    | -1   | 1   | 590,336   | ultralytics.nn.modules.Conv        | [256, 256, 3, 2]       |
| 17    | [-1, 12] | 1 | 0      | ultralytics.nn.modules.Concat       | [1]                    |
| 18    | -1   | 3   | 4,592,640 | ultralytics.nn.modules.C2f         | [768, 512, 3]          |
| 19    | -1   | 1   | 2,360,320 | ultralytics.nn.modules.Conv        | [512, 512, 3, 2]       |
| 20    | [-1, 9] | 1 | 0       | ultralytics.nn.modules.Concat       | [1]                    |
| 21    | -1   | 3   | 4,723,712 | ultralytics.nn.modules.C2f         | [1024, 512, 3]         |
| 22    | [15, 18, 21] | 1 | 5,585,113 | ultralytics.nn.modules.Detect | [3, [256, 512, 512]]   |

**Model Summary**:
- **Total Layers**: 365
- **Total Parameters**: 43,632,153
- **Total Gradients**: 43,632,137
- **GFLOPs**: 165.4

**After Training model**
```text
50 epochs completed in 0.590 hours.
Optimizer stripped from runs/detect/train/weights/last.pt, 87.6MB
Optimizer stripped from runs/detect/train/weights/best.pt, 87.6MB

Validating runs/detect/train/weights/best.pt...
Ultralytics YOLOv8.0.81 🚀 Python-3.10.12 torch-2.0.1+cu118 CUDA:0 (Tesla T4, 15102MiB)
Model summary (fused): 268 layers, 43608921 parameters, 0 gradients, 164.8 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 5/5 [00:07<00:00,  1.55s/it]
                   all        153        776      0.783      0.693      0.758      0.612
            Bike_Rider        153        265       0.75      0.668      0.732      0.622
                Helmet        153        336      0.836      0.776      0.816      0.592
             No_Helmet        153        175      0.761      0.636      0.726      0.621
Speed: 1.4ms preprocess, 14.6ms inference, 0.0ms loss, 1.6ms postprocess per image
```
  
# Dataset Configuration (data.yaml)

## Key Parameters and Their Meaning

- **names**: List of class names in the dataset.
  - `Bike_Rider`: Represents images containing bike riders.
  - `Helmet`: Represents images containing helmets.
  - `No_Helmet`: Represents images containing individuals without helmets.
- **nc**: Number of classes in the dataset (3).
- **train**: Path to the directory containing training images (`/content/dataset/train/images`).
- **val**: Path to the directory containing validation images (`/content/dataset/valid/images`).

# YOLOv8 Training Configuration (args.yaml)

## Key Parameters and Their Meaning

- **task**: The type of task being performed (`detect`).
- **mode**: The mode of operation (`train` for training the model).
- **model**: The path to the pre-trained model weights (`yolov8l.pt`).
- **data**: Path to the dataset configuration file (`/content/dataset/data.yaml`).
- **epochs**: Number of training epochs (50).
- **patience**: Number of epochs to wait for improvement before stopping (50).
- **batch**: Batch size (16).
- **imgsz**: Image size for training (640).
- **save**: Whether to save the training results (true).
- **save_period**: Frequency of saving checkpoints (-1 means only save the last).
- **cache**: Whether to cache images (false).
- **device**: Compute device to use (null for automatic selection).
- **workers**: Number of data loader workers (8).
- **project****: Project directory (null for default).
- **name**: Experiment name (null for auto-naming).
- **exist_ok**: Whether to overwrite existing experiment directory (false).
- **pretrained**: Whether to use a pretrained model (false).
- **optimizer**: Optimizer type (`SGD`).
- **verbose**: Verbose output during training (true).
- **seed**: Random seed for reproducibility (0).
- **deterministic**: Ensure deterministic behavior (true).
- **single_cls**: Treat the dataset as a single class (false).
- **image_weights**: Use weighted image sampling (false).
- **rect**: Rectangular training (false).
- **cos_lr**: Use cosine learning rate scheduler (false).
- **close_mosaic**: Close mosaic augmentation (0).
- **resume**: Resume training from last checkpoint (false).
- **amp**: Use automatic mixed precision (true).
- **overlap_mask**: Use overlap masks (true).
- **mask_ratio**: Mask ratio (4).
- **dropout**: Dropout rate (0.0).
- **val**: Whether to validate during training (true).
- **split**: Data split for validation (`val`).
- **save_json**: Save results to JSON (false).
- **save_hybrid**: Save hybrid results (false).
- **conf**: Confidence threshold (null).
- **iou**: Intersection over Union threshold (0.7).
- **max_det**: Maximum detections per image (300).
- **half**: Use half precision (false).
- **dnn**: Use OpenCV DNN module (false).
- **plots**: Generate plots (true).
- **source**: Source of the dataset (null).
- **show**: Show results (false).
- **save_txt**: Save results in TXT format (false).
- **save_conf**: Save confidence scores (false).
- **save_crop**: Save cropped images (false).
- **show_labels**: Show labels on images (true).
- **show_conf**: Show confidence scores on images (true).
- **vid_stride**: Video frame stride (1).
- **line_thickness**: Line thickness for bounding boxes (3).
- **visualize**: Visualize feature maps (false).
- **augment**: Augment data (false).
- **agnostic_nms**: Class-agnostic non-max suppression (false).
- **classes**: Filter by class (null).
- **retina_masks**: Use high-resolution masks (false).
- **boxes**: Use bounding boxes (true).
- **format**: Export format (`torchscript`).
- **keras**: Export to Keras format (false).
- **optimize**: Optimize the model (false).
- **int8**: Quantize model to int8 (false).
- **dynamic**: Use dynamic shapes (false).
- **simplify**: Simplify the model (false).
- **opset**: ONNX opset version (null).
- **workspace**: Workspace size for ONNX export (4).
- **nms**: Use non-max suppression (false).
- **lr0**: Initial learning rate (0.01).
- **lrf**: Final learning rate (0.01).
- **momentum**: Momentum for optimizer (0.937).
- **weight_decay**: Weight decay (0.0005).
- **warmup_epochs**: Warmup epochs (3.0).
- **warmup_momentum**: Warmup momentum (0.8).
- **warmup_bias_lr**: Warmup bias learning rate (0.1).
- **box**: Box loss gain (7.5).
- **cls**: Class loss gain (0.5).
- **dfl**: DFL loss gain (1.5).
- **pose**: Pose loss gain (12.0).
- **kobj**: Keypoint objectness gain (1.0).
- **label_smoothing**: Label smoothing (0.0).
- **nbs**: Nominal batch size (64).
- **hsv_h**: HSV-Hue augmentation (0.015).
- **hsv_s**: HSV-Saturation augmentation (0.7).
- **hsv_v**: HSV-Value augmentation (0.4).
- **degrees**: Degree of rotation for augmentation (0.0).
- **translate**: Translation for augmentation (0.1).
- **scale**: Scaling for augmentation (0.5).
- **shear**: Shear for augmentation (0.0).
- **perspective**: Perspective for augmentation (0.0).
- **flipud**: Vertical flip probability (0.0).
- **fliplr**: Horizontal flip probability (0.5).
- **mosaic**: Mosaic augmentation (1.0).
- **mixup**: Mixup augmentation (0.0).
- **copy_paste**: Copy-paste augmentation (0.0).
- **cfg**: Configuration file (null).
- **v5loader**: Use YOLOv5 data loader (false).
- **tracker**: Tracker configuration (`botsort.yaml`).
- **save_dir**: Directory to save results (`runs/detect/train`).


# Also see
1) https://github.com/Viddesh1/Helmet_test_1    <br />
2) https://github.com/Viddesh1/Bike-Helmet-Detection    <br />
3) https://github.com/Viddesh1/Bike-Helmet-Detectionv2  <br />
4) https://github.com/Viddesh1/Bike-Helmet-Detection-Docs   <br />
5) https://drive.google.com/drive/folders/1M4FckJJeyPQTTWqgo6xWhW8L4tf0EJ4l?usp=sharing <br />