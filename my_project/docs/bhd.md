# Bike Rider Helmet Detection using YOLOv8

This repository contains a walk through of implementation of bike rider helmet detection using YOLOv8. As we know that bike riders who do not wear helmet may which result in fatal accidents and death in some cases. The goal is to create a ML/DL model that can detect if a person is wearing helment or not. May sure to select GPU in google colab or kaggle notebook for running through the notebook. As running through the code in local machine with bo graphical processing unit will take a very significant ammount of time. <br />

### Methodology
1) Data Collection:- Collect images/videos of bike rider, helmet and no helmet for the model to train upon. <br />
2) Data pre-processing:- Pre-processing and autolabel images and videos using foundation model like DINO and SAM (Segment anything model).  <br />
3) Train YOLOv8 Model. <br />
4) Evaluate Target Model.   <br />
5) Run Inference on images and videos.  <br />

# How to setup codebase locally?
```shell
python3 -m venv .venv
source .venv/bin/activate
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
│           ├── args.yaml
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

# Also see
1) https://github.com/Viddesh1/Helmet_test_1    <br />
2) https://github.com/Viddesh1/Bike-Helmet-Detection    <br />
3) https://github.com/Viddesh1/Bike-Helmet-Detectionv2  <br />
4) https://github.com/Viddesh1/Bike-Helmet-Detection-Docs   <br />
5) https://drive.google.com/drive/folders/1M4FckJJeyPQTTWqgo6xWhW8L4tf0EJ4l?usp=sharing <br />