# Bike Rider Helmet Detection using YOLOv8

This repository contains a walk through of implementation of bike rider helmet detection using YOLOv8. As we know that bike riders who do not wear helmet may which result in fatal accidents and death in some cases. The goal is to create a ML/DL model that can detect if a person is wearing helment or not. May sure to select GPU for running through the notebook.  <br />

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
Helmet_test_1
├── .git
├── .gitignore
├── Helmet_how_to_auto_train_yolov8_model_with_autodistill.ipynb
├── kaggle_Helmet_how_to_auto_train_yolov8_model_with_autodistill.ipynb
├── README.md
└── tree_all.txt
```

## This repository output may change in the near future:-

https://drive.google.com/drive/folders/1M4FckJJeyPQTTWqgo6xWhW8L4tf0EJ4l?usp=sharing


## Output File Structure for YOLOv8
```text
YOLOv8_Helmet_V0
├── dataset
│   ├── annotations
│   ├── data.yaml
│   ├── images
│   ├── train
│   │   ├── images
│   │   │   ├── BikesHelmets0.jpg
│   │   │   ├── BikesHelmets100.jpg
│   │   │   ├── BikesHelmets101.jpg
│   │   ├── labels
│   │   │   ├── BikesHelmets0.txt
│   │   │   ├── BikesHelmets100.txt
│   │   │   ├── BikesHelmets101.txt
│   │   └── labels.cache
│   └── valid
│       ├── images
│       │   ├── BikesHelmets103.jpg
│       │   ├── BikesHelmets108.jpg
│       │   ├── BikesHelmets119.jpg
│       ├── labels
│       │   ├── BikesHelmets103.txt
│       │   ├── BikesHelmets108.txt
│       │   ├── BikesHelmets119.txt
│       └── labels.cache
├── images
│   ├── BikesHelmets0.png
│   ├── BikesHelmets100.png
│   ├── BikesHelmets101.png
├── runs
│   └── detect
│       ├── predict
│       │   └── he2.mp4
│       ├── predict2
│       │   └── test_1.mp4
│       ├── predict3
│       │   └── test_2.mp4
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
│           ├── results.csv
│           ├── results.png
│           ├── train_batch0.jpg
│           ├── train_batch1.jpg
│           ├── train_batch2.jpg
│           ├── val_batch0_labels.jpg
│           ├── val_batch0_pred.jpg
│           ├── val_batch1_labels.jpg
│           ├── val_batch1_pred.jpg
│           ├── val_batch2_labels.jpg
│           ├── val_batch2_pred.jpg
│           └── weights
│               ├── best.pt
│               └── last.pt
├── tree_all.txt
├── videos
│   ├── he2.mp4
│   ├── test_1.mp4
│   └── test_2.mp4
├── yolov8l.pt
└── yolov8n.pt
```

# Also see
1) https://github.com/Viddesh1/Bike-Helmet-Detection    <br />
2) https://github.com/Viddesh1/Bike-Helmet-Detectionv2  <br />