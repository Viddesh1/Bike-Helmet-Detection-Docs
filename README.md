# Bike-Helmet-Detection-Docs

This repository represents all the overall information related to the project naming **Bike Rider Helmet Detection** using YOLOv8 (You Only Look Once version 8) from ultralytics for detection of **bike rider**, **helmet** and **no helmet** objects which is trained on custom public dataset.
<br />

This repository provides documentation that are not only respective code respository specific but also documentation or information that other repositories has but cannot be merged together as each repository has there own implementation of respective technology for example jupyter notebook, django, streamlit etc. Respective repository has there own independent implementation. As a result this respository acts as a union of all the respositories related to this project. The scope of this repository is only limited to documentation. This repository also contains readme.md file from below projects <br />

This repository acts as a connection link between below projects present in github :-   <br />

1) https://github.com/Viddesh1/Helmet_test_1.git
2) https://github.com/Viddesh1/Bike-Helmet-Detection.git
3) https://github.com/Viddesh1/Bike-Helmet-Detectionv2.git

# How to setup this repository codebase locally?
```shell
python3 -m venv .venv
source .venv/bin/activate
git clone https://github.com/Viddesh1/Bike-Helmet-Detection-Docs.git
cd Bike-Helmet-Detection-Docs/
pip install -r requirements.txt
cd my_project/
mkdocs serve
```

# Project structure
```text
Bike-Helmet-Detection-Docs
├── .git
├── .github
│   └── workflows
│       └── page_deploy.yml
├── .gitignore
├── my_project
│   ├── docs
│   │   ├── bhd.md
│   │   ├── dj_bhd_images
│   │   │   ├── ml_app_image.png
│   │   │   ├── ml_app_predimage.png
│   │   │   ├── predicted.jpg
│   │   │   ├── Screenshot_1.png
│   │   │   ├── Screenshot_2.png
│   │   │   ├── Screenshot_3.png
│   │   │   ├── Screenshot_4.png
│   │   │   ├── Screenshot_5.png
│   │   │   └── Screenshot_6.png
│   │   ├── dj_bhd.md
│   │   ├── dj_bhd_videos
│   │   │   └── Traffic_2.mp4
│   │   ├── index.md
│   │   ├── st_bhd_images
│   │   │   ├── st_devicecam_pred.png
│   │   │   ├── st_image_pred.png
│   │   │   ├── st_video_pred.png
│   │   │   ├── st_webcam_pred.png
│   │   │   └── st_yt_pred.png
│   │   ├── st_bhd.md
│   │   └── yolov8_images
│   │       ├── confusion_matrix.png
│   │       ├── P_curve.png
│   │       ├── results.png
│   │       ├── train_batch0.jpg
│   │       ├── train_batch1.jpg
│   │       ├── train_batch2.jpg
│   │       ├── val_batch0_labels.jpg
│   │       ├── val_batch0_pred.jpg
│   │       ├── val_batch1_labels.jpg
│   │       ├── val_batch1_pred.jpg
│   │       ├── val_batch2_labels.jpg
│   │       ├── val_batch2_pred.jpg
│   │       └── yolov8_metrics.png
│   └── mkdocs.yml
├── README.md
├── requirements.txt
└── tree_all.txt
```
