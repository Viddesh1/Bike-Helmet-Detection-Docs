# Bike-Helmet-Detection-Docs

This repository represents all the overall information related to the project naming **Bike Rider Helmet Detection** using YOLOv8 (You Only Look Once version 8) from ultralytics for detection of **bike rider**, **helmet** and **no helmet** objects which is trained on custom public dataset.
<br />

This repository provides documentation that are not only respective code respository specific but also documentation or information that other repositories has but cannot be merged together as each repository has there own implementation of respective technology for example jupyter notebook, django, streamlit etc. Respective repository has there own independent implementation. As a result this respository acts as a union of all the respositories related to this project. The scope of this repository is only limited to documentation. This repository also contains readme.md file from below projects <br />

This repository acts as a connection link between below projects present in github :-   <br />

1) https://github.com/Viddesh1/Helmet_test_1.git    <br />
2) https://github.com/Viddesh1/Bike-Helmet-Detection.git    <br />
3) https://github.com/Viddesh1/Bike-Helmet-Detectionv2.git  <br />
4) https://drive.google.com/drive/folders/1M4FckJJeyPQTTWqgo6xWhW8L4tf0EJ4l?usp=sharing <br />

This project naming **Bike Rider Helmet Detection** is created in such a way where every code respository has there specific implementation as a result they adher to microservice architecture.    <br />

1) https://github.com/Viddesh1/Helmet_test_1.git    <br />

This repository has google colab **jupyter notebook implementation + Result in Google Drive.** <br />


2) https://github.com/Viddesh1/Bike-Helmet-Detection.git <br />

This repository contains the implementation of Bike helmet detection **django web application** using YOLO8. <br />

3) https://github.com/Viddesh1/Bike-Helmet-Detectionv2.git <br />

This repository contains the implementation of Bike helmet detection **streamlit web application** using YOLO8. <br />

4) https://drive.google.com/drive/folders/1M4FckJJeyPQTTWqgo6xWhW8L4tf0EJ4l?usp=sharing <br />

The above google drive link represents output that are generated by YOLOv8. <br />

# Major python librarie used for this project
```
mkdocs==1.6.0
```

# How to setup this repository codebase locally?
```shell
python3 -m venv .venv
source .venv/bin/activate
git clone https://github.com/Viddesh1/Bike-Helmet-Detection-Docs.git
cd Bike-Helmet-Detection-Docs/
pip install -r requirements.txt
cd my_project/
mkdocs serve # Runs application on local host
```

# Project structure
```text
Bike-Helmet-Detection-Docs # This github root folder
├── .git    # Git for version control
├── .github # Github for remote version control
│   └── workflows
│       └── page_deploy.yml # Automated workflow for deploying to a github branch
├── .gitignore  # Removing unncessary file not needed for version control of python
├── my_project  # mkdocs project root folder
│   ├── docs    
│   │   ├── bhd.md  # Bike Rider Helmet Detection Jupyter notebook readme file
│   │   ├── dj_bhd_images   # Demo images from django webapp of BRHD 
│   │   │   ├── ml_app_image.png
│   │   │   ├── ml_app_predimage.png
│   │   │   ├── predicted.jpg
│   │   │   ├── Screenshot_1.png
│   │   │   ├── Screenshot_2.png
│   │   │   ├── Screenshot_3.png
│   │   │   ├── Screenshot_4.png
│   │   │   ├── Screenshot_5.png
│   │   │   └── Screenshot_6.png
│   │   ├── dj_bhd.md   # Django BRHD documentation
│   │   ├── dj_bhd_videos   # Django BRHD video
│   │   │   └── Traffic_2.mp4
│   │   ├── docs_images # Screen capture of this document mkdocs itself
│   │   │   ├── docs_bhd.png
│   │   │   └── docs_index.png
│   │   ├── index_2.md  # Cut-Down version of BRHD document
│   │   ├── index.md    # Document as per project pdf
│   │   ├── Jupyter_notebook_op # Sample code and output screen capture
│   │   │   ├── sample_code_10.png
│   │   │   ├── sample_code_11.png
│   │   │   ├── sample_code_12.png
│   │   │   ├── sample_code_1.png
│   │   │   ├── sample_code_2.png
│   │   │   ├── sample_code_3.png
│   │   │   ├── sample_code_4.png
│   │   │   ├── sample_code_5.png
│   │   │   ├── sample_code_6.png
│   │   │   ├── sample_code_7.png
│   │   │   ├── sample_code_8.png
│   │   │   └── sample_code_9.png
│   │   ├── st_bhd_images # Streamlit webapp demo screen capture
│   │   │   ├── st_devicecam_pred.png
│   │   │   ├── st_image_pred.png
│   │   │   ├── st_video_pred.png
│   │   │   ├── st_webcam_pred.png
│   │   │   └── st_yt_pred.png
│   │   ├── st_bhd.md   # Streamlit documentation
│   │   ├── yolov8_architecture # YOLOv8 Architecture images
│   │   │   ├── best.pt.png # best.pt pytorch model architecture open in netron.app for more detail
│   │   │   ├── last.pt.png # last.pt pytorch model architecture open in netron.app for more detail
│   │   │   └── YOLOv8_architecture.jpg # YOLOv8 Architecture image by Rangking from github
│   │   └── yolov8_images   # YOLOv8 result from google drive generated by jupyter notebook
│   │       ├── All_Video_Pred.png
│   │       ├── confusion_matrix.png
│   │       ├── F1_curve.png
│   │       ├── labels_correlogram.jpg
│   │       ├── labels.jpg
│   │       ├── P_curve.png
│   │       ├── PR_curve.png
│   │       ├── R_curve.png
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
│   └── mkdocs.yml  # Configuration file for mkdocs
├── README.md # This readme file itself
├── requirements.txt    # Necessary dependencies and requirements
└── tree_all.txt # Tree Structure generated by command tree -a > tree_all.txt
```

# Github Branches Information
```
main : Contains Main codes related to this repository
gh-pages : Automatically generated by github workflow actions to be hosted on github-pages
```

# MKdocs Documentation hosted on Github-pages

https://viddesh1.github.io/Bike-Helmet-Detection-Docs/

# Also see

1) https://github.com/Viddesh1/Helmet_test_1
2) https://github.com/Viddesh1/Bike-Helmet-Detection
3) https://github.com/Viddesh1/Bike-Helmet-Detectionv2
4) https://github.com/Viddesh1/Bike-Helmet-Detection-Docs
5) https://drive.google.com/drive/folders/1M4FckJJeyPQTTWqgo6xWhW8L4tf0EJ4l?usp=sharing
6) https://github.com/Viddesh1/Bike-Helmet-Detection/wiki
