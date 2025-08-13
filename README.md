# Speaker Detection
Effective speaker detection inference of LoCoNet and LASER models  
  
🔥 **Improvements**  
  - Fast face detection with yolov8-face  
  - Video decoding on gpu  
  - Optimized processing with FFMPEG for fast and accurate result  
  
🔥 **Fast and effective inference pipeline of speaker detection algorithm**  
  - Inference is more than 100x times faster  
  - Processing video of any size  
  
  
### :hammer_and_wrench: Install  
  0. Clone repository:
    git clone https://github.com/ShJacub/SpeakerDetection speaker_detection
  1. Set paths in bash_scripts/docker_create.sh and create and run docker container  
    bash bash_scripts/docker_create.sh  
  2. Install libs and packages  
    bash speaker_detection/bash_scripts/install.sh  


### :luggage: Download Weights  
  1. yolov8-face - https://github.com/derronqi/yolov8-face  
  2. face_landmarker_v2_with_blendshapes.task  
  3. Speaker Detection Models  
    3.1. LoCoNet (if you will use LoCoNet model) - https://github.com/SJTUwxz/LoCoNet_ASD  
    3.2. LASER (if you will user LoCoNet_LASER model) - https://github.com/plnguyen2908/LASER_ASD  
Place weights into folder 'weights'  

### :bookmark_tabs: Config
Set configs if you need. Default settings are OK :thumbsup:  
Configs:  
  - main.yaml - main config file which contain video path, temporal directory paths, ffmpeg process parameters, etc.  
  - yolo.cfg - yolov8 settings file  
  - asd.yaml - speaker detector model settings file  


### :arrow_forward: Inference  
  python3 run.py  


# Acknowledgement
[LASER_ASD](https://github.com/plnguyen2908/LASER_ASD)

[LoCoNet_ASD](https://github.com/SJTUwxz/LoCoNet_ASD)

[yolov8-face](https://github.com/derronqi/yolov8-face)

[VideoProcessingFramework](https://github.com/NVIDIA/VideoProcessingFramework)