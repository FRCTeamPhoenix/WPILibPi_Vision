This repository contains parts of the code required for object detection on an RPi5 with a Hailo AI chip.
The yolo11n.hef and yolo11s.hef are compiled yolo11 object detection models trained to detect the yellow
balls in the 2026 FRC game.  They are specific to the Hailo AI acellerator chip. 

The RPi5 baseline environment for this code is at https://github.com/hailo-ai/hailo-rpi5-examples.
This repository is now updated. Newer baselines are at https://github.com/hailo-ai/hailo-apps.
The code and model files are likely not compatible with the newer baselines.

More information and links to Hailo tools used can be found at https://hailo.ai
Information on Yolo models is at https://docs.ultralytics.com/
