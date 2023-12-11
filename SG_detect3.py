# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs object detection on camera frames using OpenCV. 

TEST_DATA=../all_models

Run face detection model:
python3 detect.py \  

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt
"""
# this code is a blend of detect.py from google opencv camera examples https://coral.ai.docs and wpilibpi_vision.py
# scraped from https://www.chiefdelphi.com/t/what-happen-to-ai-and-axon-for-frc-removed-from-wpilib-docs-2023/422713/14
# it also includes some network tables data writing code from WPILibPi basic viion example  
# It is written to run on top of the WPILibPi raspberry pi image with updates - see google doc link above
# Note that WPILibPi image has no NTP client.  The updates and install scripts do not work without setting time and date.
# This code has been tested the WPILibPi from Feb 2023.  No 2024 (beta) version as of 12/11/23
# Runs 30 FPS from MS camera at 1280 X 720.  Inference engine is 300 X 300
# Camera resolution changes does not seem to effect FPS 
# camera configuration for image size and FPS is in /boot/frc.json

# imports added from wpilibpi_vision.py
# basic intent here is to replace the detect.py camera interface with CameraServer

from cscore import CameraServer
import ntcore
import re   #regular expression lib - may not need this - was in wpilibpi_vision

import cv2
import json
import numpy as np
import time

#imports below here are from detect.py

import argparse
import os
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

#from wpilibpi_vision.py
team = 2342				#on robot must match robot team number
server = True     #set False when deployed on robot - Rio hosts the server

def main():
	# code that sets up argument parsing - from google detect.py
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    args = parser.parse_args()

    # Code that initializes the tflite model  
    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)
    
    print('inference_size',inference_size) # inference_size is (300, 300)

    
    #camera and network tables code adapted from wpilibpi_vision.py
    
    with open('/boot/frc.json') as f:
        config = json.load(f)
    camera = config['cameras'][0]

    width = camera['width']
    height = camera['height']

    CameraServer.startAutomaticCapture()

    input_stream = CameraServer.getVideo()
    output_stream = CameraServer.putVideo('Processed', width, height)
    img = np.zeros(shape=(height, width, 3), dtype=np.uint8)

    # Table for vision output information
    # start NetworkTables
    ntinst = ntcore.NetworkTableInstance.getDefault()
    
    if server:
        print("Setting up NetworkTables server")
        ntinst.startServer()
    else:
        print("Setting up NetworkTables client for team {}".format(team))
        ntinst.startClient4("wpilibpi")
        ntinst.setServerTeam(team)
        ntinst.startDSClient()
    #vision_nt = NetworkTables.getTable('Vision')
    
    vision_nt = ntinst.getTable('Vision')        #For some reason this does not define vision_nt  
   

    # Load your model onto the TF Lite Interpreter - this code from wpilibpi_vision 
    #interpreter.allocate_tensors()
    #labels = read_label_file(labelPath)

    # Wait for NetworkTables to start
    time.sleep(0.5)

    #vision_nt = ntinst.getTable('Vision')  #moving this down here did not help
    
    
    prev_time = time.time()

    """while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
    """
    while True:
        start_time = time.time()

        frame_time, input_img = input_stream.grabFrame(img)
        output_img = np.copy(input_img)

        # Notify output of error and skip iteration
        if frame_time == 0:
            output_stream.notifyError(input_stream.getError())
            continue
            
        #cv2_im_rgb = cv2.resize(input_image, inference_size) 
        cv2_im_rgb = cv2.resize(input_img, inference_size)         
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, args.threshold)[:args.top_k]
        #cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels)
        output_img = append_objs_to_img(output_img, inference_size, objs, labels, vision_nt)

    
        # compute frames/sec
        processing_time = start_time - prev_time
        prev_time = start_time

        fps = 1 / processing_time
        cv2.putText(output_img, str(round(fps, 1)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        output_stream.putFrame(output_img)

def append_objs_to_img(output_img, inference_size, objs, labels, vision_nt):	
    x_list = []
    height, width, channels = output_img.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
        
        print (x0, y0, x1, y1, label, obj.id)  #may not want this when deployed on robot
                
        x_list.append(x0)
        x_list.append(y0)
        x_list.append(x1)
        x_list.append(y1)
        x_list.append(obj.id)
        x_list.append(obj.score)
       
        
        # probably do something like this to send data to NT
        # there are going to be very few objects we care about
        # sendign an array of numbers will improve coherency and efficiency
        #vision_nt.putNumberArray('target_data', [x0, y0, x1, y1, obj.id, obj.score])  #this works outline viewer sees 1 record
        # or x-center, y-center, box area, obj.id, obj.score
        # we should make the data as friendly to the bot drive function as possible
        #vision_nt.putNumberArray('target_data', x_list) #also works outline viewer sees varying # of records

        #cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        #cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
        output_img = cv2.rectangle(output_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        output_img = cv2.putText(output_img, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    
    vision_nt.putNumberArray('target_data', x_list)  #outline viewer sees records for all objects detected (max 3))
    return output_img

if __name__ == '__main__':
    main()
