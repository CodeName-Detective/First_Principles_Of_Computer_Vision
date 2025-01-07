'''
Notes:
1. All of your implementation for task 1 should be in this file. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''

import cv2
import numpy as np
import argparse
import json
import os
import sys
import math


from typing import Dict, List
from utils import show_image


'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: np.ndarray) -> List[List[float]]:
    """
    Args:
        img : input image is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    detection_results: List[List[float]] = [] # Please make sure your output follows this data format.

    # Add your code here. Do not modify the return and input arguments.

    # Load the Haar Cascade Classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert image to grayscale as Haar Cascade works on grayscale images
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    detection_results = [[float(x), float(y), float(w), float(h)] for (x, y, w, h) in faces]
    
    return detection_results


def parse_args():
    parser = argparse.ArgumentParser(description="cse 573 homework 4.")
    parser.add_argument(
        "--input_path", type=str, default="data/validation_folder/images",
        help="path to validation or test folder")
    parser.add_argument(
        "--output", type=str, default="./result_task1.json",
        help="path to the characters folder")

    args = parser.parse_args()
    return args

def save_results(result_dict, filename):
    results = []
    results = result_dict
    with open(filename, "w") as file:
        json.dump(results, file, indent=4)

def check_output_format(faces, img, img_name):
    if not isinstance(faces, list):
        print('Wrong output type for image %s! Should be a %s, but you get %s.' % (img_name, list, type(faces)))
        return False
    for i, face in enumerate(faces):
        if not isinstance(face, list):
            print('Wrong bounding box type in image %s the %dth face! Should be a %s, but you get %s.' % (img_name, i, list, type(face)))
            return False
        if not len(face) == 4:
            print('Wrong bounding box format in image %s the %dth face! The length should be %s , but you get %s.' % (img_name, i, 4, len(face)))
            return False
        for j, num in enumerate(face):
            if not isinstance(num, float):
                print('Wrong bounding box type in image %s the %dth face! Should be a list of %s, but you get a list of %s.' % (img_name, i, float, type(num)))
                return False
        if face[0] >= img.shape[1] or face[1] >= img.shape[0] or face[0] + face[2] >= img.shape[1] or face[1] + face[3] >= img.shape[0]:
            print('Warning: Wrong bounding box in image %s the %dth face exceeds the image size!' % (img_name, i))
            print('One possible reason of this is incorrect bounding box format. The format should be [topleft-x, topleft-y, box-width, box-height] in pixels.')
    return True


def batch_detection(img_dir):
    res = {}
    for img_name in sorted(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detect_faces(img)
        if not check_output_format(faces, img, img_name):
            print('Wrong output format!')
            sys.exit(2)
        res[img_name] = faces
    return res

def main():

    args = parse_args()
    path, filename = os.path.split(args.output)
    os.makedirs(path, exist_ok=True)
    result_list = batch_detection(args.input_path)
    save_results(result_list, args.output)

if __name__ == "__main__":
    main()

    