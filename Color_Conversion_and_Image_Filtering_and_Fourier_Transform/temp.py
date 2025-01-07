import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_image(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

def save_image(file_name, image_array, cmap=None):
    if cmap:
        plt.imsave(file_name, image_array, cmap=cmap)
    else:
        plt.imsave(file_name, image_array)
        
def create_reference_images(image_array):
    plt.imsave('hsv_image_reference.png', cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV))
    plt.imsave('lab_image_reference.png', cv2.cvtColor(image_array, cv2.COLOR_RGB2Lab))