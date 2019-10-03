#!/usr/bin/python3

################################################################
# Richard Scott McNew
# A02077329
################################################################

import os
import fnmatch
import pickle as cPickle
import gzip
import numpy as np
import cv2


# bee definitions
BEE = np.reshape(np.array([1, 0]), (2, 1))
NO_BEE = np.reshape(np.array([0, 1]), (2, 1))


#### Miscellaneous functions
# save and load pickle functions grabbed from hw02
def save_pickle(ann, file_name):
    with open(file_name, 'wb') as fp:
        cPickle.dump(ann, fp)

# restore() function to restore the file
def load_pickle(file_name):
    with open(file_name, 'rb') as fp:
        nn = cPickle.load(fp)
    return nn

def load_image(image_path, expected_width, expected_height):
    expected_length = expected_width * expected_height
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    if width != expected_width or height != expected_height:
        image = cv2.resize(image, (expected_width, expected_height), interpolation=cv2.INTER_CUBIC)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scaled_gray_image = gray_image/255.0
    shaped = np.reshape(scaled_gray_image, (expected_length, 1))  
    return shaped

data = []

def traverse_directory_for_images(path, bee_setting, expected_width, expected_height):
    for dirpath, dirnames, filenames in os.walk(path):
        for matched_file in fnmatch.filter(filenames, "*.png"):
            full_path_file = os.path.join(dirpath, matched_file)
            print("Processing {}".format(full_path_file))
            img = np.array(load_image(full_path_file, expected_width, expected_height))
            data.append((img, bee_setting))

