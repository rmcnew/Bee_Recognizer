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
BEE = [1, 0]
NO_BEE = [0, 1]

data = []

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

# Image and Sound loading functions
def load_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scaled_gray_image = gray_image/255.0
    shaped = np.reshape(scaled_gray_image, (1024, 1))  # 32 * 32 = 1024
    return shaped


def traverse_directory_for_images(path, bee_setting):
    for dirpath, dirnames, filenames in os.walk(path):
        for matched_file in fnmatch.filter(filenames, "*.png"):
            full_path_file = os.path.join(dirpath, matched_file)
            print("Processing {}".format(full_path_file))
            img = load_image(full_path_file)
            data.append([img, bee_setting])

