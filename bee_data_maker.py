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
BEE = np.array([1, 0])
NO_BEE = np.array([0, 1])

# image dimensions
BEE1_DIMENSIONS = (32, 32) 
BEE2_DIMENSIONS = (90, 90)

# relative paths; we assume that everything is running in the top-level Bee_Recognizer directory
current_dir = os.getcwd()
datasets = os.path.join(current_dir, 'datasets')
# bee1 paths
bee1 = os.path.join(datasets, 'BEE1')
bee1_training = os.path.join(bee1, 'training')
bee1_testing = os.path.join(bee1, 'testing')
bee1_validation = os.path.join(bee1, 'validation')

# bee2_1s paths
bee2_1s = os.path.join(datasets, 'BEE2_1S')
bee2_1s_training = os.path.join(bee2_1s, 'training')
bee2_1s_testing = os.path.join(bee2_1s, 'testing')
bee2_1s_validation = os.path.join(bee2_1s, 'validation')

# bee2_2s paths
bee2_2s = os.path.join(datasets, 'BEE2_2S')
bee2_2s_training = os.path.join(bee2_2s, 'training')
bee2_2s_testing = os.path.join(bee2_2s, 'testing')
bee2_2s_validation = os.path.join(bee2_2s, 'validation')


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

# expected_dimensions is a tuple (width, height)
def load_image(image_path, expected_dimensions):
    expected_width = expected_dimensions[0]
    expected_height = expected_dimensions[1]
    expected_length = expected_width * expected_height
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    if width != expected_width or height != expected_height:
        image = cv2.resize(image, (expected_width, expected_height), interpolation=cv2.INTER_CUBIC)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scaled_gray_image = gray_image/255.0
    return scaled_gray_image


# data should be a tuple of two lists: (image_data, label)
def traverse_directory_for_images(path, bee_setting, expected_dimensions, data):
    for dirpath, dirnames, filenames in os.walk(path):
        for matched_file in fnmatch.filter(filenames, "*.png"):
            full_path_file = os.path.join(dirpath, matched_file)
            # print("Processing {}".format(full_path_file))
            img = np.array(load_image(full_path_file, expected_dimensions))
            data[0].append(img)
            data[1].append(bee_setting)
    return data


def load_bee_dataset(path, expected_dimensions):
    image_data = []
    labels = []
    data = (image_data, labels)
    bee_path = os.path.join(path, 'bee')
    no_bee_path = os.path.join(path, 'no_bee')
    data = traverse_directory_for_images(bee_path, BEE, expected_dimensions, data) 
    data = traverse_directory_for_images(no_bee_path, NO_BEE, expected_dimensions, data) 
    return data

def load_bee1():
    training = load_bee_dataset(bee1_training, BEE1_DIMENSIONS) 
    testing = load_bee_dataset(bee1_testing, BEE1_DIMENSIONS) 
    validation = load_bee_dataset(bee1_validation, BEE1_DIMENSIONS) 
    return training, testing, validation

def load_bee2_1s():
    training = load_bee_dataset(bee2_1s_training, BEE2_DIMENSIONS) 
    testing = load_bee_dataset(bee2_1s_testing, BEE2_DIMENSIONS) 
    validation = load_bee_dataset(bee2_1s_validation, BEE2_DIMENSIONS) 
    return training, testing, validation

def load_bee2_2s():
    training = load_bee_dataset(bee2_2s_training, BEE2_DIMENSIONS) 
    testing = load_bee_dataset(bee2_2s_testing, BEE2_DIMENSIONS) 
    validation = load_bee_dataset(bee2_2s_validation, BEE2_DIMENSIONS) 
    return training, testing, validation


