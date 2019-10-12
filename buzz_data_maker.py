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
from scipy.io import wavfile

# buzz definitions
BEE = np.array([1, 0, 0])
CRICKET = np.array([0, 1, 0])
NOISE = np.array([0, 0, 1])

# relative paths; we assume that everything is running in the top-level Bee_Recognizer directory
current_dir = os.getcwd()
datasets = os.path.join(current_dir, 'datasets')
# buzz1 paths
buzz1 = os.path.join(datasets, 'BUZZ1')
buzz1_training = os.path.join(buzz1, 'training')
buzz1_testing = os.path.join(buzz1, 'testing')
# BUZZ1 does not have a validation group

# buzz2 paths
buzz2 = os.path.join(datasets, 'BUZZ2')
buzz2_training = os.path.join(buzz2, 'training')
buzz2_testing = os.path.join(buzz2, 'testing')
buzz2_validation = os.path.join(buzz2, 'validation')

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


def load_sound(sound_path):
    samplerate, audio = wavfile.read(sound_path)
    scaled_audio = audio/float(np.max(audio))
    return scaled_audio


def traverse_directory_for_sounds(path, buzz_setting, data):
    for dirpath, dirnames, filenames in os.walk(path):
        for matched_file in fnmatch.filter(filenames, "*.wav"):
            full_path_file = os.path.join(dirpath, matched_file)
            print("Processing {}".format(full_path_file))
            wav = np.array(load_sound(full_path_file))
            data.append((wav, buzz_setting))
    return data

def load_buzz_dataset(path):
    sound_data = []
    labels = []
    data = (sound_data, labels)
    bee_path = os.path.join(path, 'bee')
    cricket_path = os.path.join(path, 'cricket')
    noise_path = os.path.join(path, 'noise')
    data = traverse_directory_for_images(bee_path, BEE, data) 
    data = traverse_directory_for_images(cricket_path, CRICKET, data) 
    data = traverse_directory_for_images(noise_path, NOISE, data) 
    return data

def load_buzz1():
   training = load_buzz_dataset(buzz1_training)
   testing = load_buzz_dataset(buzz1_testing)
   return training, testing

def load_buzz2():
   training = load_buzz_dataset(buzz2_training)
   testing = load_buzz_dataset(buzz2_testing)
   validation = load_buzz_dataset(buzz2_validation)
   return training, testing, validation

