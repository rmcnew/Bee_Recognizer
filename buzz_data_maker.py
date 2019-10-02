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

data = []

def traverse_directory_for_sounds(path, buzz_setting):
    for dirpath, dirnames, filenames in os.walk(path):
        for matched_file in fnmatch.filter(filenames, "*.wav"):
            full_path_file = os.path.join(dirpath, matched_file)
            print("Processing {}".format(full_path_file))
            wav = np.array(load_sound(full_path_file))
            data.append((wav, buzz_setting))

