#!/usr/bin/python3

import pickle
import gzip
import numpy as np

def load_bee1_data():
    f = gzip.open('bee1.pck.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_bee2_1S_data():
    f = gzip.open('bee2_1S.pck.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)


def load_bee2_2S_data():
    f = gzip.open('bee2_2S.pck.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)
