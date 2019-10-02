#!/usr/bin/python3

import pickle
import gzip
import numpy as np

def load_bee1_data():
    f = gzip.open('bee1.pck.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)
