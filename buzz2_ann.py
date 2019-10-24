import numpy as np
import tflearn
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
# tflearn.image_preloader does not work.  It just throws exceptions.
# So I wrote my own custom image preloader and flattened / standardized
# the layout of the BEE and BUZZ datasets 
from buzz_data_maker import *

script_name = os.path.basename(__file__)
name = os.path.splitext(script_name)[0]
print("Running {}".format(name))

#training, testing, validation = load_buzz2()
training, testing = load_buzz2()

training_X, training_Y = training
testing_X, testing_Y = testing

# free memory
training = None
testing = None

# SAMPLE_RATE is defined in buzz_data_maker
training_X = np.reshape(training_X, (-1, SAMPLE_RATE, 1))
testing_X = np.reshape(testing_X, (-1, SAMPLE_RATE, 1))

training_Y = np.reshape(training_Y, (-1, 3))
testing_Y = np.reshape(testing_Y, (-1, 3))

def create_model():
    net = input_data(shape=[None, SAMPLE_RATE, 1])
    net = fully_connected(net, SAMPLE_RATE, activation='relu')
    net = fully_connected(net, 512, activation='relu')
    net = fully_connected(net, 256, activation='relu')
    net = fully_connected(net, 3, activation='softmax')
    net = regression(net, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy')
    model = tflearn.DNN(net)
    return model

def load_model(save_file):
    model = create_model()
    model.load(save_file)
    return model

def do_training():
    model = create_model()
    model.fit(training_X, training_Y, n_epoch=100, batch_size=10, shuffle=True, validation_set=(testing_X, testing_Y), show_metric=True, run_id="{}_training".format(name))
    model.save("{}_model/{}_model".format(name, name))


if __name__ == '__main__':
    do_training()
